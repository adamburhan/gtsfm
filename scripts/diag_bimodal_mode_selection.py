"""Does the bimodal depth factor select the GT-correct surface?

For each bimodal-candidate measurement, back-projects the pixel at both d and
d_alt, checks which 3D point is closer to the GT mesh, and asks whether the
optimizer's mode selection agreed. No cross-run matching needed.

Usage:
    python scripts/diag_bimodal_mode_selection.py \\
        --ba_dir .../bimodal/results/ba_output \\
        --depth_map_dir .../Replica/office0/results \\
        --gt_ply .../Replica/office0_mesh.ply \\
        --gt_traj .../Replica/office0/traj.txt \\
        --gap_thresh 0.10
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gtsam
from gtsfm.common.depth_provider import DepthProvider
from gtsfm.common.gtsfm_data import GtsfmData


def build_image_fnames(data: GtsfmData) -> dict:
    return {i: n for i, n in zip(data.get_valid_camera_indices(), data.image_filenames()) if n}


def load_gt_surface(gt_ply: str, n_samples: int = 500_000) -> np.ndarray:
    import trimesh
    import trimesh.sample
    tm = trimesh.load(gt_ply, process=False, force="mesh")
    faces = getattr(tm, "faces", None)
    if faces is not None and len(faces) > 0:
        return np.asarray(trimesh.sample.sample_surface(tm, n_samples)[0])
    print("[mesh] No triangulated faces; using raw vertices.")
    return np.asarray(getattr(tm, "vertices"))


def build_gt_tree(gt_ply: str, gt_traj: str, data: GtsfmData):
    from scipy.spatial import cKDTree  # type: ignore[import-untyped]
    from gtsfm.evaluation.eval_geometry_vs_mesh import align_recon_to_world

    gt_pts = load_gt_surface(gt_ply)
    print(f"GT surface: {len(gt_pts):,} points")

    wTi_list = []
    for i in data.get_valid_camera_indices():
        cam = data.get_camera(i)
        wTi_list.append(cam.pose() if cam is not None else None)
    fnames = [data.get_image_info(i).name for i in data.get_valid_camera_indices()]

    wSr, rms = align_recon_to_world(wTi_list, fnames, gt_traj)
    print(f"Sim3 alignment RMS: {rms * 100:.2f} cm")
    return cKDTree(gt_pts), wSr


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ba_dir", required=True)
    p.add_argument("--depth_map_dir", required=True)
    p.add_argument("--gt_ply", required=True)
    p.add_argument("--gt_traj", required=True)
    p.add_argument("--gap_thresh", type=float, default=0.10)
    p.add_argument("--depth_scale", type=float, default=6553.5)
    p.add_argument("--depth_min", type=float, default=0.1)
    p.add_argument("--depth_max", type=float, default=20.0)
    p.add_argument("--patch_radius", type=int, default=5)
    p.add_argument("--depth_filename_template", default="depth{:06d}.png")
    args = p.parse_args()

    data = GtsfmData.read_colmap(args.ba_dir)
    print(f"{len(data.get_valid_camera_indices())} cameras, {data.number_tracks()} tracks")

    provider = DepthProvider(
        depth_map_dir=args.depth_map_dir,
        image_fnames=build_image_fnames(data),
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        depth_scale=args.depth_scale,
        depth_filename_template=args.depth_filename_template,
        compute_hypotheses=True,
        patch_radius=args.patch_radius,
        gap_thresh=args.gap_thresh,
        ambiguity_thresh=0.0,
        min_valid=3,
    )

    tree, wSr = build_gt_tree(args.gt_ply, args.gt_traj, data)

    records = []
    for j in range(data.number_tracks()):
        track = data.get_track(j)
        point_w = np.array(track.point3())
        for m_idx in range(track.numberMeasurements()):
            i, uv = track.measurement(m_idx)
            cam = data.get_camera(i)
            if cam is None:
                continue
            sample = provider.get_depth(i, float(uv[0]), float(uv[1]))
            if sample is None or not sample.ambiguous or sample.depth_alt is None:
                continue

            z_pred = float(cam.pose().transformTo(point_w)[2])
            opt_mode = 2 if abs(z_pred - sample.depth_alt) < abs(z_pred - sample.depth) else 1

            uv2 = gtsam.Point2(float(uv[0]), float(uv[1]))
            p_d     = np.array(cam.backproject(uv2, sample.depth))
            p_d_alt = np.array(cam.backproject(uv2, sample.depth_alt))
            dist_d     = float(tree.query(wSr.transformFrom(p_d),     k=1)[0])
            dist_d_alt = float(tree.query(wSr.transformFrom(p_d_alt), k=1)[0])
            gt_mode = 1 if dist_d < dist_d_alt else 2

            records.append({
                "gap":      abs(sample.depth - sample.depth_alt),
                "opt_mode": opt_mode,
                "gt_mode":  gt_mode,
                "correct":  opt_mode == gt_mode,
                "dist_d":     dist_d,
                "dist_d_alt": dist_d_alt,
            })

    n  = len(records)
    n1 = sum(r["opt_mode"] == 1 for r in records)
    n2 = sum(r["opt_mode"] == 2 for r in records)
    n_correct = sum(r["correct"] for r in records)

    print(f"\n{'='*60}")
    print(f"Bimodal candidates : {n}")
    print(f"  Mode 1 selected  : {n1}  ({100*n1/n:.1f}%)")
    print(f"  Mode 2 selected  : {n2}  ({100*n2/n:.1f}%)")
    print(f"  Picked GT-closer : {n_correct}/{n}  ({100*n_correct/n:.1f}%)")
    print(f"{'='*60}")

    mode2 = [r for r in records if r["opt_mode"] == 2]
    if mode2:
        n2_gt = sum(r["correct"] for r in mode2)
        print(f"\nOf {len(mode2)} mode-2 selections, d_alt was GT-closer: {n2_gt}/{len(mode2)}  ({100*n2_gt/len(mode2):.1f}%)")

    gaps    = np.array([r["gap"]     for r in records])
    correct = np.array([r["correct"] for r in records])
    q25, q50, q75 = np.percentile(gaps, [25, 50, 75])

    print(f"\nGap-stratified accuracy (|d - d_alt|):")
    print(f"  {'Range':<24}  {'n':>5}  {'% correct':>10}")
    for lo, hi, label in [
        (0,    q25,    f"Q1  ≤{q25*100:.0f} cm"),
        (q25,  q50,    f"Q2  {q25*100:.0f}–{q50*100:.0f} cm"),
        (q50,  q75,    f"Q3  {q50*100:.0f}–{q75*100:.0f} cm"),
        (q75,  np.inf, f"Q4  >{q75*100:.0f} cm"),
    ]:
        mask = (gaps >= lo) & (gaps < hi)
        if not mask.any():
            continue
        nc, nt = int(correct[mask].sum()), int(mask.sum())
        print(f"  {label:<24}  {nt:>5}  {100*nc/nt:>8.1f}%")


if __name__ == "__main__":
    main()
