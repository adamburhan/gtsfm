"""Diagnostic: does the bimodal depth factor ever select its second hypothesis?

After a completed GTSfM run with depth_model=bimodal, re-runs the DepthProvider
on the final ba_output to identify which measurements were bimodal candidates
(sample.ambiguous=True), then checks which mode the optimizer converged to by
comparing z_pred (final camera-frame depth from optimized cameras + points)
against d (primary hypothesis) and d_alt (secondary hypothesis).

Answers two questions from the post-hoc analysis:
  1. Does the optimizer ever end up at the non-naive mode (mode 2)?
  2. When it does, does that landmark land closer to the GT surface?

Usage:
    python scripts/diag_bimodal_mode_selection.py \\
        --ba_dir /path/to/results/ba_output \\
        --depth_map_dir /path/to/Replica/office0/results \\
        --gap_thresh 0.05 \\
        [--gt_ply /path/to/office0_mesh.ply] \\
        [--gt_traj /path/to/Replica/office0/traj.txt]

Authors: Adam Burhan
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

# Allow running from repo root without install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gtsfm.common.depth_provider import DepthProvider
from gtsfm.common.gtsfm_data import GtsfmData


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _build_image_fnames(data: GtsfmData) -> dict[int, str]:
    """Map GTSfM camera index → image filename from the COLMAP images.txt."""
    indices = data.get_valid_camera_indices()
    names = data.image_filenames()
    return {idx: name for idx, name in zip(indices, names) if name is not None}


def _analyze(data: GtsfmData, provider: DepthProvider) -> dict:
    """Check mode selection for every bimodal-candidate measurement.

    Returns a dict with summary stats and per-measurement arrays.
    """
    n_candidates = 0
    n_mode1 = 0
    n_mode2 = 0

    # Per bimodal-candidate: (d, d_alt, z_pred, mode_selected)
    records = []

    for j in range(data.number_tracks()):
        track = data.get_track(j)
        point_w = np.array(track.point3())

        for m_idx in range(track.numberMeasurements()):
            i, uv = track.measurement(m_idx)
            camera = data.get_camera(i)
            if camera is None:
                continue

            sample = provider.get_depth(i, float(uv[0]), float(uv[1]))
            if sample is None or not sample.ambiguous:
                continue

            d, d_alt = sample.depth, sample.depth_alt
            if d_alt is None:
                continue

            n_candidates += 1

            # Mirror the factor's computation exactly.
            point_c = camera.pose().transformTo(point_w)
            z_pred = float(point_c[2])

            r1 = abs(z_pred - d)
            r2 = abs(z_pred - d_alt)
            mode = 2 if r2 < r1 else 1

            if mode == 1:
                n_mode1 += 1
            else:
                n_mode2 += 1

            records.append(
                {
                    "track_idx": j,
                    "cam_idx": i,
                    "d": d,
                    "d_alt": d_alt,
                    "z_pred": z_pred,
                    "r1": r1,
                    "r2": r2,
                    "mode": mode,
                    "point_w": point_w.copy(),
                }
            )

    return {
        "n_candidates": n_candidates,
        "n_mode1": n_mode1,
        "n_mode2": n_mode2,
        "records": records,
    }


def _mesh_distances(points_w: np.ndarray, gt_ply: str, gt_traj: str, data: GtsfmData) -> np.ndarray:
    """Return point-to-GT-surface distances for each row of points_w (world frame).

    Performs the same Sim3 alignment as eval_geometry_vs_mesh.py so that the
    reconstruction's world frame matches the GT mesh frame.
    """
    try:
        import trimesh
        from scipy.spatial import cKDTree

        import open3d as o3d

        from gtsfm.evaluation.eval_geometry_vs_mesh import align_recon_to_world, load_gt_points
    except ImportError as e:
        print(f"[mesh] Skipping mesh comparison — missing dependency: {e}")
        return np.full(len(points_w), np.nan)

    wTi_list = [data.get_camera(i).pose() if data.get_camera(i) is not None else None
                for i in data.get_valid_camera_indices()]
    img_fnames = [data.get_image_info(i).name for i in data.get_valid_camera_indices()]

    try:
        wSr, rms = align_recon_to_world(wTi_list, img_fnames, gt_traj)
        print(f"[mesh] Sim3 alignment RMS camera-centre residual: {rms * 100:.2f} cm")
    except Exception as e:
        print(f"[mesh] Alignment failed: {e}. Using identity.")
        import gtsam
        wSr = gtsam.Similarity3()

    aligned = np.array([wSr.transformFrom(p) for p in points_w])
    gt_pts = load_gt_points(gt_ply)
    tree = cKDTree(gt_pts)
    dists, _ = tree.query(aligned, k=1)
    return dists


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ba_dir", required=True,
                        help="COLMAP-format BA output dir (cameras.txt / images.txt / points3D.txt).")
    parser.add_argument("--depth_map_dir", required=True,
                        help="Directory containing per-image depth PNGs (e.g. Replica/<seq>/results).")
    parser.add_argument("--gap_thresh", type=float, default=0.05,
                        help="Log-depth gap threshold used during the sweep (default: 0.05).")
    parser.add_argument("--depth_scale", type=float, default=6553.5,
                        help="Divisor to convert raw uint16 depth to metres (default: 6553.5 for Replica).")
    parser.add_argument("--depth_min", type=float, default=0.1)
    parser.add_argument("--depth_max", type=float, default=20.0)
    parser.add_argument("--patch_radius", type=int, default=5)
    parser.add_argument("--depth_filename_template", default="depth{:06d}.png",
                        help="Template mapping trailing image-stem digits to depth filename.")
    # Optional mesh comparison.
    parser.add_argument("--gt_ply", default=None,
                        help="GT mesh PLY for point-to-surface comparison (optional).")
    parser.add_argument("--gt_traj", default=None,
                        help="Replica traj.txt for Sim3 alignment before mesh comparison (optional).")
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Load reconstruction                                                 #
    # ------------------------------------------------------------------ #
    print(f"Loading BA output from: {args.ba_dir}")
    data = GtsfmData.read_colmap(args.ba_dir)
    n_cams = len(data.get_valid_camera_indices())
    n_tracks = data.number_tracks()
    print(f"  {n_cams} cameras, {n_tracks} tracks")

    image_fnames = _build_image_fnames(data)
    missing = sum(1 for v in image_fnames.values() if v is None)
    if missing:
        print(f"  WARNING: {missing} cameras have no image filename; depth sampling will skip them.")

    # ------------------------------------------------------------------ #
    # Build DepthProvider (same settings as the sweep)                   #
    # ------------------------------------------------------------------ #
    provider = DepthProvider(
        depth_map_dir=args.depth_map_dir,
        image_fnames=image_fnames,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        depth_scale=args.depth_scale,
        depth_filename_template=args.depth_filename_template,
        compute_hypotheses=True,
        patch_radius=args.patch_radius,
        gap_thresh=args.gap_thresh,
        ambiguity_thresh=0.0,   # unused in current _analyze_patch
        min_valid=3,            # hardcoded in current _analyze_patch
    )

    # ------------------------------------------------------------------ #
    # Mode-selection analysis                                             #
    # ------------------------------------------------------------------ #
    print(f"\nAnalyzing mode selection (gap_thresh={args.gap_thresh}) ...")
    result = _analyze(data, provider)
    n_cand = result["n_candidates"]
    n1 = result["n_mode1"]
    n2 = result["n_mode2"]
    records = result["records"]

    print(f"\n{'='*60}")
    print(f"Bimodal-candidate measurements : {n_cand}")
    if n_cand == 0:
        print("No bimodal candidates found. Check gap_thresh or depth_map_dir.")
        return

    print(f"  Mode 1 (primary d)  selected : {n1}  ({100*n1/n_cand:.1f}%)")
    print(f"  Mode 2 (alt  d_alt) selected : {n2}  ({100*n2/n_cand:.1f}%)")
    print(f"{'='*60}\n")

    if n_cand > 0:
        r1s = np.array([r["r1"] for r in records])
        r2s = np.array([r["r2"] for r in records])
        print("Depth residual statistics across all bimodal candidates (metres):")
        print(f"  |z_pred - d|     : median={np.median(r1s):.4f}  mean={np.mean(r1s):.4f}  p95={np.percentile(r1s,95):.4f}")
        print(f"  |z_pred - d_alt| : median={np.median(r2s):.4f}  mean={np.mean(r2s):.4f}  p95={np.percentile(r2s,95):.4f}")

        gaps = np.array([abs(r["d"] - r["d_alt"]) for r in records])
        print(f"\n  |d - d_alt| (hypothesis spread):")
        print(f"    median={np.median(gaps):.4f} m  mean={np.mean(gaps):.4f} m  p95={np.percentile(gaps,95):.4f} m")

    if n2 > 0:
        mode2 = [r for r in records if r["mode"] == 2]
        improvement = np.array([r["r1"] - r["r2"] for r in mode2])  # positive = mode2 is better
        print(f"\nFor the {n2} mode-2 selections:")
        print(f"  Residual improvement |r1|-|r2| (m):")
        print(f"    median={np.median(improvement):.4f}  mean={np.mean(improvement):.4f}  p95={np.percentile(improvement,95):.4f}")

        z_preds = np.array([r["z_pred"] for r in mode2])
        ds = np.array([r["d"] for r in mode2])
        d_alts = np.array([r["d_alt"] for r in mode2])
        print(f"  z_pred vs d vs d_alt (metres):")
        print(f"    median z_pred  = {np.median(z_preds):.4f}")
        print(f"    median d       = {np.median(ds):.4f}")
        print(f"    median d_alt   = {np.median(d_alts):.4f}")

    # ------------------------------------------------------------------ #
    # Mesh comparison (optional)                                          #
    # ------------------------------------------------------------------ #
    if args.gt_ply is not None:
        if args.gt_traj is None:
            print("\n[mesh] --gt_traj required for Sim3 alignment; skipping mesh comparison.")
        else:
            print(f"\nMesh comparison vs: {args.gt_ply}")

            # Unique track indices for mode-1 and mode-2 (one 3D point per track).
            mode1_tracks = {r["track_idx"]: r["point_w"] for r in records if r["mode"] == 1}
            mode2_tracks = {r["track_idx"]: r["point_w"] for r in records if r["mode"] == 2}

            if mode1_tracks:
                pts1 = np.stack(list(mode1_tracks.values()))
                d1 = _mesh_distances(pts1, args.gt_ply, args.gt_traj, data)
                print(f"\n  Mode-1 tracks ({len(pts1)} unique landmarks):")
                print(f"    dist-to-GT  median={np.nanmedian(d1)*100:.2f} cm  mean={np.nanmean(d1)*100:.2f} cm  p95={np.nanpercentile(d1,95)*100:.2f} cm")

            if mode2_tracks:
                pts2 = np.stack(list(mode2_tracks.values()))
                d2 = _mesh_distances(pts2, args.gt_ply, args.gt_traj, data)
                print(f"\n  Mode-2 tracks ({len(pts2)} unique landmarks):")
                print(f"    dist-to-GT  median={np.nanmedian(d2)*100:.2f} cm  mean={np.nanmean(d2)*100:.2f} cm  p95={np.nanpercentile(d2,95)*100:.2f} cm")

    print("\nDone.")


if __name__ == "__main__":
    main()
