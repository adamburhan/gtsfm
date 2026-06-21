"""Evaluate a sparse reconstruction against ground-truth scene geometry.

Point-based metrics (Tanks & Temples protocol): accuracy/precision@tau, completeness/recall@tau,
F-score@tau, and point-to-GT distance percentiles.

Authors: Adam Burhan
"""

import argparse
import json
import re
from pathlib import Path

import gtsam
import numpy as np
import trimesh
import trimesh.sample
from scipy.spatial import cKDTree

import gtsfm.utils.io as io_utils
from gtsfm.utils import align

N_GT_SAMPLES = 1_000_000

ALIGN_MODES = ("none", "replica", "eth3d", "tnt")


def _gt_poses_from_traj(align_ref: str, img_fnames) -> dict[int, gtsam.Pose3]:
    """Replica traj.txt -> {recon_idx: GT cam-to-world}, matched by trailing frame number."""
    traj = np.loadtxt(align_ref).reshape(-1, 4, 4)
    poses = {}
    for i, fname in enumerate(img_fnames):
        match = re.search(r"(\d+)$", Path(fname).stem)
        if match is not None:
            T = traj[int(match.group(1))]
            poses[i] = gtsam.Pose3(gtsam.Rot3(T[:3, :3]), T[:3, 3])
    return poses


def _gt_poses_from_colmap(align_ref: str, img_fnames) -> dict[int, gtsam.Pose3]:
    """COLMAP GT dir (ETH3D) -> {recon_idx: GT cam-to-world}, matched by filename."""
    gt_wTi, gt_names, *_ = io_utils.read_scene_data_from_colmap_format(align_ref)
    by_name = {Path(n).name: p for n, p in zip(gt_names, gt_wTi) if p is not None}
    return {i: by_name[Path(f).name] for i, f in enumerate(img_fnames) if Path(f).name in by_name}


def sim3_align(wTi_list, gt_poses: dict[int, gtsam.Pose3]) -> tuple[gtsam.Similarity3, float]:
    """Robust Sim(3) from estimated cameras to GT poses; returns (wSr, camera-center RMS_m).

    Same alignment as the pose-AUC metric. The RMS surfaces a degenerate fit instead of
    silently corrupting the geometry numbers.
    """
    aTi = {i: gt_poses[i] for i in gt_poses if wTi_list[i] is not None}
    bTi = {i: wTi_list[i] for i in aTi}
    if len(aTi) < 3:
        raise ValueError(f"Need >= 3 matched cameras to align, got {len(aTi)}.")
    wSr = align.sim3_from_Pose3_maps_robust(aTi, bTi)
    residuals = [np.linalg.norm(wSr.transformFrom(bTi[i].translation()) - aTi[i].translation()) for i in aTi]
    return wSr, float(np.sqrt(np.mean(np.square(residuals))))


def load_gt_points(gt_ply: str) -> np.ndarray:
    """GT geometry as a dense point cloud (trimesh sampling; falls back to raw vertices)."""
    tm = trimesh.load(gt_ply, process=False, force="mesh")
    if getattr(tm, "faces", None) is not None and len(tm.faces) > 0:
        return np.asarray(trimesh.sample.sample_surface(tm, N_GT_SAMPLES)[0])
    print("[load_gt_points] No triangulated faces; using raw vertices.")
    return np.asarray(getattr(tm, "vertices", np.zeros((0, 3))))


def evaluate_points(points: np.ndarray, gt_points: np.ndarray, taus: list[float]) -> dict:
    """Compute T&T-style precision/recall/F-score and distance statistics."""
    d_acc = cKDTree(gt_points).query(points, k=1)[0]  # recon -> GT
    d_comp = cKDTree(points).query(gt_points, k=1)[0]  # GT -> recon

    metrics: dict = {
        "n_points": int(points.shape[0]),
        "accuracy_median_m": float(np.median(d_acc)),
        "accuracy_mean_m": float(np.mean(d_acc)),
        "accuracy_p95_m": float(np.percentile(d_acc, 95)),
        "accuracy_max_m": float(np.max(d_acc)),
    }
    for tau in taus:
        precision = float((d_acc < tau).mean())
        recall = float((d_comp < tau).mean())
        f_score = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        key = f"{tau * 100:g}cm"
        metrics[f"precision@{key}"] = precision
        metrics[f"recall@{key}"] = recall
        metrics[f"fscore@{key}"] = f_score
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sfm_output", required=True, help="COLMAP-format reconstruction dir (e.g. results/merged).")
    parser.add_argument("--gt_ply", required=True, help="GT geometry: triangle mesh or point cloud (.ply).")
    parser.add_argument("--align_mode", choices=ALIGN_MODES, default="none", help="Recon->GT-frame alignment adapter.")
    parser.add_argument(
        "--align_ref",
        default=None,
        help="Alignment reference: traj.txt (replica), COLMAP GT dir (eth3d), or 4x4 *_trans.txt (tnt).",
    )
    parser.add_argument("--tau", type=float, nargs="+", default=[0.025, 0.05], help="Distance thresholds in meters.")
    parser.add_argument("--out", default=None, help="Output JSON path (default: <sfm_output>/../geometry_metrics.json).")
    args = parser.parse_args()

    if args.align_mode != "none" and args.align_ref is None:
        parser.error(f"--align_mode {args.align_mode} requires --align_ref")

    wTi_list, img_fnames, _, points, _, _ = io_utils.read_scene_data_from_colmap_format(args.sfm_output)
    if args.align_mode == "tnt":
        T = np.loadtxt(args.align_ref).reshape(4, 4)
        points = points @ T[:3, :3].T + T[:3, 3]
    elif args.align_mode in ("replica", "eth3d"):
        load_poses = _gt_poses_from_traj if args.align_mode == "replica" else _gt_poses_from_colmap
        wSr, rms_m = sim3_align(wTi_list, load_poses(args.align_ref, img_fnames))
        print(f"recon->world Sim(3): scale={wSr.scale():.6f}, camera-center RMS={rms_m:.4f} m")
        points = np.array([wSr.transformFrom(p) for p in points])

    metrics = evaluate_points(points, load_gt_points(args.gt_ply), args.tau)

    out_path = Path(args.out) if args.out else Path(args.sfm_output).parent / "geometry_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {out_path}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
