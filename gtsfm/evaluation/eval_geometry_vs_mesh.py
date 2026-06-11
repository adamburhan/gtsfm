"""Evaluate a sparse reconstruction against ground-truth scene geometry.

Computes the standard point-based reconstruction metrics (Tanks & Temples
protocol, also used in Gaussian-splatting papers for geometry):

  - accuracy/precision@tau: fraction of reconstructed points within tau of GT.
  - completeness/recall@tau: fraction of GT samples within tau of the reconstruction.
  - F-score@tau: harmonic mean of the two.
  - median / p95 / mean point-to-GT distance ("floater"-sensitive direction).

The reconstruction is read from a COLMAP-format directory (GTSfM's
`results/ba_output`, which is already Sim(3)-aligned to the GT poses). The GT
geometry may be a triangle mesh (e.g. Replica `{sequence}_mesh.ply`, sampled
uniformly) or a point cloud (e.g. Tanks & Temples laser scans).

Note: completeness of a *sparse* SfM cloud is inherently low (keypoints cannot
cover untextured surfaces); it is reported for completeness but the
hypothesis-relevant direction is accuracy (floaters).

Example:
    python gtsfm/evaluation/eval_geometry_vs_mesh.py \
        --ba_dir results/ba_output --gt_ply office0_mesh.ply --out metrics.json

Authors: Adam Burhan
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

import gtsfm.utils.io as io_utils

N_GT_SAMPLES = 1_000_000  # GT mesh surface samples (dense enough for tau >= 1cm).


def umeyama_sim3(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity transform (s, R, t) such that dst ~= s * R @ src + t (Umeyama 1991)."""
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - mu_s, dst - mu_d
    cov = dst_c.T @ src_c / src.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / (src_c**2).sum(axis=1).mean()
    t = mu_d - s * R @ mu_s
    return float(s), R, t


def estimate_export_to_world_sim3(gt_cameras_dir: str, gt_traj: str) -> tuple[float, np.ndarray, np.ndarray]:
    """Estimate the Sim(3) mapping GTSfM's exported frame to the dataset (mesh) world frame.

    GTSfM exports `ba_output` (and the GT cameras, in `results/ba_gt`) in an axis-aligned
    visualization frame, not the original world frame the GT mesh lives in. Since `ba_gt`
    holds the GT cameras *after* that transform and the trajectory file holds them *before*,
    the Sim(3) between the two camera-center sets recovers the export->world mapping exactly.

    Args:
        gt_cameras_dir: COLMAP-format dir of GT cameras in the exported frame (results/ba_gt).
        gt_traj: Replica-style traj.txt (one flattened 4x4 cam-to-world row per frame).

    Returns:
        (s, R, t) such that p_world = s * R @ p_export + t.
    """
    wTi_list, img_fnames, _, _, _, _ = io_utils.read_scene_data_from_colmap_format(gt_cameras_dir)
    traj = np.loadtxt(gt_traj).reshape(-1, 4, 4)

    export_centers, world_centers = [], []
    for wTi, fname in zip(wTi_list, img_fnames):
        match = re.search(r"(\d+)$", Path(fname).stem)
        if wTi is None or match is None:
            continue
        export_centers.append(wTi.translation())
        world_centers.append(traj[int(match.group(1))][:3, 3])
    if len(export_centers) < 3:
        raise ValueError(f"Need >= 3 matched cameras to align, got {len(export_centers)}.")
    return umeyama_sim3(np.array(export_centers), np.array(world_centers))


def load_gt_points(gt_ply: str) -> np.ndarray:
    """Load GT geometry as a dense point cloud (sampling the surface if it is a mesh)."""
    mesh = o3d.io.read_triangle_mesh(gt_ply)
    if len(mesh.triangles) > 0:
        pcd = mesh.sample_points_uniformly(number_of_points=N_GT_SAMPLES)
    else:
        pcd = o3d.io.read_point_cloud(gt_ply)
    return np.asarray(pcd.points)


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
    parser.add_argument("--ba_dir", required=True, help="COLMAP-format reconstruction dir (e.g. results/ba_output).")
    parser.add_argument("--gt_ply", required=True, help="GT geometry: triangle mesh or point cloud (.ply).")
    parser.add_argument(
        "--transform_txt",
        default=None,
        help="Optional 4x4 transform (text file) mapping the reconstruction into the GT frame "
        "(e.g. Tanks & Temples *_trans.txt). Identity if omitted (Replica).",
    )
    parser.add_argument(
        "--gt_cameras_dir",
        default=None,
        help="COLMAP-format dir of GT cameras in GTSfM's exported frame (results/ba_gt). Together "
        "with --gt_traj, used to Sim(3)-align the reconstruction to the mesh world frame.",
    )
    parser.add_argument(
        "--gt_traj",
        default=None,
        help="Replica-style traj.txt with GT cam-to-world poses in the mesh world frame.",
    )
    parser.add_argument("--tau", type=float, nargs="+", default=[0.025, 0.05], help="Distance thresholds in meters.")
    parser.add_argument("--out", default=None, help="Output JSON path (default: <ba_dir>/../geometry_metrics.json).")
    args = parser.parse_args()

    _, _, _, points, _, _ = io_utils.read_scene_data_from_colmap_format(args.ba_dir)
    if args.transform_txt is not None:
        T = np.loadtxt(args.transform_txt).reshape(4, 4)
        points = points @ T[:3, :3].T + T[:3, 3]
    if args.gt_cameras_dir is not None and args.gt_traj is not None:
        s, R, t = estimate_export_to_world_sim3(args.gt_cameras_dir, args.gt_traj)
        print(f"Export->world Sim(3): scale={s:.6f}")
        points = s * points @ R.T + t

    gt_points = load_gt_points(args.gt_ply)
    metrics = evaluate_points(points, gt_points, args.tau)

    out_path = Path(args.out) if args.out else Path(args.ba_dir).parent / "geometry_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {out_path}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
