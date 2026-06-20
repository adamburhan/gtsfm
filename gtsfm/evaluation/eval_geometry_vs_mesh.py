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

import gtsam
import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree

import gtsfm.utils.io as io_utils
from gtsfm.utils import align

N_GT_SAMPLES = 1_000_000  # GT mesh surface samples (dense enough for tau >= 1cm).


def align_recon_to_world(wTi_list, img_fnames, gt_traj: str) -> tuple[gtsam.Similarity3, float]:
    """Robustly Sim(3)-align the reconstruction's cameras to the GT world (mesh) frame.

    Mirrors the pose-AUC alignment (`GtsfmData.align_via_sim3_and_transform`), which uses
    `sim3_from_Pose3_maps_robust` on estimated-vs-GT poses, rather than a separate
    least-squares fit through `ba_gt`. Returns the transform and the RMS camera-center
    residual so a degenerate alignment is visible instead of silently corrupting the metric.

    Args:
        wTi_list: Estimated camera poses (cam-to-world) from the reconstruction.
        img_fnames: Image filenames parallel to wTi_list (trailing digits index gt_traj).
        gt_traj: Replica-style traj.txt (one flattened 4x4 cam-to-world row per frame).

    Returns:
        (wSr, rms_m) such that p_world = wSr.transformFrom(p_recon).
    """
    traj = np.loadtxt(gt_traj).reshape(-1, 4, 4)
    aTi: dict[int, gtsam.Pose3] = {}
    bTi: dict[int, gtsam.Pose3] = {}
    for i, (wTi, fname) in enumerate(zip(wTi_list, img_fnames)):
        match = re.search(r"(\d+)$", Path(fname).stem)
        if wTi is None or match is None:
            continue
        T = traj[int(match.group(1))]
        aTi[i] = gtsam.Pose3(gtsam.Rot3(T[:3, :3]), T[:3, 3])
        bTi[i] = wTi
    if len(aTi) < 3:
        raise ValueError(f"Need >= 3 matched cameras to align, got {len(aTi)}.")

    wSr = align.sim3_from_Pose3_maps_robust(aTi, bTi)
    residuals = [np.linalg.norm(wSr.transformFrom(bTi[i].translation()) - aTi[i].translation()) for i in aTi]
    return wSr, float(np.sqrt(np.mean(np.square(residuals))))


def load_gt_points(gt_ply: str) -> np.ndarray:
    """Load GT geometry as a dense point cloud (sampling the surface if it is a mesh).

    Uses trimesh as primary loader (handles Replica quad/n-gon PLYs that Open3D silently
    fails on), falling back to raw vertices if triangulation fails.
    """
    import trimesh.sample
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
    parser.add_argument("--ba_dir", required=True, help="COLMAP-format reconstruction dir (e.g. results/ba_output).")
    parser.add_argument("--gt_ply", required=True, help="GT geometry: triangle mesh or point cloud (.ply).")
    parser.add_argument(
        "--transform_txt",
        default=None,
        help="Optional 4x4 transform (text file) mapping the reconstruction into the GT frame "
        "(e.g. Tanks & Temples *_trans.txt). Identity if omitted (Replica).",
    )
    parser.add_argument(
        "--gt_traj",
        default=None,
        help="Replica-style traj.txt with GT cam-to-world poses in the mesh world frame. When given, "
        "the reconstruction is robustly Sim(3)-aligned to it (same alignment as the pose-AUC metric).",
    )
    parser.add_argument("--tau", type=float, nargs="+", default=[0.025, 0.05], help="Distance thresholds in meters.")
    parser.add_argument("--out", default=None, help="Output JSON path (default: <ba_dir>/../geometry_metrics.json).")
    args = parser.parse_args()

    wTi_list, img_fnames, _, points, _, _ = io_utils.read_scene_data_from_colmap_format(args.ba_dir)
    if args.transform_txt is not None:
        T = np.loadtxt(args.transform_txt).reshape(4, 4)
        points = points @ T[:3, :3].T + T[:3, 3]
    if args.gt_traj is not None:
        wSr, rms_m = align_recon_to_world(wTi_list, img_fnames, args.gt_traj)
        print(f"recon->world Sim(3): scale={wSr.scale():.6f}, camera-center RMS={rms_m:.4f} m")
        points = np.array([wSr.transformFrom(p) for p in points])

    gt_points = load_gt_points(args.gt_ply)
    metrics = evaluate_points(points, gt_points, args.tau)

    out_path = Path(args.out) if args.out else Path(args.ba_dir).parent / "geometry_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {out_path}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
