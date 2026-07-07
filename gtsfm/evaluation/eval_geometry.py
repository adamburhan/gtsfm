"""Evaluate a sparse reconstruction against ground-truth scene geometry.

Geometry (Tanks & Temples protocol): accuracy/precision@tau, completeness/recall@tau, F-score@tau,
and point-to-GT-surface distance percentiles (true point-to-surface distance for a mesh GT).
Also computes camera-pose metrics (relative pose AUC + absolute rotation/translation error) from the
saved cameras vs the GT reference, written into the same JSON. (NVS / 3DGS metrics are evaluated
separately.)

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
from gtsfm.utils import align, transform
from gtsfm.utils import metrics as metrics_utils

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


def _gt_poses_from_log(align_ref: str, img_fnames) -> dict[int, gtsam.Pose3]:
    """T&T scene dir -> {recon_idx: GT cam-to-world} from *_COLMAP_SfM.log (1-based image filenames)."""
    from gtsfm.loader.tanks_and_temples_loader import _parse_redwood_data_log_file

    log_poses = _parse_redwood_data_log_file(str(next(Path(align_ref).glob("*_COLMAP_SfM.log"))))
    poses = {}
    for i, fname in enumerate(img_fnames):
        match = re.search(r"(\d+)$", Path(fname).stem)
        if match is not None and int(match.group(1)) - 1 in log_poses:
            poses[i] = log_poses[int(match.group(1)) - 1]
    return poses


_GT_POSE_LOADERS = {"replica": _gt_poses_from_traj, "eth3d": _gt_poses_from_colmap, "tnt": _gt_poses_from_log}


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


def compute_pose_metrics(wTi_list, img_fnames, align_mode: str, align_ref) -> dict | None:
    """Relative pose AUC + absolute rot/trans errors vs GT, computed from the saved cameras.

    Recovers the pose metrics that BA's own `evaluate()` skipped (it early-returns when image_info is
    empty). Uses only the estimated poses in the reconstruction plus the GT reference, so it can be run
    offline over existing `ba_output` dirs with no re-reconstruction. Emits the same keys and JSON shape
    as `bundle_adjustment_metrics.json` (pose_auc_@X_deg scalars; rotation_angle_error_deg /
    translation_error_distance as {"summary": {...}} distributions), so the sweep aggregator reads them
    identically. Returns None when GT poses are unavailable (align_mode "none") or too few match.
    """
    if align_mode == "none":
        return None
    gt_poses = _GT_POSE_LOADERS[align_mode](align_ref, img_fnames)
    matched = {i: wTi_list[i] for i in gt_poses if i < len(wTi_list) and wTi_list[i] is not None}
    if len(matched) < 3:
        return None
    # Align estimated cameras into the GT frame (AUC uses gauge-free relative errors; the absolute
    # rot/trans metrics need this Sim(3)).
    wSr, _ = sim3_align(wTi_list, gt_poses)
    aligned = transform.Pose3_map_with_sim3(wSr, matched)
    group = metrics_utils.compute_ba_pose_metrics(
        gt_wTi={i: gt_poses[i] for i in matched},
        computed_wTi=aligned,
        metric_constructed_only=True,
    )
    return next(iter(group.get_metrics_as_dict().values()))


def load_gt_points(gt_ply: str) -> np.ndarray:
    """GT geometry as a point cloud: sample a mesh's surface, or use a point cloud's vertices.

    No `force="mesh"`: that coerces a point-cloud PLY into a faceless mesh and silently drops
    all points. Branch on the loaded type instead (Trimesh -> sample, PointCloud -> vertices).
    """
    geo = trimesh.load(gt_ply, process=False)
    if hasattr(geo, "faces") and len(geo.faces) > 0:
        return np.asarray(trimesh.sample.sample_surface(geo, N_GT_SAMPLES)[0])
    return np.asarray(geo.vertices)


def build_gt(gt_ply: str):
    """Returns (gt_points, gt_dist): GT surface points for completeness, and a recon->GT distance fn.

    For a point-cloud PLY (e.g. the merged ETH3D scan), gt_dist is nearest-point (point-to-point);
    scan sampling density sets the distance floor, so sub-voxel taus are not meaningful. For a mesh
    PLY, gt_dist is true point-to-SURFACE distance (open3d raycasting), and gt_points are uniformly
    sampled from the surface — but a reconstructed mesh's interpolated surface biases accuracy, so
    the raw scan is preferred as GT.
    """
    import open3d as o3d

    o3d_mesh = o3d.io.read_triangle_mesh(gt_ply)
    if len(o3d_mesh.triangles) > 0:
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(o3d_mesh))
        gt_points = np.asarray(o3d_mesh.sample_points_uniformly(N_GT_SAMPLES).points)  # completeness
        return gt_points, lambda p: scene.compute_distance(o3d.core.Tensor(np.asarray(p, np.float32))).numpy()
    gt_points = load_gt_points(gt_ply)
    tree = cKDTree(gt_points)
    return gt_points, lambda p: tree.query(np.asarray(p), workers=-1)[0]


def evaluate_points(points: np.ndarray, gt_points: np.ndarray, taus: list[float], gt_dist=None) -> dict:
    """Compute T&T-style precision/recall/F-score and distance statistics.

    `gt_dist` (recon-point -> GT distance) defaults to nearest-point over `gt_points`; pass a
    point-to-surface fn (see `build_gt` on a mesh) for the fairer accuracy metric.
    """
    d_acc = gt_dist(points) if gt_dist is not None else cKDTree(gt_points).query(points, k=1, workers=-1)[0]
    d_comp = cKDTree(points).query(gt_points, k=1, workers=-1)[0]  # GT -> recon

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


def crop_to_tnt_volume(points: np.ndarray, align_ref) -> np.ndarray:
    """Official T&T protocol: crop recon points to the scene's bounding polyhedron (GT-frame *.json).

    Points outside the scanned volume (through windows, down corridors) otherwise get billed the
    nearest-neighbor distance to a GT scan that never covered them.
    """
    import open3d as o3d

    vol = o3d.visualization.read_selection_polygon_volume(str(next(Path(align_ref).glob("*.json"))))
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points)))
    cropped = np.asarray(vol.crop_point_cloud(pcd).points)
    print(f"tnt crop volume: kept {len(cropped)}/{len(points)} points")
    return cropped


def build_to_world(align_mode: str, align_ref, wTi_list, img_fnames):
    """Map recon points into the GT/world frame. Returns (to_world, alignment).

    All pose modes Sim(3)-fit the recon cameras to GT poses; `alignment` surfaces the fit's
    `{"sim3_scale", "camera_rms_m"}` for the metrics JSON (None for align_mode "none"). For tnt the
    fit lands in the COLMAP GT frame, then *_trans.txt maps COLMAP -> LiDAR (GT ply) frame.
    """
    if align_mode == "none":
        return (lambda p: np.asarray(p, dtype=float)), None
    wSr, rms_m = sim3_align(wTi_list, _GT_POSE_LOADERS[align_mode](align_ref, img_fnames))
    print(f"recon->world Sim(3): scale={wSr.scale():.6f}, camera-center RMS={rms_m:.4f} m")
    to_world = lambda p: np.array(wSr.transformFrom(np.asarray(p, dtype=float)))
    if align_mode == "tnt":
        T = np.loadtxt(next(Path(align_ref).glob("*_trans.txt"))).reshape(4, 4)
        R, t = T[:3, :3], T[:3, 3]
        colmap_to_world = to_world
        to_world = lambda p: colmap_to_world(p) @ R.T + t
    return to_world, {"sim3_scale": float(wSr.scale()), "camera_rms_m": float(rms_m)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sfm_output", required=True, help="COLMAP-format reconstruction dir (e.g. results/merged).")
    parser.add_argument("--gt_ply", required=True, help="GT geometry: triangle mesh or point cloud (.ply).")
    parser.add_argument("--align_mode", choices=ALIGN_MODES, default="none", help="Recon->GT-frame alignment adapter.")
    parser.add_argument(
        "--align_ref",
        default=None,
        help="Alignment reference: traj.txt (replica), COLMAP GT dir (eth3d), or scene dir with "
        "*_COLMAP_SfM.log + *_trans.txt (tnt).",
    )
    parser.add_argument("--tau", type=float, nargs="+", default=[0.025, 0.05], help="Distance thresholds in meters.")
    parser.add_argument("--out", default=None, help="Output JSON path (default: <sfm_output>/../geometry_metrics.json).")
    args = parser.parse_args()

    if args.align_mode != "none" and args.align_ref is None:
        parser.error(f"--align_mode {args.align_mode} requires --align_ref")

    wTi_list, img_fnames, _, points, _, _ = io_utils.read_scene_data_from_colmap_format(args.sfm_output)
    to_world, alignment = build_to_world(args.align_mode, args.align_ref, wTi_list, img_fnames)
    points = np.array([to_world(p) for p in points])
    if args.align_mode == "tnt":
        points = crop_to_tnt_volume(points, args.align_ref)

    gt_points, gt_dist = build_gt(args.gt_ply)  # point-to-surface accuracy for a mesh PLY
    metrics = evaluate_points(points, gt_points, args.tau, gt_dist)
    if alignment is not None:
        metrics["alignment"] = alignment  # sim3_scale + camera-center RMS (m) for the sweep table

    pose_metrics = compute_pose_metrics(wTi_list, img_fnames, args.align_mode, args.align_ref)
    if pose_metrics is not None:
        metrics.update(pose_metrics)  # pose AUC + rot/trans errors (recovered offline; see aggregate_modes)

    out_path = Path(args.out) if args.out else Path(args.sfm_output).parent / "geometry_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {out_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
