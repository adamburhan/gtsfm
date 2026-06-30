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
from gtsfm.common.depth_provider import DepthProvider
from gtsfm.common.gtsfm_data import GtsfmData
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

    For a mesh PLY, gt_dist is true point-to-SURFACE distance (open3d raycasting), and gt_points are
    uniformly sampled from the surface. For a point-cloud PLY it falls back to nearest-point (which
    overestimates near the surface). Prefer a mesh (e.g. ETH3D occlusion/surface_mesh.ply).
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
    return gt_points, lambda p: tree.query(np.asarray(p))[0]


def evaluate_points(points: np.ndarray, gt_points: np.ndarray, taus: list[float], gt_dist=None) -> dict:
    """Compute T&T-style precision/recall/F-score and distance statistics.

    `gt_dist` (recon-point -> GT distance) defaults to nearest-point over `gt_points`; pass a
    point-to-surface fn (see `build_gt` on a mesh) for the fairer accuracy metric.
    """
    d_acc = gt_dist(points) if gt_dist is not None else cKDTree(gt_points).query(points, k=1)[0]  # recon -> GT
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

def build_to_world(align_mode: str, align_ref, wTi_list, img_fnames):
    """Return f(point)->point mapping a recon point into the GT/world frame (same alignment as pose-AUC)."""
    if align_mode == "none":
        return lambda p: np.asarray(p, dtype=float)
    if align_mode == "tnt":
        T = np.loadtxt(align_ref).reshape(4, 4)
        R, t = T[:3, :3], T[:3, 3]
        return lambda p: np.asarray(p, dtype=float) @ R.T + t
    load_poses = _gt_poses_from_traj if align_mode == "replica" else _gt_poses_from_colmap
    wSr, rms_m = sim3_align(wTi_list, load_poses(align_ref, img_fnames))
    print(f"recon->world Sim(3): scale={wSr.scale():.6f}, camera-center RMS={rms_m:.4f} m")
    return lambda p: np.array(wSr.transformFrom(np.asarray(p, dtype=float)))


def build_image_fnames(data: GtsfmData) -> dict[int, str]:
    out = {}
    for i in data.get_valid_camera_indices():
        info = data.get_image_info(i)
        if info is not None and info.name:
            out[i] = info.name
    return out


def _load_depth_npz(path: str, data: GtsfmData) -> dict[int, np.ndarray]:
    """Load a node's depth.npz, re-keyed to read_colmap's 0-based sorted-filename order."""
    raw = np.load(path)
    arrays = {int(k): raw[k] for k in raw.keys()}
    global_keys = sorted(arrays)
    n_cams = len(data.get_valid_camera_indices())
    if n_cams != len(global_keys):
        raise ValueError(f"{n_cams} recon cams != {len(global_keys)} depth cams; cannot align indices.")
    return {local: arrays[g] for local, g in enumerate(global_keys)}


def build_provider(args, data: GtsfmData):
    """DepthProvider from on-disk maps (--depth_map_dir) or in-memory VGGT depth (--depth_npz), else None."""
    if args.depth_map_dir and args.depth_npz:
        raise ValueError("Pass only one of --depth_map_dir / --depth_npz.")
    if args.depth_npz:
        return DepthProvider(
            depth_arrays=_load_depth_npz(args.depth_npz, data),
            depth_min=args.depth_min, depth_max=args.depth_max, compute_hypotheses=True,
            patch_radius=args.patch_radius, gap_thresh=args.gap_thresh, ambiguity_thresh=0.0, min_valid=args.min_valid,
        )
    if args.depth_map_dir:
        return DepthProvider(
            depth_map_dir=args.depth_map_dir, image_fnames=build_image_fnames(data),
            depth_scale=args.depth_scale, depth_filename_template=args.depth_filename_template,
            depth_min=args.depth_min, depth_max=args.depth_max, compute_hypotheses=True,
            patch_radius=args.patch_radius, gap_thresh=args.gap_thresh, ambiguity_thresh=0.0, min_valid=args.min_valid,
        )
    return None


def mode_records(data: GtsfmData, provider, gt_dist, to_world) -> tuple[list[dict], set[int]]:
    """Walk measurements once: per ambiguous measurement, did BA converge to the GT-closer depth mode?

    Returns the per-measurement records and the set of tracks with >=1 ambiguous measurement.
    """
    records, ambiguous_tracks = [], set()
    for j in range(data.number_tracks()):
        track = data.get_track(j)
        point_w = np.array(track.point3())
        for m in range(track.numberMeasurements()):
            i, uv = track.measurement(m)
            cam = data.get_camera(i)
            if cam is None:
                continue
            sample = provider.get_depth(i, float(uv[0]), float(uv[1]))
            if sample is None or not sample.ambiguous or sample.depth_alt is None:
                continue
            ambiguous_tracks.add(j)
            z = float(cam.pose().transformTo(point_w)[2])  # converged camera-frame Z
            opt_mode = 2 if abs(z - sample.depth_alt) < abs(z - sample.depth) else 1
            uv2 = gtsam.Point2(float(uv[0]), float(uv[1]))
            dists = gt_dist(np.array([to_world(cam.backproject(uv2, sample.depth)),
                                      to_world(cam.backproject(uv2, sample.depth_alt))]))
            dist_d, dist_alt = float(dists[0]), float(dists[1])
            records.append({
                "gap": abs(sample.depth - sample.depth_alt),
                "opt_mode": opt_mode,
                "gt_mode": 1 if dist_d < dist_alt else 2,
                "dist_best": min(dist_d, dist_alt),
                "dist_selected": dist_d if opt_mode == 1 else dist_alt,  # GT distance of the chosen mode
            })
    return records, ambiguous_tracks


def summarize_modes(records: list[dict], tau: float) -> dict:
    """Aggregate mode-selection records into scalar metrics."""
    n = len(records)
    if n == 0:
        return {"n_ambiguous_measurements": 0}
    opt = np.array([r["opt_mode"] for r in records])
    gt = np.array([r["gt_mode"] for r in records])
    gap = np.array([r["gap"] for r in records])
    best = np.array([r["dist_best"] for r in records])
    selected = np.array([r["dist_selected"] for r in records])
    correct = opt == gt
    mode2 = opt == 2
    return {
        "n_ambiguous_measurements": n,
        "mode2_selected_frac": float(mode2.mean()),
        "mode_correct_frac": float(correct.mean()),                # BA picked the GT-closer hypothesis
        "mode2_correct_frac": float(correct[mode2].mean()) if mode2.any() else 0.0,
        "primary_correct_frac": float((gt == 1).mean()),           # always-pick-primary baseline
        "bimodal_over_primary": float(correct.mean() - (gt == 1).mean()),
        "oracle_within_tau_frac": float((best < tau).mean()),      # is a GT-accurate hypothesis even present
        "selection_cost_mean_m": float((selected - best).mean()),  # GT-distance lost to wrong mode choices
        "dist_selected_median_m": float(np.median(selected)),      # chosen surface's distance to GT
        "gap_median": float(np.median(gap)),
    }


def subset_metrics(data: GtsfmData, track_ids: set[int], to_world, gt_points: np.ndarray, taus: list[float], gt_dist) -> dict:
    """Global geometry metrics restricted to the ambiguous-track subset."""
    if not track_ids:
        return {"n_points": 0}
    pts = np.array([to_world(data.get_track(j).point3()) for j in sorted(track_ids)])
    return evaluate_points(pts, gt_points, taus, gt_dist)


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
    # Depth source (enables ambiguous-subset geometry + mode-correctness metrics). Pass at most one.
    parser.add_argument("--depth_map_dir", default=None, help="On-disk depth maps (e.g. Replica GT).")
    parser.add_argument("--depth_npz", default=None, help="A node's depth.npz (VGGT in-memory depth).")
    parser.add_argument("--depth_scale", type=float, default=6553.5, help="Divisor for on-disk depth (1.0 for float).")
    parser.add_argument("--depth_filename_template", default="depth{:06d}.png")
    parser.add_argument("--depth_min", type=float, default=0.1)
    parser.add_argument("--depth_max", type=float, default=20.0)
    parser.add_argument("--gap_thresh", type=float, default=0.10)
    parser.add_argument("--patch_radius", type=int, default=5)
    parser.add_argument("--min_valid", type=int, default=10, help="Min valid patch pixels (match BA depth_min_valid).")
    parser.add_argument("--mode_tau", type=float, default=0.05, help="Oracle band: a hypothesis this close to GT counts as available.")
    args = parser.parse_args()

    if args.align_mode != "none" and args.align_ref is None:
        parser.error(f"--align_mode {args.align_mode} requires --align_ref")

    wTi_list, img_fnames, _, points, _, _ = io_utils.read_scene_data_from_colmap_format(args.sfm_output)
    to_world = build_to_world(args.align_mode, args.align_ref, wTi_list, img_fnames)
    points = np.array([to_world(p) for p in points])

    gt_points, gt_dist = build_gt(args.gt_ply)  # point-to-surface accuracy for a mesh PLY
    metrics = evaluate_points(points, gt_points, args.tau, gt_dist)

    if args.depth_map_dir or args.depth_npz:
        data = GtsfmData.read_colmap(args.sfm_output)
        provider = build_provider(args, data)
        records, ambiguous_tracks = mode_records(data, provider, gt_dist, to_world)
        metrics["modes"] = summarize_modes(records, args.mode_tau)
        metrics["ambiguous_subset"] = subset_metrics(data, ambiguous_tracks, to_world, gt_points, args.tau, gt_dist)

    out_path = Path(args.out) if args.out else Path(args.sfm_output).parent / "geometry_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {out_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
