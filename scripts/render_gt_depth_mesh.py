"""Render clean GT depth by raycasting ETH3D's occlusion surface mesh into the undistorted cameras.

A point-splat z-buffer of the raw laser cloud is noisy (scatter), holey, and foreground-biased,
which corrupts BA. ETH3D instead ships an occlusion-aware surface mesh (occlusion/surface_mesh.ply);
raycasting it gives dense (100% coverage), occlusion-correct, artifact-free depth. Verified to match ETH3D's official ground_truth_depth at the image center to
~1 mm. Output is <image_stem>.npy at the loader's working resolution (template=null + depth_ext=.npy),
in the undistorted frame so it aligns with the track uv.

Usage:
    python scripts/render_gt_depth_mesh.py \
        --mesh       $DATA/kicker/occlusion/surface_mesh.ply \
        --colmap_dir $DATA/kicker/dslr_calibration_undistorted \
        --out_dir    $DATA/kicker/gt_depth_mesh_760 \
        --max_resolution 760

Authors: Adam Burhan
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

from gtsfm.utils import io as io_utils
from gtsfm.utils.images import get_downsampling_factor_per_axis


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", required=True, help="ETH3D occlusion surface mesh (occlusion/surface_mesh.ply)")
    ap.add_argument("--colmap_dir", required=True, help="Undistorted COLMAP dir (poses + intrinsics)")
    ap.add_argument("--out_dir", required=True, help="Output dir for loader-resolution GT depth .npy")
    ap.add_argument("--max_resolution", type=int, default=760, help="Must match the GTSfM loader --max_resolution")
    args = ap.parse_args()

    print("Loading mesh ...")
    mesh = o3d.io.read_triangle_mesh(args.mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    wTi_list, img_fnames, calibrations, _, _, img_dims = io_utils.read_scene_data_from_colmap_format(args.colmap_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(wTi_list)} cameras; mesh has {len(mesh.triangles):,} faces.")

    n_done = 0
    for wTi, fname, cal, (h, w) in zip(wTi_list, img_fnames, calibrations, img_dims):
        if wTi is None or cal is None:
            continue
        scale_u, scale_v, new_h, new_w = get_downsampling_factor_per_axis(h, w, args.max_resolution)
        K = np.asarray(cal.K(), dtype=np.float64).copy()
        K[0] *= scale_u
        K[1] *= scale_v

        cTw = wTi.inverse()  # world -> camera (open3d extrinsic convention)
        extr = np.eye(4)
        extr[:3, :3] = np.asarray(cTw.rotation().matrix())
        extr[:3, 3] = np.asarray(cTw.translation())

        rays = scene.create_rays_pinhole(o3d.core.Tensor(K), o3d.core.Tensor(extr), new_w, new_h)
        thit = scene.cast_rays(rays)["t_hit"].numpy()
        r = rays.numpy()
        hit_valid = np.isfinite(thit)
        # Miss rays have inf t_hit (and possibly inf direction); zero them out before the matmul to
        # avoid inf*0=NaN warnings, then mask back to NaN.
        hit = r[..., :3] + np.where(hit_valid, thit, 0.0)[..., None] * np.nan_to_num(r[..., 3:])
        depth = (hit @ extr[:3, :3].T + extr[:3, 3])[..., 2].astype(np.float32)  # planar cam-frame Z
        depth[~hit_valid] = np.nan

        valid = float(np.isfinite(depth).mean())
        np.save(out_dir / (Path(fname).stem + ".npy"), depth)
        if n_done == 0:
            print(f"[ok] {fname}: ({h},{w}) -> ({new_h},{new_w}); coverage={valid:.1%}, "
                  f"median={np.nanmedian(depth):.3f} m")
        n_done += 1

    print(f"\nWrote {n_done} GT depth maps to {out_dir} (max_resolution={args.max_resolution}).")


if __name__ == "__main__":
    main()
