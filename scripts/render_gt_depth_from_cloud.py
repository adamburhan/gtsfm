"""Render GT depth maps from the ETH3D laser-scan point cloud into the undistorted cameras
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh

from gtsfm.utils import io as io_utils
from gtsfm.utils.images import get_downsampling_factor_per_axis


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt_ply", required=True, help="Merged laser-scan cloud ($GT)")
    ap.add_argument("--colmap_dir", required=True, help="Undistorted COLMAP dir (dslr_calibration_undistorted)")
    ap.add_argument("--out_dir", required=True, help="Output dir for loader-resolution GT depth .npy")
    ap.add_argument("--max_resolution", type=int, default=760, help="Must match the GTSfM loader --max_resolution")
    args = ap.parse_args()

    geo = trimesh.load(args.gt_ply, process=False)
    pts = np.asarray(geo.vertices, dtype=np.float64)  # (N,3) world points
    wTi_list, img_fnames, calibrations, _, _, img_dims = io_utils.read_scene_data_from_colmap_format(args.colmap_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Cloud: {pts.shape[0]:,} points; {len(img_fnames)} cameras.")

    n_done = 0
    for wTi, fname, cal, (h, w) in zip(wTi_list, img_fnames, calibrations, img_dims):
        if wTi is None or cal is None:
            continue
        scale_u, scale_v, new_h, new_w = get_downsampling_factor_per_axis(h, w, args.max_resolution)
        K = np.asarray(cal.K(), dtype=np.float64)
        fx, fy = K[0, 0] * scale_u, K[1, 1] * scale_v
        cx, cy = K[0, 2] * scale_u, K[1, 2] * scale_v

        cTw = wTi.inverse()  # world -> camera
        pc = pts @ np.asarray(cTw.rotation().matrix()).T + np.asarray(cTw.translation())
        z = pc[:, 2]
        front = z > 1e-6
        pc, z = pc[front], z[front]
        col = np.round(fx * pc[:, 0] / z + cx).astype(np.int64)
        row = np.round(fy * pc[:, 1] / z + cy).astype(np.int64)
        inb = (col >= 0) & (col < new_w) & (row >= 0) & (row < new_h)
        col, row, z = col[inb], row[inb], z[inb]

        # z-buffer: sort far->near so the nearest point wins at each pixel.
        depth = np.full(new_h * new_w, np.inf, dtype=np.float32)
        order = np.argsort(-z)
        depth[row[order] * new_w + col[order]] = z[order]
        depth = depth.reshape(new_h, new_w)
        valid = float(np.isfinite(depth).mean())
        depth[~np.isfinite(depth)] = np.nan
        np.save(out_dir / (Path(fname).stem + ".npy"), depth)
        if n_done == 0:
            print(f"[ok] {fname}: ({h},{w}) -> ({new_h},{new_w}); depth coverage={valid:.1%}")
        n_done += 1

    print(f"\nWrote {n_done} GT depth maps to {out_dir} (max_resolution={args.max_resolution}).")


if __name__ == "__main__":
    main()
