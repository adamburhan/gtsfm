"""Merge ETH3D laser scans into one GT point cloud in the shared ground-truth frame.

ETH3D ships per-position scans (scan_clean/scan*.ply) plus a MeshLab project
(scan_alignment.mlp) giving each scan's rigid transform into the common frame — the
same frame as the dslr_calibration GT poses. This writes a single merged, transformed
.ply to pass to eval_geometry.py --gt_ply.

Usage:
    python scripts/eth3d_merge_scans.py <scene>/scan_clean/scan_alignment.mlp \
        --out <scene>_gt.ply [--voxel 0.01]
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import open3d as o3d  # type: ignore


def parse_mlp(mlp_path: str) -> list[tuple[Path, np.ndarray]]:
    """Return [(scan_ply_path, 4x4 mesh->world transform)] from a MeshLab project."""
    base = Path(mlp_path).parent
    out = []
    for mesh in ET.parse(mlp_path).getroot().iter("MLMesh"):
        mat = mesh.find("MLMatrix44")
        T = np.fromstring(mat.text.replace("\n", " "), sep=" ").reshape(4, 4)
        out.append((base / mesh.get("filename"), T))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mlp", help="Path to scan_alignment.mlp")
    p.add_argument("--out", required=True, help="Output merged .ply")
    p.add_argument("--voxel", type=float, default=0.0, help="Optional voxel-downsample size in meters (0 = off)")
    args = p.parse_args()

    clouds = []
    for ply_path, T in parse_mlp(args.mlp):
        pcd = o3d.io.read_point_cloud(str(ply_path))
        pcd.transform(T)
        clouds.append(np.asarray(pcd.points))
        print(f"{ply_path.name}: {len(pcd.points):,} pts")

    merged = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack(clouds)))
    if args.voxel > 0:
        merged = merged.voxel_down_sample(args.voxel)
    o3d.io.write_point_cloud(args.out, merged)
    print(f"Wrote {len(merged.points):,} pts -> {args.out}")


if __name__ == "__main__":
    main()
