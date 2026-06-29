"""Resize Depth Pro depth maps to GTSfM's loader resolution so they align with track uv.

GTSfM's loader downsamples each image's short side to ``max_resolution`` (preserving aspect, see
``gtsfm.utils.images.get_downsampling_factor_per_axis``), and keypoint uv live in that grid. Depth
Pro maps are produced at full image resolution, so a raw lookup would index the wrong pixel. This
applies the *identical* target dimensions to each depth map — but with nearest-neighbor sampling, so
depth discontinuities (the fg/bg structure the bimodal factor relies on) are preserved rather than
blended (the loader uses cubic for RGB, which is wrong for depth).

Assumes each depth ``.npy`` is at the same full resolution as its image. Maps already at/below
``max_resolution`` on the short side are copied unchanged (the loader would not resize either).

Usage:
    python scripts/resize_depthpro_to_loader.py --in_dir <depth_pro> --out_dir <depth_pro_760> --max_resolution 760

Authors: Adam Burhan
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

from gtsfm.utils.images import get_downsampling_factor_per_axis


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in_dir", required=True, help="Directory of full-res Depth Pro .npy maps")
    ap.add_argument("--out_dir", required=True, help="Output directory for loader-resolution maps")
    ap.add_argument("--max_resolution", type=int, default=760, help="Must match the GTSfM loader --max_resolution")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npys = sorted(in_dir.glob("*.npy"))
    if not npys:
        raise SystemExit(f"No .npy files in {in_dir}")

    for p in npys:
        depth = np.load(p)
        h, w = depth.shape[:2]
        if min(h, w) <= args.max_resolution:
            shutil.copy2(p, out_dir / p.name)  # loader would not resize this one either
            continue
        _, _, new_h, new_w = get_downsampling_factor_per_axis(h, w, args.max_resolution)
        # Nearest-neighbor: preserve depth discontinuities; cv2.resize takes (width, height).
        resized = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        np.save(out_dir / p.name, resized)
        print(f"{p.name}: ({h},{w}) -> ({new_h},{new_w})")

    print(f"\nWrote {len(npys)} maps to {out_dir} (max_resolution={args.max_resolution}).")


if __name__ == "__main__":
    main()
