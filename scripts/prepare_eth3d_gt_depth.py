"""Convert ETH3D ground-truth depth maps to loader-resolution .npy for the depth-factor pipeline.

ETH3D's high-res multi-view training scenes ship per-image GT depth in ``ground_truth_depth/`` as a
*raw little-endian float32* blob (row-major, H*W, no header), dims = the undistorted image, invalid
pixels = +inf. This reads each, marks invalid as NaN, resizes to the loader's working resolution
(short side <= max_resolution, nearest-neighbor so discontinuities aren't blended — same grid as
``resize_depthpro_to_loader.py``), and writes ``<image_stem>.npy`` so the DepthProvider's
``template=null`` + ``depth_ext=.npy`` path picks it up. Use as the GT-depth-as-source oracle.

The format is verified per file (size == H*W*4); a mismatch aborts loudly rather than reshaping
garbage. If your ground_truth_depth turns out to be a different format, the error tells you.

Usage:
    python scripts/prepare_eth3d_gt_depth.py \
        --gt_depth_dir $DATA/kicker/ground_truth_depth \
        --images_dir   $DATA/kicker/images/dslr_images_undistorted \
        --out_dir      $DATA/kicker/gt_depth_760 \
        --max_resolution 760

Authors: Adam Burhan
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from gtsfm.utils.images import get_downsampling_factor_per_axis


def _find_gt_file(gt_dir: Path, image_path: Path) -> Path | None:
    """Locate the GT depth blob for an image (ETH3D names them by image name or stem)."""
    for cand in (gt_dir / image_path.name, gt_dir / image_path.stem):
        if cand.exists():
            return cand
    matches = list(gt_dir.glob(image_path.stem + "*"))
    return matches[0] if matches else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt_depth_dir", required=True, help="ETH3D ground_truth_depth dir (raw float32 blobs)")
    ap.add_argument("--images_dir", required=True, help="Undistorted images dir (for dims + filenames)")
    ap.add_argument("--out_dir", required=True, help="Output dir for loader-resolution .npy")
    ap.add_argument("--max_resolution", type=int, default=760, help="Must match the GTSfM loader --max_resolution")
    args = ap.parse_args()

    gt_dir, img_dir, out_dir = Path(args.gt_depth_dir), Path(args.images_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not images:
        raise SystemExit(f"No images in {img_dir}")

    n_done = 0
    for img_path in images:
        gt_path = _find_gt_file(gt_dir, img_path)
        if gt_path is None:
            print(f"WARN: no GT depth for {img_path.name}; skipping")
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"WARN: cannot read image {img_path.name}; skipping")
            continue
        h, w = img.shape[:2]

        raw = np.fromfile(gt_path, dtype="<f4")
        if raw.size != h * w:
            raise SystemExit(
                f"{gt_path.name}: {raw.size} floats != {h}x{w}={h * w} (image dims). "
                f"The GT depth is not raw float32 at the image resolution — inspect the format."
            )
        depth = raw.reshape(h, w).astype(np.float32)
        depth[~np.isfinite(depth) | (depth <= 0)] = np.nan  # ETH3D marks invalid as +inf

        if min(h, w) > args.max_resolution:
            _, _, new_h, new_w = get_downsampling_factor_per_axis(h, w, args.max_resolution)
            depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        valid = float(np.isfinite(depth).mean())
        np.save(out_dir / (img_path.stem + ".npy"), depth)
        if n_done == 0:
            print(f"[format OK] {gt_path.name}: {h}x{w} float32 -> {depth.shape}, valid={valid:.1%}")
        n_done += 1

    print(f"\nWrote {n_done} GT depth maps to {out_dir} (max_resolution={args.max_resolution}).")


if __name__ == "__main__":
    main()
