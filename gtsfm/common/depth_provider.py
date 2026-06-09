"""Optional per-image depth maps for the depth bundle-adjustment factor.

Reads dense depth maps from disk and samples them at track-measurement pixels.
Keyed by GTSfM image index; the index->filename map is supplied by the caller
(bundle adjustment receives it from the multi-view optimizer's per-view data),
because the `GtsfmData` reaching BA does not yet carry image filenames.

Depth is interpreted as camera-frame planar Z in meters (the same quantity the
depth factor predicts via `wTc.transformTo(point_w).z`). For uint16 PNG depth
(e.g. Replica) the raw values are divided by `depth_scale` to recover meters.

This is the first, unimodal contribution: a single depth value per pixel. The
bimodal/max-mixture hypothesis logic is added in a later stage.

Authors: Adam Burhan
"""

import re
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

import gtsfm.utils.logger as logger_utils

logger = logger_utils.get_logger()


class DepthProvider:
    """Reads native-resolution depth maps and samples them at pixel locations.

    Holds only paths + the index->filename map, so it is cheap to ship into the
    delayed BA task; depth arrays are loaded lazily and cached per process.

    Note: v1 assumes the depth map resolution matches the resolution the track
    pixels (`uv`) live in (true for Replica at the default loader resolution,
    where no downsampling occurs). If images are downsampled relative to the
    depth maps, a uv->depth scale mapping must be added here.
    """

    def __init__(
        self,
        depth_map_dir: str,
        image_fnames: Dict[int, str],
        *,
        depth_min: float,
        depth_max: float,
        depth_scale: float = 1.0,
        depth_ext: str = ".png",
        depth_filename_template: Optional[str] = "depth{:06d}.png",
    ) -> None:
        """
        Args:
            depth_map_dir: Directory containing the per-image depth maps.
            image_fnames: Map from GTSfM image index to image filename.
            depth_min: Minimum valid depth in meters; samples below are dropped.
            depth_max: Maximum valid depth in meters; samples above are dropped.
            depth_scale: Divisor applied to raw depth values to recover meters
                (e.g. Replica's uint16 PNG scale). Use 1.0 for float depth.
            depth_ext: Extension fallback used when `depth_filename_template` is
                None; the depth file is then `<image-stem><depth_ext>`.
            depth_filename_template: Template applied to the integer parsed from
                the trailing digits of the image filename stem. Replica images
                are `frame000123.jpg` and depth maps `depth000123.png`, so the
                default maps one to the other. Set to None to mirror the image
                stem with `depth_ext` instead.
        """
        self._dir = Path(depth_map_dir)
        self._fnames = image_fnames
        self._depth_min = depth_min
        self._depth_max = depth_max
        self._depth_scale = depth_scale
        self._ext = depth_ext
        self._template = depth_filename_template
        self._cache: Dict[int, Optional[np.ndarray]] = {}

    def _depth_filename(self, image_id: int) -> str:
        """Derive the depth filename for an image index from its image filename."""
        stem = Path(self._fnames[image_id]).stem
        if self._template is not None:
            match = re.search(r"(\d+)$", stem)
            if match is not None:
                return self._template.format(int(match.group(1)))
        return stem + self._ext

    def _depth_map(self, image_id: int) -> Optional[np.ndarray]:
        """Lazily load and cache the (native-resolution) depth map in meters."""
        if image_id not in self._cache:
            path = self._dir / self._depth_filename(image_id)
            if not path.exists():
                logger.warning("DepthProvider: no depth map for image %d at %s", image_id, path)
                self._cache[image_id] = None
            elif path.suffix == ".npy":
                self._cache[image_id] = np.load(path).astype(np.float64) / self._depth_scale
            else:
                raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                self._cache[image_id] = None if raw is None else raw.astype(np.float64) / self._depth_scale
        return self._cache[image_id]

    def get_depth(self, image_id: int, u: float, v: float) -> Optional[float]:
        """Sample the depth (meters) at pixel (u, v), or None if missing/invalid.

        Args:
            image_id: GTSfM image index.
            u: Horizontal pixel coordinate (column).
            v: Vertical pixel coordinate (row).

        Returns:
            Depth in meters within [depth_min, depth_max], else None.
        """
        depth_map = self._depth_map(image_id)
        if depth_map is None:
            return None
        h, w = depth_map.shape[:2]
        col = int(np.clip(round(u), 0, w - 1))
        row = int(np.clip(round(v), 0, h - 1))
        d = float(depth_map[row, col])
        if not np.isfinite(d) or d < self._depth_min or d > self._depth_max:
            return None
        return d
