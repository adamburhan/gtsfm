"""Optional per-image depth maps for the depth bundle-adjustment factor.

Reads dense depth maps from disk and samples them at track-measurement pixels.
Keyed by GTSfM image index; the index->filename map is supplied by the caller
(bundle adjustment receives it from the multi-view optimizer's per-view data),
because the `GtsfmData` reaching BA does not yet carry image filenames.

Depth is interpreted as camera-frame planar Z in meters (the same quantity the
depth factor predicts via `wTc.transformTo(point_w).z`). For uint16 PNG depth
(e.g. Replica) the raw values are divided by `depth_scale` to recover meters.

When hypothesis extraction is enabled, each sample is additionally analyzed for
depth ambiguity near discontinuities (largest gap in log-depth over a local
patch), yielding an optional second depth hypothesis for the bimodal
(max-mixture) depth factor.

Authors: Adam Burhan
"""

import re
from pathlib import Path
from typing import Dict, NamedTuple, Optional

import cv2
import numpy as np

import gtsfm.utils.logger as logger_utils

logger = logger_utils.get_logger()


class DepthSample(NamedTuple):
    """A depth measurement at a pixel, with optional second hypothesis.

    Attributes:
        depth: Depth (meters) at the pixel.
        depth_alt: Second depth hypothesis (meters) when the sample is ambiguous, else None.
        ambiguous: Whether the pixel lies near a depth discontinuity (fg/bg ambiguity).
        score: Ambiguity score (largest log-depth gap, weighted by mode balance).
    """

    depth: float
    depth_alt: Optional[float]
    ambiguous: bool
    score: float


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
        compute_hypotheses: bool = False,
        patch_radius: int = 5,
        gap_thresh: float = 0.15,
        ambiguity_thresh: float = 0.20,
        min_valid: int = 10,
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
            compute_hypotheses: Analyze a local patch around each sample for
                fg/bg depth ambiguity (largest-gap method) and extract a second
                depth hypothesis. Required by the `drop_ambiguous` and `bimodal`
                depth-factor modes.
            patch_radius: Half-size of the square patch used for ambiguity analysis.
            gap_thresh: Minimum largest gap in sorted log-depths for a patch to be
                considered bimodal.
            ambiguity_thresh: Minimum ambiguity score (gap * mode balance) to flag
                a sample as ambiguous.
            min_valid: Minimum number of valid depths in the patch to attempt the
                analysis.
        """
        self._dir = Path(depth_map_dir)
        self._fnames = image_fnames
        self._depth_min = depth_min
        self._depth_max = depth_max
        self._depth_scale = depth_scale
        self._ext = depth_ext
        self._template = depth_filename_template
        self._compute_hypotheses = compute_hypotheses
        self._patch_radius = patch_radius
        self._gap_thresh = gap_thresh
        self._ambiguity_thresh = ambiguity_thresh
        self._min_valid = min_valid
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

    def get_depth(self, image_id: int, u: float, v: float) -> Optional[DepthSample]:
        """Sample the depth (meters) at pixel (u, v), or None if missing/invalid.

        Args:
            image_id: GTSfM image index.
            u: Horizontal pixel coordinate (column).
            v: Vertical pixel coordinate (row).

        Returns:
            DepthSample with depth in [depth_min, depth_max] (and, when
            `compute_hypotheses` is set, the ambiguity analysis), else None.
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
        if not self._compute_hypotheses:
            return DepthSample(depth=d, depth_alt=None, ambiguous=False, score=0.0)
        return self._analyze_patch(depth_map, row, col, d)

    def _analyze_patch(self, depth_map: np.ndarray, row: int, col: int, d_center: float) -> DepthSample:
        """Largest-gap ambiguity analysis on the patch around (row, col).

        Sorts the valid log-depths in the patch and finds the largest gap. If the
        gap is wide and the two resulting modes are balanced, the sample is near a
        depth discontinuity: it is flagged ambiguous, and the median of the mode
        farther (in log-depth) from the center depth becomes the second hypothesis.
        """
        r = self._patch_radius
        h, w = depth_map.shape[:2]
        patch = depth_map[max(0, row - r) : min(h, row + r + 1), max(0, col - r) : min(w, col + r + 1)]
        valid = patch[np.isfinite(patch) & (patch >= self._depth_min) & (patch <= self._depth_max)]
        if valid.size < self._min_valid:
            return DepthSample(depth=d_center, depth_alt=None, ambiguous=False, score=0.0)

        logs = np.sort(np.log(valid))
        gaps = np.diff(logs)
        if gaps.size == 0:
            return DepthSample(depth=d_center, depth_alt=None, ambiguous=False, score=0.0)

        k = int(np.argmax(gaps))
        max_gap = float(gaps[k])
        if max_gap < self._gap_thresh:
            return DepthSample(depth=d_center, depth_alt=None, ambiguous=False, score=0.0)

        split = 0.5 * (logs[k] + logs[k + 1])
        near = valid[np.log(valid) <= split]
        far = valid[np.log(valid) > split]
        d_near, d_far = float(np.median(near)), float(np.median(far))
        log_c = np.log(d_center)
        d_alt = d_far if abs(log_c - np.log(d_near)) <= abs(log_c - np.log(d_far)) else d_near

        frac_far = far.size / valid.size
        balance = 1.0 - abs(0.5 - frac_far) * 2.0
        score = max_gap * balance
        if score >= self._ambiguity_thresh:
            return DepthSample(depth=d_center, depth_alt=d_alt, ambiguous=True, score=score)
        return DepthSample(depth=d_center, depth_alt=None, ambiguous=False, score=score)
