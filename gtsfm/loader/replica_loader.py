"""Loader for the Replica dataset (rendered RGB-D sequences with GT poses).

Reads a single sequence laid out as:
    {dataset_dir}/cam_params.json          # shared pinhole intrinsics (fx, fy, cx, cy)
    {dataset_dir}/{sequence}/traj.txt      # N rows of 16 = flattened 4x4 cam-to-world (wTc)
    {dataset_dir}/{sequence}/results/frame{idx:06d}.jpg
    {dataset_dir}/{sequence}/results/depth{idx:06d}.png   # sampled separately by DepthProvider
    {dataset_dir}/{sequence}_mesh.ply      # optional GT surface, for evaluation

Depth maps are NOT read here: the depth bundle-adjustment factor samples them via
`gtsfm.common.depth_provider.DepthProvider`, pointed at the `results/` directory.

Note on pose convention: Replica `traj.txt` stores cam-to-world transforms, which is
exactly GTSfM's `wTi` convention, so poses are used directly (no inversion).

Authors: Adam Burhan
"""

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import trimesh
from gtsam import Cal3_S2, Pose3, Rot3  # type: ignore
from trimesh import Trimesh

import gtsfm.utils.io as io_utils
import gtsfm.utils.logger as logger_utils
from gtsfm.common.image import Image
from gtsfm.loader.loader_base import LoaderBase

logger = logger_utils.get_logger()


class ReplicaLoader(LoaderBase):
    """Loader class that reads a single Replica sequence from disk."""

    def __init__(
        self,
        dataset_dir: str,
        sequence: str,
        stride: int = 1,
        max_frames: Optional[int] = None,
        max_resolution: int = 760,
        use_gt_intrinsics: bool = True,
        max_frame_lookahead: Optional[int] = None,
        input_worker: Optional[str] = None,
    ) -> None:
        """Initialize loader for a Replica sequence.

        Args:
            dataset_dir: Path to the Replica root (containing `cam_params.json` and sequences).
            sequence: Sequence name, e.g. "office0".
            stride: Keep every `stride`-th frame.
            max_frames: If set, keep at most this many frames (after striding).
            max_resolution: Maximum length of the image's short side. Keep >= the Replica
                short side (680) so images are not downsampled relative to the depth maps.
            use_gt_intrinsics: Use the dataset intrinsics (always True for Replica; rendered).
            max_frame_lookahead: If set, only pairs within this frame-index gap are valid.
            input_worker: Optional Dask worker address for image I/O.
        """
        super().__init__(max_resolution, input_worker)

        self._dataset_dir = Path(dataset_dir)
        self._sequence = sequence
        self._seq_dir = self._dataset_dir / sequence
        self._images_dir = self._seq_dir / "results"
        self._use_gt_intrinsics = use_gt_intrinsics
        self._max_frame_lookahead = max_frame_lookahead

        # Shared pinhole intrinsics (no distortion; images are rendered).
        cam = json.loads((self._dataset_dir / "cam_params.json").read_text())["camera"]
        self._K = Cal3_S2(cam["fx"], cam["fy"], 0.0, cam["cx"], cam["cy"])

        # Ground-truth poses: each row is a flattened 4x4 cam-to-world (wTc == wTi).
        traj = np.loadtxt(self._seq_dir / "traj.txt").reshape(-1, 4, 4)

        # Enumerate frames present on disk, then apply stride / max_frames.
        frame_ids = sorted(int(p.stem[len("frame"):]) for p in self._images_dir.glob("frame*.jpg"))
        frame_ids = frame_ids[::stride]
        if max_frames is not None:
            frame_ids = frame_ids[:max_frames]
        if len(frame_ids) == 0:
            raise ValueError(f"No frames found in {self._images_dir}")

        self._frame_ids = frame_ids
        self._image_paths = [self._images_dir / f"frame{i:06d}.jpg" for i in frame_ids]
        self._wTi = [Pose3(Rot3(traj[i, :3, :3]), traj[i, :3, 3]) for i in frame_ids]

        self._mesh_path = self._dataset_dir / f"{sequence}_mesh.ply"
        logger.info("ReplicaLoader: %s, %d frames (stride=%d).", sequence, len(frame_ids), stride)

    def image_filenames(self) -> List[str]:
        """Return the file names corresponding to each image index."""
        return [p.name for p in self._image_paths]

    def __len__(self) -> int:
        """The number of images in the loaded sequence."""
        return len(self._image_paths)

    def get_image_full_res(self, index: int) -> Image:
        """Get the image at the given index, at full resolution."""
        if index < 0 or index >= len(self):
            raise IndexError(f"Image index {index} is invalid")
        img = io_utils.load_image(str(self._image_paths[index]))
        return Image(value_array=img.value_array, exif_data=img.exif_data, file_name=img.file_name)

    def get_camera_intrinsics_full_res(self, index: int) -> Optional[Cal3_S2]:
        """Get the (shared) camera intrinsics at the given index, for a full-res image."""
        if index < 0 or index >= len(self):
            raise IndexError(f"Image index {index} is invalid")
        return self._K

    def get_camera_pose(self, index: int) -> Optional[Pose3]:
        """Get the ground-truth camera pose wTi (cam-to-world) at the given index."""
        if index < 0 or index >= len(self):
            raise IndexError(f"Image index {index} is invalid")
        return self._wTi[index]

    def is_valid_pair(self, idx1: int, idx2: int) -> bool:
        """Checks if (idx1, idx2) is a valid pair (idx1 < idx2 and within lookahead)."""
        valid = super().is_valid_pair(idx1, idx2)
        if self._max_frame_lookahead is not None:
            valid = valid and abs(idx1 - idx2) <= self._max_frame_lookahead
        return valid

    def get_gt_scene_trimesh(self) -> Optional[Trimesh]:
        """Return the GT scene mesh for evaluation, if present."""
        if not self._mesh_path.exists():
            return None
        return trimesh.load(self._mesh_path, process=False, maintain_order=True)
