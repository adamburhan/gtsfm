"""Smoke test: VGGT in-memory depth -> bimodal depth factors in cluster BA.

Exercises the real cluster code path (crop-mode preprocessing, `_run_vggt_pipeline`
with depth extraction, `_run_cluster_ba` with depth factors) on a tiny scene, with
NO depth files on disk -- depth comes from VGGT in memory.

Run on a CUDA box:
    python scripts/smoke_test_vggt_depth_factors.py \
        --scene_dir tests/data/astrovision/test_2011212_opnav_022

PASS criteria (printed at the end):
  * depth_arrays populated, one map per image, finite values present
  * BA logs "Depth factors (bimodal): ... N bimodal ..." (the provider fired)
  * cluster BA returns a non-empty reconstruction without error
"""

import argparse

import numpy as np
import torch

from gtsfm.bundle.bundle_adjustment import BundleAdjustmentOptions
from gtsfm.cluster_optimizer.cluster_vggt import _run_cluster_ba, _run_vggt_pipeline
from gtsfm.frontend.multi_view_tracker import MultiViewTracker, TrackingConfig
from gtsfm.frontend.vggt_geometry_transformer import (
    VggtGeometryConfig,
    VggtGeometryTransformer,
    load_image_batch_vggt_loader,
)
from gtsfm.loader.astrovision_loader import AstrovisionLoader
from gtsfm.loader.mobilebrick_loader import MobilebrickLoader
from gtsfm.loader.replica_loader import ReplicaLoader


def _build_loader(args: argparse.Namespace):
    """Instantiate the requested loader at VGGT's 518 resolution."""
    if args.loader == "astrovision":
        return AstrovisionLoader(dataset_dir=args.scene_dir, max_resolution=518)
    if args.loader == "mobilebrick":
        # Object-against-background scene -> fg/bg discontinuities -> exercises bimodal.
        return MobilebrickLoader(dataset_dir=args.scene_dir, max_resolution=518)
    if args.loader == "replica":
        # Indoor scenes with sharp object/wall boundaries. Stride gives the few
        # frames real baseline (consecutive 30fps frames are near-degenerate).
        return ReplicaLoader(
            dataset_dir=args.scene_dir,
            sequence=args.sequence,
            stride=args.stride,
            max_frames=args.max_frames,
            max_resolution=518,
        )
    raise ValueError(f"Unknown loader: {args.loader}")


def main() -> None:
    parser = argparse.ArgumentParser(description="VGGT depth-factor smoke test")
    parser.add_argument("--scene_dir", required=True, help="Scene/dataset root for the chosen loader.")
    parser.add_argument(
        "--loader",
        default="astrovision",
        choices=["astrovision", "mobilebrick", "replica"],
        help="mobilebrick/replica have depth discontinuities and are more likely to produce bimodal factors.",
    )
    parser.add_argument("--max_query_pts", type=int, default=512, help="Lower = fewer depth factors = faster BA.")
    parser.add_argument("--sequence", default="office0", help="Replica sequence name (replica loader only).")
    parser.add_argument("--stride", type=int, default=20, help="Replica frame stride (replica loader only).")
    parser.add_argument("--max_frames", type=int, default=6, help="Replica max frames after striding (replica only).")
    args = parser.parse_args()

    loader = _build_loader(args)
    global_indices = tuple(range(len(loader)))
    image_names = tuple(loader.image_filenames())
    print(f"Loaded {len(global_indices)} images from {args.scene_dir}")

    # Real cluster preprocessing (crop mode), exactly as _load_vggt_inputs does.
    image_batch, original_coords = load_image_batch_vggt_loader(loader, list(global_indices), mode="crop")

    transformer = VggtGeometryTransformer(VggtGeometryConfig(confidence_threshold=5.0))
    tracker = MultiViewTracker(
        TrackingConfig(
            tracking=True,
            max_query_pts=args.max_query_pts,
            query_frame_num=3,
            keypoint_extractor="aliked+sp+sift",
            # Mirror the production vggt configs' permissive filtering: the defaults
            # (reproj 14px, min angle 10deg) drop all tracks on small-baseline object
            # scans like MobileBrick.
            vggt_max_reproj_error=0.0,
            min_triangulation_angle=0.0,
        )
    )

    # Step 1: VGGT pipeline with depth extraction (the new path).
    gtsfm_data, depth_arrays = _run_vggt_pipeline(
        image_batch,
        original_coords,
        transformer=transformer,
        tracker=tracker,
        image_indices=global_indices,
        image_names=image_names,
        seed=42,
        model_cache_key=("smoke", None),  # non-None -> model is loaded and shared with the tracker
        loader_kwargs={},
        weights_path=None,
        cluster_label="smoke",
        extract_depth=True,
    )

    assert depth_arrays is not None, "depth_arrays is None despite extract_depth=True"
    assert set(depth_arrays.keys()) == set(global_indices), "depth_arrays keys != camera indices"
    for idx, d in depth_arrays.items():
        assert d.ndim == 2 and np.isfinite(d).any(), f"bad depth map for cam {idx}: shape={d.shape}"
    shapes = {idx: d.shape for idx, d in depth_arrays.items()}
    print(f"depth_arrays OK: {len(depth_arrays)} maps, shapes={shapes}")
    print(f"pre-BA reconstruction: {gtsfm_data.number_tracks()} tracks, "
          f"{len(gtsfm_data.get_valid_camera_indices())} cameras")

    # Step 2: cluster BA with bimodal depth factors fed from the in-memory maps.
    # VGGT depth is cluster-local scale (not metric), so use a wide validity range.
    ba_options = BundleAdjustmentOptions(
        shared_calib=True,
        use_calibration_prior=False,
        depth_model="bimodal",
        depth_factor_sigma=0.1,
        depth_min=0.0,
        depth_max=1e9,
        depth_gap_thresh=0.08
    )
    image_fnames = {idx: name for idx, name in zip(global_indices, image_names)}

    post_ba, _pre_ba = _run_cluster_ba(
        gtsfm_data,
        ba_options=ba_options,
        depth_arrays=depth_arrays,
        image_fnames=image_fnames,
        pre_ba_max_reproj_error=14.0,
        post_ba_max_reproj_error=3.0,
        min_track_length=2,
        cluster_label="smoke",
    )

    assert post_ba.number_tracks() > 0, "post-BA reconstruction is empty"
    print(f"post-BA reconstruction: {post_ba.number_tracks()} tracks, "
          f"{len(post_ba.get_valid_camera_indices())} cameras")
    print("\nSMOKE TEST PASS -- look for a 'Depth factors (bimodal): ...' log line above "
          "with a nonzero bimodal/unimodal count to confirm the factors fired.")


if __name__ == "__main__":
    with torch.no_grad():
        main()
