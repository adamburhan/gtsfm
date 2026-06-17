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
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  
import matplotlib.pyplot as plt 
import numpy as np  
import torch  

from gtsfm.common.depth_provider import DepthProvider  
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


def _collect_ambiguous_samples(gtsfm_data, provider, global_indices):
    """Re-sample every track measurement and bucket valid/ambiguous points per image.

    Returns two dicts keyed by camera index: all valid (u, v) samples and the
    subset flagged ambiguous as (u, v, depth, depth_alt, score). Mirrors the loop
    in BA's `__depth_factors`, so the visualized points are exactly the ones that
    would receive a (bi)modal depth factor.
    """
    valid = {idx: [] for idx in global_indices}
    ambiguous = {idx: [] for idx in global_indices}
    for j in range(gtsfm_data.number_tracks()):
        track = gtsfm_data.get_track(j)
        for m in range(track.numberMeasurements()):
            cam_idx, uv = track.measurement(m)
            sample = provider.get_depth(int(cam_idx), float(uv[0]), float(uv[1]))
            if sample is None:
                continue
            valid[int(cam_idx)].append((float(uv[0]), float(uv[1])))
            if sample.ambiguous:
                ambiguous[int(cam_idx)].append(
                    (float(uv[0]), float(uv[1]), sample.depth, sample.depth_alt, sample.score)
                )
    return valid, ambiguous


def _visualize_ambiguous(image_batch, depth_arrays, valid, ambiguous, global_indices, image_names, out_dir):
    """Save per-image RGB|depth panels with ambiguous samples marked in red.

    RGB is the VGGT-processed image and depth is the in-memory map, both in the
    same pixel frame as the track uv, so an overlaid red dot lands on the exact
    pixel that was flagged. Red points should sit on visible depth discontinuities.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, cam_idx in enumerate(global_indices):
        amb = ambiguous[cam_idx]
        rgb = image_batch[k].detach().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0)
        depth = depth_arrays[cam_idx]
        finite = np.isfinite(depth)
        vmin, vmax = np.percentile(depth[finite], [2, 98]) if finite.any() else (0.0, 1.0)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].imshow(rgb)
        axes[0].set_title(f"{image_names[k]} — RGB")
        depth_disp = np.where(finite, depth, np.nan)
        im = axes[1].imshow(depth_disp, cmap="turbo", vmin=vmin, vmax=vmax)
        axes[1].set_title("VGGT depth")
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        for ax in axes:
            if valid[cam_idx]:
                vu, vv = zip(*valid[cam_idx])
                ax.scatter(vu, vv, s=2, c="white", alpha=0.25, linewidths=0)  # coverage context
            if amb:
                ax.scatter(
                    [p[0] for p in amb], [p[1] for p in amb],
                    s=28, facecolors="none", edgecolors="red", linewidths=1.4,
                )
            ax.set_xlim(0, depth.shape[1])
            ax.set_ylim(depth.shape[0], 0)
            ax.axis("off")

        fig.suptitle(f"cam {cam_idx}: {len(amb)} ambiguous / {len(valid[cam_idx])} valid samples")
        fig.tight_layout()
        save_path = out_dir / f"ambiguous_cam{cam_idx:03d}.png"
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  cam {cam_idx}: {len(amb):4d} ambiguous / {len(valid[cam_idx]):5d} valid -> {save_path}")


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
    parser.add_argument("--viz_dir", default="outputs/smoke_viz", help="Where to save RGB|depth ambiguity overlays.")
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

    # Visualize which measurements get flagged ambiguous, using the SAME thresholds
    # as ba_options, so the red overlays match the factors BA actually creates.
    viz_provider = DepthProvider(
        depth_arrays=depth_arrays,
        depth_min=ba_options.depth_min,
        depth_max=ba_options.depth_max,
        compute_hypotheses=True,
        patch_radius=ba_options.depth_patch_radius,
        gap_thresh=ba_options.depth_gap_thresh,
        ambiguity_thresh=ba_options.depth_ambiguity_thresh,
        min_valid=ba_options.depth_min_valid,
    )
    valid_samples, ambiguous_samples = _collect_ambiguous_samples(gtsfm_data, viz_provider, global_indices)
    total_amb = sum(len(v) for v in ambiguous_samples.values())
    print(f"ambiguity overlay ({total_amb} ambiguous samples total):")
    _visualize_ambiguous(
        image_batch, depth_arrays, valid_samples, ambiguous_samples, global_indices, image_names, args.viz_dir
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
