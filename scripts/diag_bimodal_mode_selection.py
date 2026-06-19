"""Diagnostic: does the bimodal depth factor ever select its second hypothesis?

After a completed GTSfM run with depth_model=bimodal, re-runs the DepthProvider
on the final ba_output to identify which measurements were bimodal candidates
(sample.ambiguous=True), then checks which mode the optimizer converged to by
comparing z_pred (final camera-frame depth from optimized cameras + points)
against d (primary hypothesis) and d_alt (secondary hypothesis).

Answers two questions from the post-hoc analysis:
  1. Does the optimizer ever end up at the non-naive mode (mode 2)?
  2. When it does, does that landmark land closer to the GT surface?

Question 2 is answered as a per-landmark counterfactual: for every landmark that
switched to mode 2 in the bimodal run, compare its distance-to-GT against the
same landmark's position in the unimodal run (--unimodal_ba_dir). This avoids
the confound of comparing different landmarks (mode-1 vs mode-2 are different
points in different parts of the scene).

Note: bimodal and unimodal runs start from identical frontends and data
association, so track index j is the same landmark in both ba_outputs.

Usage:
    python scripts/diag_bimodal_mode_selection.py \\
        --ba_dir /path/to/bimodal/results/ba_output \\
        --depth_map_dir /path/to/Replica/office0/results \\
        --gap_thresh 0.10 \\
        [--unimodal_ba_dir /path/to/unimodal/results/ba_output] \\
        [--gt_ply /path/to/office0_mesh.ply] \\
        [--gt_traj /path/to/Replica/office0/traj.txt]

Authors: Adam Burhan
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gtsfm.common.depth_provider import DepthProvider
from gtsfm.common.gtsfm_data import GtsfmData


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _build_image_fnames(data: GtsfmData) -> dict[int, str]:
    """Map GTSfM camera index → image filename from the COLMAP images.txt."""
    indices = data.get_valid_camera_indices()
    names = data.image_filenames()
    return {idx: name for idx, name in zip(indices, names) if name is not None}


def _analyze(data: GtsfmData, provider: DepthProvider) -> dict:
    """Check mode selection for every bimodal-candidate measurement.

    Returns summary stats and a list of per-measurement record dicts.
    """
    records = []
    n_mode1 = n_mode2 = 0

    for j in range(data.number_tracks()):
        track = data.get_track(j)
        point_w = np.array(track.point3())

        for m_idx in range(track.numberMeasurements()):
            i, uv = track.measurement(m_idx)
            camera = data.get_camera(i)
            if camera is None:
                continue

            sample = provider.get_depth(i, float(uv[0]), float(uv[1]))
            if sample is None or not sample.ambiguous or sample.depth_alt is None:
                continue

            # Mirror make_bimodal_depth_factor exactly.
            point_c = camera.pose().transformTo(point_w)
            z_pred = float(point_c[2])
            r1 = abs(z_pred - sample.depth)
            r2 = abs(z_pred - sample.depth_alt)
            mode = 2 if r2 < r1 else 1

            if mode == 1:
                n_mode1 += 1
            else:
                n_mode2 += 1

            records.append({
                "track_idx": j,
                "cam_idx": i,
                "d": sample.depth,
                "d_alt": sample.depth_alt,
                "z_pred": z_pred,
                "r1": r1,
                "r2": r2,
                "mode": mode,
                "point_w": point_w.copy(),
            })

    return {"n_candidates": len(records), "n_mode1": n_mode1, "n_mode2": n_mode2, "records": records}


def _load_gt_surface(gt_ply: str, n_samples: int = 500_000) -> np.ndarray:
    """Load GT surface as a dense point cloud. Uses trimesh first (handles quads)."""
    import trimesh
    import trimesh.sample

    tm = trimesh.load(gt_ply, process=False, force="mesh")
    faces = getattr(tm, "faces", None)
    if faces is not None and len(faces) > 0:
        pts = trimesh.sample.sample_surface(tm, n_samples)[0]
        return np.asarray(pts, dtype=np.float64)
    # Quad / degenerate mesh: fall back to raw vertices (still a valid surface proxy).
    print("[mesh] WARNING: trimesh found no triangulated faces; using raw vertices as GT surface.")
    vertices = getattr(tm, "vertices", None)
    if vertices is None:
        raise RuntimeError(f"Could not extract any geometry from {gt_ply}")
    return np.asarray(vertices, dtype=np.float64)


def _build_gt_tree(gt_ply: str, gt_traj: str, data: GtsfmData):
    """Return (KDTree of GT surface in world frame, wSr Sim3 aligning recon→world)."""
    from scipy.spatial import cKDTree  # type: ignore[import-untyped]
    from gtsfm.evaluation.eval_geometry_vs_mesh import align_recon_to_world

    gt_pts = _load_gt_surface(gt_ply)
    print(f"[mesh] GT surface: {len(gt_pts):,} sampled points")

    wTi_list = []
    for i in data.get_valid_camera_indices():
        cam = data.get_camera(i)
        wTi_list.append(cam.pose() if cam is not None else None)
    img_fnames = [data.get_image_info(i).name for i in data.get_valid_camera_indices()]
    wSr, rms = align_recon_to_world(wTi_list, img_fnames, gt_traj)
    print(f"[mesh] Sim3 alignment RMS camera-centre residual: {rms*100:.2f} cm")

    return cKDTree(gt_pts), wSr


def _dist_to_gt(points_world: np.ndarray, tree, wSr) -> np.ndarray:  # type: ignore[type-arg]
    aligned = np.array([wSr.transformFrom(p) for p in points_world])
    return tree.query(aligned, k=1)[0]  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ba_dir", required=True,
                        help="Bimodal run: COLMAP-format BA output dir.")
    parser.add_argument("--depth_map_dir", required=True,
                        help="Directory containing per-image depth PNGs (e.g. Replica/<seq>/results).")
    parser.add_argument("--gap_thresh", type=float, default=0.05,
                        help="Log-depth gap threshold used during the sweep (default: 0.05).")
    parser.add_argument("--depth_scale", type=float, default=6553.5,
                        help="uint16→metres divisor (default: 6553.5 for Replica).")
    parser.add_argument("--depth_min", type=float, default=0.1)
    parser.add_argument("--depth_max", type=float, default=20.0)
    parser.add_argument("--patch_radius", type=int, default=5)
    parser.add_argument("--depth_filename_template", default="depth{:06d}.png")
    # Counterfactual comparison.
    parser.add_argument("--unimodal_ba_dir", default=None,
                        help="Unimodal run: BA output dir (same scene/frontend). "
                             "Enables per-landmark counterfactual comparison.")
    # GT geometry.
    parser.add_argument("--gt_ply", default=None,
                        help="GT mesh PLY for point-to-surface distances.")
    parser.add_argument("--gt_traj", default=None,
                        help="Replica traj.txt for Sim3 alignment (required with --gt_ply).")
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Load bimodal reconstruction                                         #
    # ------------------------------------------------------------------ #
    print(f"Loading bimodal BA output: {args.ba_dir}")
    data = GtsfmData.read_colmap(args.ba_dir)
    print(f"  {len(data.get_valid_camera_indices())} cameras, {data.number_tracks()} tracks")

    image_fnames = _build_image_fnames(data)

    # ------------------------------------------------------------------ #
    # Build DepthProvider (same settings as the sweep)                   #
    # ------------------------------------------------------------------ #
    provider = DepthProvider(
        depth_map_dir=args.depth_map_dir,
        image_fnames=image_fnames,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        depth_scale=args.depth_scale,
        depth_filename_template=args.depth_filename_template,
        compute_hypotheses=True,
        patch_radius=args.patch_radius,
        gap_thresh=args.gap_thresh,
        ambiguity_thresh=0.0,  # currently unused in _analyze_patch
        min_valid=3,           # hardcoded in _analyze_patch
    )

    # ------------------------------------------------------------------ #
    # Mode-selection analysis                                             #
    # ------------------------------------------------------------------ #
    print(f"\nAnalyzing mode selection (gap_thresh={args.gap_thresh}) ...")
    result = _analyze(data, provider)
    n_cand = result["n_candidates"]
    n1, n2 = result["n_mode1"], result["n_mode2"]
    records = result["records"]

    print(f"\n{'='*60}")
    print(f"Bimodal-candidate measurements : {n_cand}")
    if n_cand == 0:
        print("No bimodal candidates found. Check --gap_thresh or --depth_map_dir.")
        return
    print(f"  Mode 1 (primary d)  selected : {n1}  ({100*n1/n_cand:.1f}%)")
    print(f"  Mode 2 (alt  d_alt) selected : {n2}  ({100*n2/n_cand:.1f}%)")
    print(f"{'='*60}\n")

    r1s = np.array([r["r1"] for r in records])
    r2s = np.array([r["r2"] for r in records])
    gaps = np.array([abs(r["d"] - r["d_alt"]) for r in records])
    print("Depth residuals across all bimodal candidates (metres):")
    print(f"  |z_pred - d|     median={np.median(r1s):.4f}  mean={np.mean(r1s):.4f}  p95={np.percentile(r1s,95):.4f}")
    print(f"  |z_pred - d_alt| median={np.median(r2s):.4f}  mean={np.mean(r2s):.4f}  p95={np.percentile(r2s,95):.4f}")
    print(f"  |d - d_alt| (hypothesis spread):")
    print(f"    median={np.median(gaps):.4f} m  mean={np.mean(gaps):.4f} m  p95={np.percentile(gaps,95):.4f} m")

    if n2 > 0:
        mode2_recs = [r for r in records if r["mode"] == 2]
        improvement = np.array([r["r1"] - r["r2"] for r in mode2_recs])
        print(f"\nFor the {n2} mode-2 selections:")
        print(f"  Residual improvement |r1|-|r2| (m):")
        print(f"    median={np.median(improvement):.4f}  mean={np.mean(improvement):.4f}  p95={np.percentile(improvement,95):.4f}")

        z_preds = np.array([r["z_pred"] for r in mode2_recs])
        ds = np.array([r["d"] for r in mode2_recs])
        d_alts = np.array([r["d_alt"] for r in mode2_recs])
        print(f"  z_pred / d / d_alt (metres):")
        print(f"    median z_pred={np.median(z_preds):.4f}  d={np.median(ds):.4f}  d_alt={np.median(d_alts):.4f}")

    # ------------------------------------------------------------------ #
    # Counterfactual mesh comparison (optional)                           #
    # ------------------------------------------------------------------ #
    if args.gt_ply is None or args.gt_traj is None:
        if args.gt_ply is not None:
            print("\n[mesh] --gt_traj required for Sim3 alignment; skipping mesh comparison.")
        print("\nDone.")
        return

    print(f"\nBuilding GT surface KD-tree from: {args.gt_ply}")
    tree, wSr = _build_gt_tree(args.gt_ply, args.gt_traj, data)

    # Unique mode-2 track indices (one 3D point per track, multiple measurements may vote for mode 2).
    mode2_track_indices = sorted({r["track_idx"] for r in records if r["mode"] == 2})
    print(f"\n{len(mode2_track_indices)} unique landmarks switched to mode 2.")

    # Bimodal positions for those tracks.
    bimodal_pts = np.array([np.array(data.get_track(j).point3()) for j in mode2_track_indices])
    d_bimodal = _dist_to_gt(bimodal_pts, tree, wSr)

    if args.unimodal_ba_dir is not None:
        # Per-landmark counterfactual: bimodal position vs unimodal position.
        # Track index j is the same landmark in both runs (identical frontend + data association).
        print(f"Loading unimodal BA output: {args.unimodal_ba_dir}")
        uni_data = GtsfmData.read_colmap(args.unimodal_ba_dir)
        n_uni = uni_data.number_tracks()
        print(f"  {n_uni} tracks")

        valid_indices = [j for j in mode2_track_indices if j < n_uni]
        skipped = len(mode2_track_indices) - len(valid_indices)
        if skipped:
            print(f"  WARNING: {skipped} mode-2 track indices exceed unimodal track count; skipping them.")

        # Verify track correspondence: if BA filtering removed different tracks in each
        # run, index j in bimodal and index j in unimodal are different landmarks.
        # Check by comparing the first measurement (camera_idx, pixel) of each track.
        n_check = min(len(valid_indices), 50)
        n_mismatch = 0
        for j in valid_indices[:n_check]:
            bi_cam, bi_uv = data.get_track(j).measurement(0)
            uni_cam, uni_uv = uni_data.get_track(j).measurement(0)
            if bi_cam != uni_cam or not np.allclose(bi_uv, uni_uv, atol=0.5):
                n_mismatch += 1
        print(f"\n  Track correspondence check ({n_check} sampled):")
        print(f"    Mismatched by first measurement: {n_mismatch}/{n_check}")
        if n_mismatch > 0:
            print(f"    WARNING: index mismatch detected — counterfactual comparison is invalid.")
            print(f"    The {n_mismatch}/{n_check} mismatches mean BA filtering removed different")
            print(f"    tracks in each run, shifting subsequent indices. Re-run with pixel-matched")
            print(f"    track correspondence to get valid results.")
        else:
            print(f"    OK — all sampled indices map to the same landmark in both runs.")

        if valid_indices:
            uni_pts = np.array([np.array(uni_data.get_track(j).point3()) for j in valid_indices])
            d_uni = _dist_to_gt(uni_pts, tree, wSr)
            d_bim_valid = _dist_to_gt(
                np.array([np.array(data.get_track(j).point3()) for j in valid_indices]),
                tree, wSr,
            )

            delta = d_uni - d_bim_valid  # positive = bimodal moved the point closer to GT
            n_total = len(valid_indices)
            n_improved = int((delta > 0).sum())

            print(f"\n{'='*60}")
            print(f"COUNTERFACTUAL: mode-2 landmarks, bimodal vs unimodal")
            print(f"{'='*60}")
            print(f"  Landmarks compared          : {n_total}")
            print(f"  Bimodal closer to GT        : {n_improved} / {n_total}  ({100*n_improved/n_total:.1f}%)")
            print(f"  Median improvement (cm)     : {np.median(delta)*100:+.2f}")
            print(f"  Mean   improvement (cm)     : {np.mean(delta)*100:+.2f}")
            print(f"  p25 / p75 improvement (cm)  : {np.percentile(delta,25)*100:+.2f} / {np.percentile(delta,75)*100:+.2f}")
            print(f"\n  dist-to-GT (cm):")
            print(f"    bimodal  median={np.median(d_bim_valid)*100:.2f}  mean={np.mean(d_bim_valid)*100:.2f}  p95={np.percentile(d_bim_valid,95)*100:.2f}")
            print(f"    unimodal median={np.median(d_uni)*100:.2f}  mean={np.mean(d_uni)*100:.2f}  p95={np.percentile(d_uni,95)*100:.2f}")

            # ---- Gap-stratified breakdown --------------------------------- #
            # For each mode-2 track, take the max |d - d_alt| across its
            # measurements (largest observed discontinuity for that landmark).
            track_max_gap: dict[int, float] = {}
            for r in records:
                if r["mode"] == 2:
                    j = r["track_idx"]
                    track_max_gap[j] = max(track_max_gap.get(j, 0.0), abs(r["d"] - r["d_alt"]))

            valid_gaps = np.array([track_max_gap.get(j, 0.0) for j in valid_indices])
            q25, q50, q75 = np.percentile(valid_gaps, [25, 50, 75])

            print(f"\n  Gap-stratified (|d - d_alt|):")
            print(f"  {'Gap range':<22}  {'n':>4}  {'closer':>9}  {'median Δ':>10}  {'mean Δ':>8}")
            bins = [
                (0.0,  q25,  f"Q1  ≤{q25*100:.1f} cm"),
                (q25,  q50,  f"Q2  {q25*100:.1f}–{q50*100:.1f} cm"),
                (q50,  q75,  f"Q3  {q50*100:.1f}–{q75*100:.1f} cm"),
                (q75,  np.inf, f"Q4  >{q75*100:.1f} cm"),
            ]
            for lo, hi, label in bins:
                mask = (valid_gaps >= lo) & (valid_gaps < hi)
                if not mask.any():
                    continue
                dq = delta[mask]
                nq = int(mask.sum())
                ni = int((dq > 0).sum())
                print(f"  {label:<22}  {nq:>4}  {ni:>4}/{nq} ({100*ni/nq:2.0f}%)  "
                      f"{np.median(dq)*100:>+8.2f} cm  {np.mean(dq)*100:>+6.2f} cm")
    else:
        # No unimodal run — just report mode-2 landmark distances.
        print(f"\nMode-2 landmark distances to GT (cm) [mode-1 vs mode-2 landmarks are different points — provide --unimodal_ba_dir for a fair comparison]:")
        mode1_track_indices = sorted({r["track_idx"] for r in records if r["mode"] == 1})
        bimodal_mode1_pts = np.array([np.array(data.get_track(j).point3()) for j in mode1_track_indices])
        d_mode1 = _dist_to_gt(bimodal_mode1_pts, tree, wSr)
        print(f"  mode-1 landmarks ({len(mode1_track_indices)}): median={np.median(d_mode1)*100:.2f}  mean={np.mean(d_mode1)*100:.2f}")
        print(f"  mode-2 landmarks ({len(mode2_track_indices)}): median={np.median(d_bimodal)*100:.2f}  mean={np.mean(d_bimodal)*100:.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
