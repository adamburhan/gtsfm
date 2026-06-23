"""Stratify mode-selection metrics by keypoint pixel-distance-to-edge / empirical drift.

Tests whether the null bimodal result is the mechanism failing or a contaminated ambiguous
population. patch_radius flags a keypoint ambiguous if a 2nd surface lies anywhere in the
patch, but drift (~reprojection error) can only flip keypoints within ~1-2 px of the edge.
This measures, per ambiguous measurement, the keypoint's distance to the depth edge in units
of the median reprojection error, and reports mode-selection metrics per distance bin.

The gt_margin column = |dist(d) - dist(d_alt)| to the GT surface: how decisive the gt_mode
label is. Where it falls below the alignment floor (~RMS), gt_mode is a coin flip and the
mode-correctness numbers in that bin are unreliable -- read them with that caveat.

Usage:
    python scripts/diag_edge_stratify.py <sfm_output> --depth_npz <depth.npz> \
        --align_mode eth3d --align_ref <colmap GT dir> --gt_ply <gt.ply> \
        [--gap_thresh 0.10 --patch_radius 5 --min_valid 10]
"""

import argparse
import sys
from pathlib import Path

import gtsam
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gtsfm.common.depth_provider import DepthProvider
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.evaluation.eval_geometry import build_to_world, load_gt_points, _load_depth_npz


def patch_split(dmap, row, col, r, dmin, dmax):
    """(split log-depth, center_is_near, (r0,c0)) for the largest-gap split, or None."""
    h, w = dmap.shape[:2]
    r0, c0 = max(0, row - r), max(0, col - r)
    patch = dmap[r0 : min(h, row + r + 1), c0 : min(w, col + r + 1)]
    valid = patch[np.isfinite(patch) & (patch >= dmin) & (patch <= dmax)]
    if valid.size < 2:
        return None
    logs = np.sort(np.log(valid))
    gaps = np.diff(logs)
    if gaps.size == 0:
        return None
    k = int(np.argmax(gaps))
    split = 0.5 * (logs[k] + logs[k + 1])
    return split, np.log(float(dmap[row, col])) <= split, (r0, c0)


def dist_to_edge(dmap, row, col, r, split, center_is_near, dmin, dmax):
    """Pixel distance from the keypoint to the nearest pixel of the *other* depth cluster."""
    h, w = dmap.shape[:2]
    r0, c0 = max(0, row - r), max(0, col - r)
    patch = dmap[r0 : min(h, row + r + 1), c0 : min(w, col + r + 1)]
    valid = np.isfinite(patch) & (patch >= dmin) & (patch <= dmax)
    with np.errstate(invalid="ignore"):
        lp = np.log(patch)
    other = valid & ((lp > split) if center_is_near else (lp <= split))
    if not other.any():
        return np.inf
    ys, xs = np.where(other)
    return float(np.min(np.hypot(ys - (row - r0), xs - (col - c0))))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sfm_output")
    p.add_argument("--depth_npz", required=True)
    p.add_argument("--align_mode", default="eth3d")
    p.add_argument("--align_ref", required=True)
    p.add_argument("--gt_ply", required=True)
    p.add_argument("--gap_thresh", type=float, default=0.10)
    p.add_argument("--patch_radius", type=int, default=5)
    p.add_argument("--min_valid", type=int, default=10)
    p.add_argument("--depth_min", type=float, default=0.0)
    p.add_argument("--depth_max", type=float, default=1e9)
    args = p.parse_args()

    data = GtsfmData.read_colmap(args.sfm_output)
    arrays = _load_depth_npz(args.depth_npz, data)
    provider = DepthProvider(
        depth_arrays=arrays, depth_min=args.depth_min, depth_max=args.depth_max, compute_hypotheses=True,
        patch_radius=args.patch_radius, gap_thresh=args.gap_thresh, ambiguity_thresh=0.0, min_valid=args.min_valid,
    )

    n = max(data.get_valid_camera_indices()) + 1
    wTi_list = [None] * n
    img_fnames = [""] * n
    for i in data.get_valid_camera_indices():
        wTi_list[i] = data.get_camera(i).pose()
        info = data.get_image_info(i)
        img_fnames[i] = info.name if info else ""
    to_world = build_to_world(args.align_mode, args.align_ref, wTi_list, img_fnames)
    gt_tree = cKDTree(load_gt_points(args.gt_ply))

    # Empirical drift scale = median reprojection error (px).
    reproj = []
    for j in range(data.number_tracks()):
        track = data.get_track(j)
        X = track.point3()
        for m in range(track.numberMeasurements()):
            i, uv = track.measurement(m)
            cam = data.get_camera(i)
            try:
                proj = cam.project(X)
                reproj.append(float(np.hypot(proj[0] - uv[0], proj[1] - uv[1])))
            except Exception:
                pass
    drift = float(np.median(reproj))
    print(f"drift scale (median reprojection error) = {drift:.3f} px\n")

    recs = []
    for j in range(data.number_tracks()):
        track = data.get_track(j)
        Xw = np.array(track.point3())
        for m in range(track.numberMeasurements()):
            i, uv = track.measurement(m)
            cam = data.get_camera(i)
            if cam is None:
                continue
            s = provider.get_depth(i, float(uv[0]), float(uv[1]))
            if s is None or not s.ambiguous or s.depth_alt is None:
                continue
            dmap = arrays[i]
            h, w = dmap.shape[:2]
            col, row = int(np.clip(round(uv[0]), 0, w - 1)), int(np.clip(round(uv[1]), 0, h - 1))
            ps = patch_split(dmap, row, col, args.patch_radius, args.depth_min, args.depth_max)
            if ps is None:
                continue
            split, center_is_near, _ = ps
            de = dist_to_edge(dmap, row, col, args.patch_radius, split, center_is_near, args.depth_min, args.depth_max)

            z = float(cam.pose().transformTo(Xw)[2])
            opt_mode = 2 if abs(z - s.depth_alt) < abs(z - s.depth) else 1
            uv2 = gtsam.Point2(float(uv[0]), float(uv[1]))
            dd = float(gt_tree.query(to_world(cam.backproject(uv2, s.depth)))[0])
            da = float(gt_tree.query(to_world(cam.backproject(uv2, s.depth_alt)))[0])
            gt_mode = 1 if dd < da else 2
            recs.append({
                "edge_sigma": de / drift if drift > 0 else np.inf,
                "opt_mode": opt_mode, "gt_mode": gt_mode,
                "selection_cost": (dd if opt_mode == 1 else da) - min(dd, da),
                # |dd - da|: how decisive the gt_mode label is. If below the alignment floor
                # (~RMS), gt_mode is a coin flip and mode-correctness here is unreliable.
                "gt_margin": abs(dd - da),
            })

    df = pd.DataFrame(recs)
    if df.empty:
        raise SystemExit("No ambiguous measurements.")
    bins = [0, 1, 2, 3, np.inf]
    labels = ["<1σ (near edge)", "1-2σ", "2-3σ", ">3σ (deep)"]
    df["bin"] = pd.cut(df["edge_sigma"], bins=bins, labels=labels, right=False)

    rows = []
    for label in labels + ["ALL"]:
        g = df if label == "ALL" else df[df["bin"] == label]
        if len(g) == 0:
            continue
        correct = g["opt_mode"] == g["gt_mode"]
        primary = g["gt_mode"] == 1
        m2 = g["opt_mode"] == 2
        rows.append({
            "bin": label, "n": len(g), "frac": round(len(g) / len(df), 3),
            "mode_correct": round(correct.mean(), 3),
            "primary_correct": round(primary.mean(), 3),
            "bimodal_over_primary": round(correct.mean() - primary.mean(), 4),
            "mode2_selected": round(m2.mean(), 3),
            "mode2_correct": round(correct[m2].mean(), 3) if m2.any() else float("nan"),
            "sel_cost_mean_m": round(g["selection_cost"].mean(), 4),
            "gt_margin_median_m": round(g["gt_margin"].median(), 4),
        })
    print(pd.DataFrame(rows).to_markdown(index=False))


if __name__ == "__main__":
    main()
