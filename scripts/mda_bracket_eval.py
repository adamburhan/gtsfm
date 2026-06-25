"""Across all genuinely-bimodal MDA pixels, do the experts bracket the GT surface?

MDA-only (no patch, no recon keypoints). For every pixel where MDA commits to multiple
surfaces (max-min expert gap >= gap_thresh), align the experts to a per-image depth scale,
project each expert to 3D via the recon camera, and measure distance to the GT scan. Reports
over the whole population:
  bracket_rate   -- best-of-K expert distance < tau   (is the truth one of the modes)
  gt_is_near     -- nearest expert closer than farthest (is GT the near surface)
  near/far/best-of-K median GT distance.

--align_to controls what the per-image expert scale+shift is fit against, which is the whole
point of this diagnostic:
  vggt : fit experts -> in-recon VGGT depth (depth.npz). The pipeline default; its quality is
         confounded by VGGT depth error.
  gt   : fit experts -> GT depth rendered by z-buffering the scan into each recon camera
         (recon frame, so still recon scale). Removes the VGGT confound -- the upper bound on
         what the experts can do given perfect scale (the alignment MDA uses at inference).

Run both and compare bracket_rate to isolate how much VGGT alignment is costing the modes.

Run in the gtsfm env on a single-cluster scene:
  python scripts/mda_bracket_eval.py --align_to gt \
    --mda_dir $SCRATCH/mda_mixture/kicker_mda \
    --sfm_output $SCENE/bimodal/results/vggt --depth_npz $SCENE/bimodal/results/depth.npz \
    --gt_ply $SCRATCH/datasets/eth3d/kicker/kicker_gt.ply \
    --align_mode eth3d --align_ref $SCRATCH/datasets/eth3d/kicker/dslr_calibration_undistorted
"""

import argparse
import glob
import os

import gtsam
import numpy as np
from scipy.spatial import cKDTree

import gtsfm.utils.io as io_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.evaluation.eval_geometry import (
    _gt_poses_from_colmap,
    _load_depth_npz,
    load_gt_points,
    sim3_align,
)


def fit_scale_shift(decoded, ref, dmin, dmax):
    """Per-image (scale, shift) mapping MDA decoded depth -> the reference depth grid."""
    m = np.isfinite(decoded) & np.isfinite(ref) & (ref > dmin) & (ref < dmax)
    if int(m.sum()) < 100:
        return None
    a = np.stack([decoded[m], np.ones(int(m.sum()))], axis=1)
    (s, t), *_ = np.linalg.lstsq(a, ref[m], rcond=None)
    return float(s), float(t)


def render_gt_depth(pts_recon, cam, H, W):
    """Z-buffer the (recon-frame) GT scan into a recon camera -> (H, W) depth map (recon scale).

    Pinhole projection with the Cal3Bundler focal/principal point; ETH3D is undistorted so the
    radial terms are ~0. Nearest surface wins per pixel; unseen pixels are NaN.
    """
    pose = cam.pose()
    Xc = (pts_recon - pose.translation()) @ pose.rotation().matrix()  # world->cam (R_cw = Rwc.T)
    z = Xc[:, 2]
    cal = cam.calibration()
    front = z > 1e-6
    u = (cal.fx() * Xc[:, 0] / z + cal.px()).round().astype(int)
    v = (cal.fx() * Xc[:, 1] / z + cal.py()).round().astype(int)
    ok = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    buf = np.full((H, W), np.inf)
    np.minimum.at(buf, (v[ok], u[ok]), z[ok])
    buf[~np.isfinite(buf)] = np.nan
    return buf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mda_dir", required=True)
    ap.add_argument("--sfm_output", required=True)
    ap.add_argument("--depth_npz", required=True, help="recon depth grid (resolution always; values if --align_to vggt).")
    ap.add_argument("--gt_ply", required=True)
    ap.add_argument("--align_to", choices=["gt", "vggt"], default="gt", help="reference for the per-image expert scale+shift.")
    ap.add_argument("--align_mode", default="eth3d")
    ap.add_argument("--align_ref", required=True)
    ap.add_argument("--gap_thresh", type=float, default=0.10)
    ap.add_argument("--tau", type=float, default=0.05, help="absolute bracket threshold (m).")
    ap.add_argument("--tau_rel", type=float, default=0.05, help="scale-fair bracket: GT dist / world depth.")
    ap.add_argument("--depth_min", type=float, default=0.1)
    ap.add_argument("--depth_max", type=float, default=1e9)
    ap.add_argument("--sample_per_view", type=int, default=400, help="cap bimodal pixels scored per view.")
    args = ap.parse_args()

    wTi_list, img_fnames, *_ = io_utils.read_scene_data_from_colmap_format(args.sfm_output)
    wSr, rms_m = sim3_align(wTi_list, _gt_poses_from_colmap(args.align_ref, img_fnames))
    print(f"recon->world Sim(3): scale={wSr.scale():.6f}, camera-center RMS={rms_m:.4f} m")
    to_world = lambda p: np.array(wSr.transformFrom(np.asarray(p, dtype=float)))

    gt_world = load_gt_points(args.gt_ply)
    gt_tree = cKDTree(gt_world)
    data = GtsfmData.read_colmap(args.sfm_output)
    depth_arrays = _load_depth_npz(args.depth_npz, data)
    # GT scan in recon frame (world->recon): inverse of recon->world Sim(3).
    gt_recon = ((gt_world - wSr.translation()) @ wSr.rotation().matrix()) / wSr.scale()

    files = sorted(glob.glob(os.path.join(args.mda_dir, "[0-9]" * 6 + ".npz")))
    coords_path = os.path.join(args.mda_dir, "original_coords.npy")
    coords = np.load(coords_path) if os.path.exists(coords_path) else None

    d_near, d_far, d_best, r_best, n_bimodal_total, n_skip = [], [], [], [], 0, 0
    rng = np.random.default_rng(0)
    for i, f in enumerate(files):
        cam = data.get_camera(i)
        recon_depth = depth_arrays.get(i)
        if cam is None or recon_depth is None:
            continue
        z = np.load(f)
        means = z["means"].astype(np.float64)                 # (K, h, w)
        decoded = z["decoded"].astype(np.float64)             # (h, w)
        K, h, w = means.shape
        ct = int(round(coords[i, 1])) if coords is not None else 0
        H, W = recon_depth.shape

        if args.align_to == "vggt":
            ref = recon_depth[ct : ct + h, :w]
        else:
            ref = render_gt_depth(gt_recon, cam, H, W)[ct : ct + h, :w]
        st = fit_scale_shift(decoded, ref, args.depth_min, args.depth_max)
        if st is None:
            n_skip += 1
            continue
        s, t = st
        means = means * s + t                                 # recon scale

        valid = (means > args.depth_min) & (means < args.depth_max)  # (K, h, w)
        nvalid = valid.sum(axis=0)
        mu_hi = np.where(valid, means, -np.inf).max(axis=0)
        mu_lo = np.where(valid, means, np.inf).min(axis=0)
        bimodal = (nvalid >= 2) & ((mu_hi - mu_lo) >= args.gap_thresh)
        ys, xs = np.where(bimodal)
        n_bimodal_total += len(ys)
        if len(ys) == 0:
            continue
        if len(ys) > args.sample_per_view:
            sub = rng.choice(len(ys), args.sample_per_view, replace=False)
            ys, xs = ys[sub], xs[sub]

        cc = to_world(np.asarray(cam.pose().translation(), dtype=float))  # world cam center
        for y, x in zip(ys, xs):
            uv = gtsam.Point2(float(x), float(y + ct))

            def mode_dist(depth):  # (GT distance m, relative = GT dist / world depth)
                pw = to_world(cam.backproject(uv, float(depth)))
                dist = float(gt_tree.query(pw)[0])
                return dist, dist / (np.linalg.norm(pw - cc) + 1e-9)

            modes = means[valid[:, y, x], y, x]
            dn, _ = mode_dist(modes.min())
            df, _ = mode_dist(modes.max())
            dk = [mode_dist(m) for m in modes]
            d_near.append(dn)
            d_far.append(df)
            d_best.append(min(d for d, _ in dk))
            r_best.append(min(r for _, r in dk))

    d_near, d_far = np.array(d_near), np.array(d_far)
    d_best, r_best = np.array(d_best), np.array(r_best)
    if d_near.size == 0:
        print(f"No genuinely-bimodal pixels found (views skipped for too-few align pixels: {n_skip}).")
        return
    gt_near = d_near < d_far
    print(f"\n=== MDA bimodal bracket test (align_to={args.align_to}, MDA-only) ===")
    print(f"genuinely-bimodal pixels: {n_bimodal_total}  (scored {d_near.size}; views skipped: {n_skip})")
    print(f"bracket_rate  best-of-K  GT within {args.tau * 100:.0f}cm:            {(d_best < args.tau).mean():.3f}")
    print(f"bracket_rate  best-of-K  GT within {args.tau_rel * 100:.0f}% depth (scale-fair): {(r_best < args.tau_rel).mean():.3f}")
    print(f"gt_is_near (GT closer to nearest expert):          {gt_near.mean():.3f}")
    print(f"near-mode GT dist  median {np.median(d_near):.3f} m  p90 {np.percentile(d_near, 90):.3f} m")
    print(f"far-mode  GT dist  median {np.median(d_far):.3f} m  p90 {np.percentile(d_far, 90):.3f} m")
    print(f"best-of-K GT dist  median {np.median(d_best):.3f} m  p90 {np.percentile(d_best, 90):.3f} m  "
          f"| rel median {np.median(r_best) * 100:.1f}%")


if __name__ == "__main__":
    main()
