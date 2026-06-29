"""Controlled toy experiment: does the max-mixture depth factor help under genuine depth ambiguity?

We plant a scene whose monocular depth is *intrinsically* bimodal and check that the
mixture depth factor lets multi-view BA recover the correct surface — matching an
oracle that is told the right mode, and beating a naive unimodal factor that commits
to the (sometimes wrong) monocular argmax. Because the ambiguity is planted, the GT is
known exactly and the claim "the factor is nothing but helpful" becomes a falsifiable
assertion rather than a vibe.

Scene
-----
Two fronto-parallel planes at depths Z_near, Z_far. Cameras look down +Z with identity
rotation and a small lateral baseline, so camera-frame planar depth equals world Z
exactly (no baseline correction) while the baseline still gives parallax for
triangulation. Each point lives on one plane; its monocular depth "prediction" is the
pair {Z_near, Z_far} (mono cannot tell which plane a pixel belongs to). Only multi-view
reprojection disambiguates.

Init (faithful to the real pipeline)
------------------------------------
Points are initialized by linear (DLT) triangulation of the noisy 2D tracks — exactly as
BA is seeded from the SfM reconstruction. With a *small* baseline this fixes the coarse
mode (the planes are far apart relative to the along-ray error) but leaves depth
imprecise; the depth factor's job is to sharpen it. This also puts the mixture in the
reprojection-preferred basin from the start, so its per-iteration mode selection is not a
coin flip (a midpoint init + strong depth factor would let the mixture lock onto the
wrong mode — a real failure mode worth probing separately by sweeping baseline/sigma).

Two point populations
---------------------
  on-plane  : on Z_near or Z_far (one mode is correct) — tests mode selection.
  no-mode   : strictly between the planes (NEITHER mode is within reach) — tests the null hypothesis.

Five arms (same scene, same triangulated init, poses pinned by strong priors)
----------------------------------------------------------------------------
  none            : reprojection only (baseline).
  unimodal-oracle : depth factor at the TRUE mode (upper bound; depth helps if you know the mode).
  unimodal-naive  : depth factor at the monocular argmax, wrong on a `p_wrong` fraction (commitment risk).
  mixture         : max-mixture over {Z_near, Z_far}; re-selects the mode each iteration.
  mixture+null    : same, plus a null hypothesis (`null_nsigma`) that opts out when even the best
                    mode is too far from the geometry, falling back to reprojection only.

Headlines:
  - On-plane: mixture ≈ oracle ≫ naive, both ≫ none (multi-view resolves the ambiguity).
  - No-mode:  plain mixture is DRAGGED onto a wrong plane; the null opts out and recovers
              reprojection-only accuracy. Overall, mixture+null ≪ mixture.
  - The null is a *safety trade*, not free: at init a far-but-correct mode and a no-good mode look
    identical, so the null also declines to recover heavy-tail-init on-plane points (small cost,
    dominated by avoiding the no-mode disaster). Sweep `null_nsigma` / `baseline` to see it.

Authors: Adam Burhan
"""

import argparse
import sys
from pathlib import Path

import gtsam
import numpy as np
from gtsam import Cal3_S2, NonlinearFactorGraph, PinholeCameraCal3_S2, Point3, Pose3, Rot3, Values
from gtsam.symbol_shorthand import P, X

from gtsfm.bundle.bundle_adjustment import make_depth_factor, make_mixture_depth_factor

ARMS = ["none", "unimodal-oracle", "unimodal-naive", "mixture", "mixture+null"]


def _proj_matrix(K, x_cam):
    """3x4 world->image projection matrix for an identity-rotation camera at (x_cam, 0, 0)."""
    Kmat = K.K()  # 3x3 intrinsics
    Rt = np.array([[1.0, 0.0, 0.0, -x_cam], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    return Kmat @ Rt


def _triangulate_dlt(proj_mats, uvs):
    """Linear DLT triangulation of one world point from >=2 views."""
    rows = []
    for Pm, (u, v) in zip(proj_mats, uvs):
        rows.append(u * Pm[2] - Pm[0])
        rows.append(v * Pm[2] - Pm[1])
    _, _, vh = np.linalg.svd(np.asarray(rows))
    Xh = vh[-1]
    return Xh[:3] / Xh[3]


def build_scene(args):
    """Plant the scene: on-plane (ambiguous) points + off-plane (no-good-candidate) points.

    Two populations share the same monocular hypothesis set {z_near, z_far}:
      - on-plane points lie on one of the two planes (one mode is correct) -> tests mode selection.
      - no-mode points lie strictly *between* the planes (NEITHER mode is within reach) -> tests the
        null hypothesis: a good factor should opt out and fall back to reprojection there.
    """
    rng = np.random.default_rng(args.seed)
    z_near, z_far = args.z_near, args.z_far
    cx, cy = args.width / 2.0, args.height / 2.0
    K = Cal3_S2(args.focal, args.focal, 0.0, cx, cy)

    # Cameras: identity rotation (look +Z), small lateral baseline along x, at world Z=0.
    cam_x = np.linspace(-args.baseline / 2.0, args.baseline / 2.0, args.num_cams)
    poses = [Pose3(Rot3(), Point3(float(x), 0.0, 0.0)) for x in cam_x]

    # Population split. No-mode points sit in the middle band, >= mode_margin from both planes,
    # so both monocular hypotheses are grossly wrong (many sigmas off) for them.
    no_mode = rng.random(args.num_points) < args.frac_no_mode
    on_plane = ~no_mode
    on_near = rng.random(args.num_points) < 0.5  # only meaningful for on-plane points
    z_gt = np.where(on_near, z_near, z_far)
    z_gt = np.where(no_mode, rng.uniform(z_near + args.mode_margin, z_far - args.mode_margin, args.num_points), z_gt)

    r = 0.35 * z_near  # keeps |u-cx| < focal*r/z_near within the image half-width on both planes
    xy = rng.uniform(-r, r, size=(args.num_points, 2))
    pts_gt = np.column_stack([xy, z_gt])  # (N,3) world points

    # Per-(cam, point) 2D observations with pixel noise.
    obs = {}
    for i, pose in enumerate(poses):
        cam = PinholeCameraCal3_S2(pose, K)
        for j in range(args.num_points):
            uv = cam.project(Point3(*pts_gt[j]))
            uv = uv + rng.normal(0.0, args.pixel_noise, size=2)
            obs[(i, j)] = uv

    # Init by multi-view DLT triangulation (as BA is seeded from the SfM reconstruction):
    # at small baseline this fixes the coarse depth but leaves it imprecise.
    proj_mats = [_proj_matrix(K, float(x)) for x in cam_x]
    pts_init = np.array([
        _triangulate_dlt(proj_mats, [obs[(i, j)] for i in range(args.num_cams)])
        for j in range(args.num_points)
    ])

    # Monocular argmax for the naive arm: the true on-plane mode, flipped to the decoy on a p_wrong
    # fraction; for no-mode points mono still commits to the nearest (wrong) plane.
    decoy = np.where(on_near, z_far, z_near)
    flip = rng.random(args.num_points) < args.p_wrong
    naive_depth = np.where(flip, decoy, z_gt)
    nearest_plane = np.where(np.abs(z_gt - z_near) < np.abs(z_gt - z_far), z_near, z_far)
    naive_depth = np.where(no_mode, nearest_plane, naive_depth)

    return dict(K=K, poses=poses, pts_gt=pts_gt, pts_init=pts_init, obs=obs, z_gt=z_gt,
                on_near=on_near, on_plane=on_plane, no_mode=no_mode, naive_depth=naive_depth,
                n_flipped=int((flip & on_plane).sum()), n_no_mode=int(no_mode.sum()))


def run_arm(arm, scene, args):
    """Build the factor graph for one arm, optimize, and return recovered 3D points."""
    K, poses, obs = scene["K"], scene["poses"], scene["obs"]
    z_near, z_far = args.z_near, args.z_far
    reproj_noise = gtsam.noiseModel.Isotropic.Sigma(2, args.pixel_noise if args.pixel_noise > 0 else 1.0)
    pose_prior_noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-4)  # pins poses (gauge + fixed extrinsics)
    depth_noise = gtsam.noiseModel.Isotropic.Sigma(1, args.depth_sigma)
    unit_noise = gtsam.noiseModel.Isotropic.Sigma(1, 1.0)  # mixture whitens by its own per-mode sigma

    graph = NonlinearFactorGraph()
    values = Values()
    for i, pose in enumerate(poses):
        values.insert(X(i), pose)
        graph.push_back(gtsam.PriorFactorPose3(X(i), pose, pose_prior_noise))
    for j in range(args.num_points):
        values.insert(P(j), Point3(*scene["pts_init"][j]))

    for (i, j), uv in obs.items():
        graph.push_back(gtsam.GenericProjectionFactorCal3_S2(uv, reproj_noise, X(i), P(j), K))
        if arm == "none":
            continue
        if arm == "unimodal-oracle":
            graph.push_back(make_depth_factor(X(i), P(j), float(scene["z_gt"][j]), depth_noise))
        elif arm == "unimodal-naive":
            graph.push_back(make_depth_factor(X(i), P(j), float(scene["naive_depth"][j]), depth_noise))
        elif arm in ("mixture", "mixture+null"):
            null_nsigma = args.null_nsigma if arm == "mixture+null" else None
            graph.push_back(make_mixture_depth_factor(
                X(i), P(j), [z_near, z_far], [args.depth_sigma, args.depth_sigma], [0.0, 0.0],
                unit_noise, null_nsigma=null_nsigma))

    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(args.max_iters)
    result = gtsam.LevenbergMarquardtOptimizer(graph, values, params).optimize()
    return np.array([result.atPoint3(P(j)) for j in range(args.num_points)])


def _rmse(z_rec, z_gt, mask=None):
    if mask is not None:
        z_rec, z_gt = z_rec[mask], z_gt[mask]
    if z_rec.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((z_rec - z_gt) ** 2)))


def metrics(pts_rec, scene, args):
    """Depth error (overall / on-plane / no-mode subsets) and on-plane mode-selection accuracy."""
    z_rec, z_gt = pts_rec[:, 2], scene["z_gt"]
    on_plane, no_mode = scene["on_plane"], scene["no_mode"]
    picked_near = np.abs(z_rec - args.z_near) < np.abs(z_rec - args.z_far)
    # Mode accuracy is only defined where a correct mode exists (on-plane points).
    mode_acc = float(np.mean(picked_near[on_plane] == scene["on_near"][on_plane])) if on_plane.any() else float("nan")
    return dict(
        depth_rmse=_rmse(z_rec, z_gt),
        depth_rmse_onplane=_rmse(z_rec, z_gt, on_plane),
        depth_rmse_nomode=_rmse(z_rec, z_gt, no_mode),
        pt_rmse=float(np.sqrt(np.mean(np.sum((pts_rec - scene["pts_gt"]) ** 2, axis=1)))),
        mode_acc=mode_acc,
    )


def plot(results, scene, args, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    colors = ["#9e9e9e", "#2ca02c", "#d62728", "#1f77b4", "#ff7f0e"]

    for ax, key, title, fmt in [
        (axes[0], "depth_rmse", "Depth RMSE — all points (m) ↓", "%.3f"),
        (axes[1], "depth_rmse_nomode", "Depth RMSE — no-good-mode subset (m) ↓", "%.3f"),
    ]:
        vals = [results[a][key] for a in ARMS]
        bars = ax.bar(range(len(ARMS)), vals, color=colors)
        ax.set_xticks(range(len(ARMS)))
        ax.set_xticklabels(ARMS, rotation=20, ha="right")
        ax.set_title(title)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, fmt % v, ha="center", va="bottom", fontsize=9)

    # Recovered vs true depth (mixture vs mixture+null): no-mode points (true depth between the
    # planes) get dragged onto a plane by plain mixture, but stay on the diagonal under the null.
    ax = axes[2]
    jit = np.random.default_rng(0).normal(0, 0.03, args.num_points)
    for arm, c, m in [("mixture", "#1f77b4", "x"), ("mixture+null", "#ff7f0e", "o")]:
        ax.scatter(scene["z_gt"] + jit, results[arm]["_z_rec"], s=14, alpha=0.5, c=c, marker=m, label=arm)
    for z in (args.z_near, args.z_far):
        ax.axhline(z, color="k", lw=0.5, ls=":", alpha=0.4)
    lim = [args.z_near - 0.5, args.z_far + 0.5]
    ax.plot(lim, lim, "k--", lw=1, alpha=0.6)
    ax.axvspan(args.z_near + args.mode_margin, args.z_far - args.mode_margin, color="orange", alpha=0.07)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("true depth (m)"); ax.set_ylabel("recovered depth (m)")
    ax.set_title("Recovered vs true (shaded = no-mode band)"); ax.legend(fontsize=8)

    fig.suptitle(
        f"Depth-ambiguity + null toy: {args.num_points} pts, {args.num_cams} cams, baseline={args.baseline} m, "
        f"sep={args.z_far - args.z_near} m, naive wrong {scene['n_flipped']}, no-mode {scene['n_no_mode']}",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = out_dir / "toy_depth_ambiguity.png"
    fig.savefig(path, dpi=130)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num_points", type=int, default=200)
    ap.add_argument("--num_cams", type=int, default=4)
    ap.add_argument("--z_near", type=float, default=2.0)
    ap.add_argument("--z_far", type=float, default=5.0)
    ap.add_argument("--baseline", type=float, default=0.1, help="lateral camera spread (m); small => weak triangulation")
    ap.add_argument("--focal", type=float, default=500.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--pixel_noise", type=float, default=0.5)
    ap.add_argument("--depth_sigma", type=float, default=0.1, help="depth-factor sigma (m)")
    ap.add_argument("--p_wrong", type=float, default=0.4, help="fraction of on-plane points the naive mono argmax gets wrong")
    ap.add_argument("--frac_no_mode", type=float, default=0.3, help="fraction of points with NO good depth mode (off both planes)")
    ap.add_argument("--mode_margin", type=float, default=1.0, help="min gap (m) from each plane for no-mode points")
    ap.add_argument("--null_nsigma", type=float, default=6.0, help="null hypothesis: opt out if best mode > this many sigmas off")
    ap.add_argument("--max_iters", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="toy_depth_results")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene = build_scene(args)
    results = {}
    for arm in ARMS:
        pts_rec = run_arm(arm, scene, args)
        m = metrics(pts_rec, scene, args)
        m["_z_rec"] = pts_rec[:, 2]
        results[arm] = m

    # Report.
    print(f"\nScene: {args.num_points} pts, {args.num_cams} cams, baseline={args.baseline} m, "
          f"planes @ {args.z_near}/{args.z_far} m | naive wrong on {scene['n_flipped']} on-plane, "
          f"{scene['n_no_mode']} no-good-mode points\n")
    print(f"{'arm':<18}{'depth_all':>12}{'depth_onplane':>15}{'depth_nomode':>14}{'mode_acc':>10}")
    print("-" * 69)
    for arm in ARMS:
        r = results[arm]
        print(f"{arm:<18}{r['depth_rmse']:>12.4f}{r['depth_rmse_onplane']:>15.4f}"
              f"{r['depth_rmse_nomode']:>14.4f}{r['mode_acc']:>10.3f}")

    path = plot(results, scene, args, out_dir)
    print(f"\nWrote {path}")

    mix, mixn, ora, naive, none = (results[a] for a in
                                   ["mixture", "mixture+null", "unimodal-oracle", "unimodal-naive", "none"])
    checks = [
        # On-plane: the mixture resolves the ambiguity, matching the oracle and beating naive.
        ("mixture mode_acc(on-plane) >= 0.98", mix["mode_acc"] >= 0.98),
        ("mixture depth(on-plane) <= 1.5x oracle", mix["depth_rmse_onplane"] <= 1.5 * ora["depth_rmse_onplane"] + 1e-9),
        ("mixture depth(on-plane) < naive", mix["depth_rmse_onplane"] < naive["depth_rmse_onplane"]),
        # No-mode: plain mixture is dragged to a wrong plane; the null opts out and recovers reprojection.
        ("plain mixture is dragged on no-mode pts (mixture > none)", mix["depth_rmse_nomode"] > none["depth_rmse_nomode"]),
        ("null beats plain mixture on no-mode pts", mixn["depth_rmse_nomode"] < mix["depth_rmse_nomode"]),
        ("null recovers reprojection on no-mode pts (<= 1.5x none)",
         mixn["depth_rmse_nomode"] <= 1.5 * none["depth_rmse_nomode"] + 1e-9),
        # The null must not damage the good case: same on-plane mode resolution as plain mixture.
        ("null keeps mode_acc(on-plane) >= 0.98", mixn["mode_acc"] >= 0.98),
        ("null never worse than mixture overall", mixn["depth_rmse"] <= mix["depth_rmse"] + 1e-9),
    ]
    print("\nAssertions:")
    ok = True
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    print(f"\n{'✅ ALL CHECKS PASSED' if ok else '❌ SOME CHECKS FAILED'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
