"""Adversarial validation tests for depth-aware bundle adjustment.

Complements tests/bundle/test_depth_factors.py (which covers the unimodal/bimodal
factors) with the pieces that back the ETH3D sweep claims:

  1. Mixture factor (`make_mixture_depth_factor`): finite-difference Jacobians for
     each active mode, whitened mode selection, cost equivalence with an
     `Isotropic.Sigma` unimodal factor, degenerate two-identical-modes case, and
     the null-hypothesis opt-out (zero residual/Jacobian, no NaN in optimization).
  2. DepthProvider conventions: (u,v) <-> (col,row) indexing, silent clipping on a
     resolution mismatch, filename-template mismatch producing zero factors.
  3. GMM hypothesis extraction: degenerate-patch guard, near->far ordering,
     sigma floor, behavior of a noise-only unimodal patch.
  4. End-to-end plumbing on a synthetic scene through the real
     BundleAdjustmentOptimizer + on-disk .npy depth (the sweep's exact config
     path: template=null, ext=.npy): self-consistent depth is a fixed point;
     auto-scale recovers a synthetic gauge scale exactly; omitting the scale
     drags the recon to metric scale (positive control); depth noise degrades
     the recon monotonically.

Authors: Adam Burhan (validation brief)
"""

import tempfile
import unittest
from pathlib import Path

import gtsam  # type: ignore
import numpy as np
from gtsam import Cal3Bundler, PinholeCameraCal3Bundler, Point3, Pose3, Rot3, Values
from gtsam.symbol_shorthand import P, X  # type: ignore

from gtsfm.bundle.bundle_adjustment import (
    BundleAdjustmentOptions,
    DepthFactorRecord,
    make_depth_factor,
    make_mixture_depth_factor,
    refit_depth_scales,
)
from gtsfm.common.depth_provider import DepthProvider
from gtsfm.common.gtsfm_data import GtsfmData

UNIT_NOISE = gtsam.noiseModel.Isotropic.Sigma(1, 1.0)


def _values(wTc: Pose3, point_w) -> Values:
    values = Values()
    values.insert(X(0), wTc)
    values.insert(P(0), Point3(point_w))
    return values


def _scene_at_z(z: float):
    """Camera + world point with camera-frame depth exactly z."""
    wTc = Pose3(Rot3.RzRyRx(0.1, -0.2, 0.3), np.array([0.5, -1.0, 2.0]))
    point_c = np.array([0.4, -0.2, z])
    point_w = wTc.transformFrom(Point3(point_c))
    return wTc, point_w, _values(wTc, point_w)


class TestMixtureDepthFactor(unittest.TestCase):
    """make_mixture_depth_factor: residual, Jacobians, mode selection, null branch."""

    def test_single_mode_matches_isotropic_unimodal(self):
        """Mixture with one mode under unit noise == unimodal under Isotropic.Sigma(1, sigma).

        Verifies total factor cost 0.5*(r/sigma)^2 agrees, so mixture and unimodal rows of the
        sweep tables are on the same cost scale.
        """
        d, sigma = 2.5, 0.17
        mixture = make_mixture_depth_factor(X(0), P(0), [d], [sigma], [0.0], UNIT_NOISE)
        unimodal = make_depth_factor(X(0), P(0), d, gtsam.noiseModel.Isotropic.Sigma(1, sigma))
        for z in [0.5, 2.5, 3.0, 7.9]:
            _, _, values = _scene_at_z(z)
            expected_cost = 0.5 * ((z - d) / sigma) ** 2
            self.assertAlmostEqual(mixture.error(values), expected_cost, places=7)
            self.assertAlmostEqual(unimodal.error(values), expected_cost, places=7)

    def test_two_identical_modes_is_noop(self):
        """Two coincident modes == the unimodal factor (treating unimodal pixels as mixtures is safe)."""
        d, sigma = 3.2, 0.2
        mixture = make_mixture_depth_factor(X(0), P(0), [d, d], [sigma, sigma], [0.0, 0.0], UNIT_NOISE)
        unimodal = make_depth_factor(X(0), P(0), d, gtsam.noiseModel.Isotropic.Sigma(1, sigma))
        for z in [2.0, 3.2, 5.5]:
            _, _, values = _scene_at_z(z)
            self.assertAlmostEqual(mixture.error(values), unimodal.error(values), places=7)

    def test_whitened_mode_selection(self):
        """Selection minimizes 0.5*(r/sigma)^2 + log(sigma), not raw |r|."""
        depths, sigmas = [2.0, 5.0], [0.1, 1.0]
        factor = make_mixture_depth_factor(X(0), P(0), depths, sigmas, [0.0, 0.0], UNIT_NOISE)
        # z=3.4: raw residuals are 1.4 (near mode) vs -1.6 (far mode) -> raw min-|r| would pick the
        # near mode, but whitened cost (98.0-2.3 vs 1.28+0.0) picks the far mode.
        _, _, values = _scene_at_z(3.4)
        r = factor.unwhitenedError(values)
        self.assertAlmostEqual(r[0], (3.4 - 5.0) / 1.0, places=6)

    def test_log_weights_are_inert(self):
        """Documents that log_weights currently do NOT affect selection ("temporarily uniform prior").

        depth_mda_near_prior and the GMM weights are plumbed through but have no effect. If this
        test starts failing, the prior was re-enabled and the sweep semantics changed.
        """
        depths, sigmas = [2.0, 5.0], [1.0, 1.0]
        # Huge prior on the near mode; selection should STILL pick the closer (far) mode at z=4.9.
        factor = make_mixture_depth_factor(X(0), P(0), depths, sigmas, [1e6, -1e6], UNIT_NOISE)
        _, _, values = _scene_at_z(4.9)
        self.assertAlmostEqual(factor.unwhitenedError(values)[0], (4.9 - 5.0), places=6)

    def test_jacobians_match_finite_differences_per_mode(self):
        """FD check of the whitened Jacobian with the mode-dependent 1/sigma_k scaling."""
        depths, sigmas = [2.0, 5.0], [0.3, 0.8]
        factor = make_mixture_depth_factor(X(0), P(0), depths, sigmas, [0.0, 0.0], UNIT_NOISE)
        for z in [2.1, 4.8]:  # activates mode 0, then mode 1
            wTc, point_w, values = _scene_at_z(z)
            A, _ = factor.linearize(values).jacobian()
            eps = 1e-6

            def residual(pose, pt):
                return factor.unwhitenedError(_values(pose, pt))[0]

            for k in range(6):
                delta = np.zeros(6)
                delta[k] = eps
                fd = (residual(wTc.retract(delta), point_w) - residual(wTc.retract(-delta), point_w)) / (2 * eps)
                self.assertAlmostEqual(A[0, k], fd, places=5, msg=f"pose col {k} at z={z}")
            for k in range(3):
                delta = np.zeros(3)
                delta[k] = eps
                fd = (residual(wTc, point_w + delta) - residual(wTc, point_w - delta)) / (2 * eps)
                self.assertAlmostEqual(A[0, 6 + k], fd, places=5, msg=f"point col {k} at z={z}")

    def test_null_hypothesis_zero_residual_and_jacobian(self):
        """Best mode > null_nsigma sigmas off -> zero cost, zero Jacobian, finite linearization."""
        factor = make_mixture_depth_factor(X(0), P(0), [2.0], [0.1], [0.0], UNIT_NOISE, null_nsigma=5.0)
        _, _, values = _scene_at_z(3.0)  # |r| = 10 sigma > 5
        self.assertEqual(factor.error(values), 0.0)
        A, b = factor.linearize(values).jacobian()
        np.testing.assert_array_equal(A, np.zeros_like(A))
        np.testing.assert_array_equal(b, np.zeros_like(b))

    def test_null_hypothesis_inactive_inside_band(self):
        """Best mode within null_nsigma -> factor behaves like the gated-off version."""
        factor = make_mixture_depth_factor(X(0), P(0), [2.0], [0.1], [0.0], UNIT_NOISE, null_nsigma=5.0)
        _, _, values = _scene_at_z(2.3)  # |r| = 3 sigma < 5
        self.assertAlmostEqual(factor.unwhitenedError(values)[0], (2.3 - 2.0) / 0.1, places=6)

    def test_null_hypothesis_does_not_break_optimization(self):
        """LM on a graph containing a null-selected factor converges without NaN and without pull."""
        wTc, point_w, values = _scene_at_z(3.0)
        graph = gtsam.NonlinearFactorGraph()
        graph.push_back(gtsam.PriorFactorPose3(X(0), wTc, gtsam.noiseModel.Isotropic.Sigma(6, 1e-3)))
        graph.push_back(gtsam.PriorFactorPoint3(P(0), Point3(point_w), gtsam.noiseModel.Isotropic.Sigma(3, 1e-2)))
        graph.push_back(make_mixture_depth_factor(X(0), P(0), [2.0], [0.1], [0.0], UNIT_NOISE, null_nsigma=5.0))
        result = gtsam.LevenbergMarquardtOptimizer(graph, values).optimize()
        moved = np.linalg.norm(result.atPoint3(P(0)) - np.asarray(point_w))
        self.assertTrue(np.isfinite(graph.error(result)))
        self.assertLess(moved, 1e-6, "null-selected depth factor should exert no pull")


class TestDepthProviderConventions(unittest.TestCase):
    """Sampling conventions and the known silent failure modes of DepthProvider."""

    def _provider(self, depth_map: np.ndarray, fname="DSC_0675.JPG", template=None, ext=".npy", **kwargs):
        tmp_dir = Path(tempfile.mkdtemp())
        np.save(tmp_dir / (Path(fname).stem + ".npy"), depth_map)
        defaults = dict(depth_min=0.1, depth_max=100.0, depth_filename_template=template, depth_ext=ext)
        defaults.update(kwargs)
        return DepthProvider(str(tmp_dir), image_fnames={0: fname}, **defaults)

    def test_uv_is_col_row(self):
        """get_depth(u, v) must read depth_map[v, u] (u = x = column, per gtsfm Keypoints)."""
        depth_map = np.full((120, 160), np.nan)
        depth_map[10, 30] = 4.25  # row 10, col 30
        provider = self._provider(depth_map)
        sample = provider.get_depth(0, u=30.0, v=10.0)
        self.assertIsNotNone(sample)
        self.assertAlmostEqual(sample.depth, 4.25)
        # The transposed lookup must miss (NaN -> None).
        self.assertIsNone(provider.get_depth(0, u=10.0, v=30.0))

    def test_resolution_mismatch_is_silent_clipping(self):
        """DOCUMENTS A HAZARD: a wrong-resolution depth map is sampled with edge clipping, no error.

        If the loader resolution and the depth-map resolution ever diverge, every out-of-bounds
        keypoint silently reads the border pixel — factors get built with garbage depths and the
        only symptom is bad results. `_depth_map` has no shape check against the image.
        """
        depth_map = np.full((60, 80), 2.0)  # half the 120x160 image resolution
        depth_map[59, 79] = 9.99
        provider = self._provider(depth_map)
        sample = provider.get_depth(0, u=150.0, v=110.0)  # valid in the image, outside the map
        self.assertIsNotNone(sample, "mismatch should have been detected, but is silently clipped")
        self.assertAlmostEqual(sample.depth, 9.99, msg="clipped to the border pixel")

    def test_wrong_template_yields_no_samples(self):
        """The sweep's .npy sources REQUIRE template=null; the dataclass default silently misses.

        With the default template 'depth{:06d}.png' and an ETH3D fname, the provider looks for
        depth000675.png, finds nothing, and every sample is None -> a 'depth' run silently
        reproduces the baseline. This is the failure the n_factors>0 canary must catch.
        """
        provider = self._provider(np.full((10, 10), 2.0), template="depth{:06d}.png", ext=".png")
        self.assertIsNone(provider.get_depth(0, u=5.0, v=5.0))

    def test_scale_division(self):
        """Raw values are divided by depth_scale on load."""
        provider = self._provider(np.full((10, 10), 5000.0), depth_scale=1000.0)
        sample = provider.get_depth(0, u=5.0, v=5.0)
        self.assertAlmostEqual(sample.depth, 5.0)


class TestGmmHypothesisExtraction(unittest.TestCase):
    """_analyze_patch_gmm: guards, ordering, sigma floor."""

    def _provider(self, depth_map: np.ndarray, **kwargs):
        tmp_dir = Path(tempfile.mkdtemp())
        np.save(tmp_dir / "img.npy", depth_map)
        defaults = dict(
            depth_min=0.1, depth_max=100.0, depth_filename_template=None, depth_ext=".npy",
            compute_hypotheses=True, hypothesis_method="gmm", patch_radius=3, min_valid=10,
        )
        defaults.update(kwargs)
        return DepthProvider(str(tmp_dir), image_fnames={0: "img.jpg"}, **defaults)

    def test_constant_patch_hits_degenerate_guard(self):
        """Exactly constant patch -> the log-range < 1e-3 guard returns a plain unimodal sample."""
        sample = self._provider(np.full((50, 50), 2.0)).get_depth(0, u=25.0, v=25.0)
        self.assertFalse(sample.is_mixture)
        self.assertFalse(sample.ambiguous)
        self.assertAlmostEqual(sample.depth, 2.0)

    def test_two_plane_patch_ordered_near_far_with_sigma_floor(self):
        """fg/bg boundary -> two modes near->far, sigmas >= floor * mean, is_mixture set."""
        depth_map = np.full((50, 50), 1.0)
        depth_map[:, 25:] = 3.0
        sample = self._provider(depth_map).get_depth(0, u=24.0, v=25.0)
        self.assertTrue(sample.is_mixture)
        self.assertEqual(len(sample.depths), 2)
        self.assertLess(sample.depths[0], sample.depths[1])
        self.assertAlmostEqual(sample.depths[0], 1.0, places=2)
        self.assertAlmostEqual(sample.depths[1], 3.0, places=2)
        for mu, sig in zip(sample.depths, sample.sigmas):
            self.assertGreaterEqual(sig, 0.05 * mu * (1 - 1e-9))

    def test_noisy_unimodal_patch_is_not_a_razor_split(self):
        """A noise-only patch must not produce two overconfident far-apart modes.

        The GMM may still split it (both weights can exceed gmm_min_weight), but the two means
        must be close (within the noise) and the sigma floor must keep each mode's sigma sane,
        so the resulting mixture factor is behaviorally unimodal.
        """
        rng = np.random.default_rng(0)
        depth_map = 2.0 * np.exp(0.01 * rng.standard_normal((50, 50)))
        sample = self._provider(depth_map).get_depth(0, u=25.0, v=25.0)
        if sample.is_mixture:
            spread = abs(np.log(sample.depths[1]) - np.log(sample.depths[0]))
            self.assertLess(spread, 0.05, "noise split produced far-apart modes")
            for mu, sig in zip(sample.depths, sample.sigmas):
                self.assertGreaterEqual(sig, 0.05 * mu * (1 - 1e-9))
        else:
            self.assertAlmostEqual(sample.depth, 2.0, delta=0.05)


def _build_synthetic_scene(n_cams=8, n_points=60, seed=0):
    """Zero-noise synthetic scene: cameras on a ring looking at a point cloud at the origin.

    Measurements are exact projections, so the ground-truth configuration is an exact global
    optimum of the reprojection-only problem — any motion under self-consistent depth factors
    indicates a plumbing bug.
    """
    rng = np.random.default_rng(seed)
    calibration = Cal3Bundler(200.0, 0.0, 0.0, 80.0, 60.0)
    img_h, img_w = 120, 160

    cameras = {}
    for i in range(n_cams):
        theta = 2.0 * np.pi * i / n_cams
        eye = np.array([4.0 * np.cos(theta), 4.0 * np.sin(theta), 0.5 * np.sin(2 * theta)])
        cameras[i] = PinholeCameraCal3Bundler.Lookat(Point3(eye), Point3(0, 0, 0), Point3(0, 0, 1), calibration)

    points = rng.uniform(-1.0, 1.0, size=(n_points, 3))
    data = GtsfmData(number_images=n_cams)
    for i, cam in cameras.items():
        data.add_camera(i, cam)

    measurements = []  # (track_j, cam_i, uv, z)
    n_tracks = 0
    for p in points:
        track = gtsam.SfmTrack(Point3(p))
        obs = []
        for i, cam in cameras.items():
            point_c = cam.pose().transformTo(Point3(p))
            if point_c[2] <= 0.1:
                continue
            uv = cam.project(Point3(p))
            if not (0.0 <= uv[0] < img_w and 0.0 <= uv[1] < img_h):
                continue
            obs.append((i, uv, float(point_c[2])))
        if len(obs) < 3:
            continue
        for i, uv, z in obs:
            track.addMeasurement(i, uv)
            measurements.append((n_tracks, i, uv, z))
        data.add_track(track)
        n_tracks += 1
    return data, measurements, (img_h, img_w)


def _write_depth_maps(data, measurements, img_hw, out_dir, fnames, depth_of_z=lambda z: z):
    """Render per-measurement depth into NaN-filled .npy maps (rounded-pixel writes).

    Returns the number of rounded-pixel collisions (must be 0 for a clean fixed-point test).
    """
    img_h, img_w = img_hw
    maps = {i: np.full((img_h, img_w), np.nan) for i in fnames}
    collisions = 0
    for _, i, uv, z in measurements:
        col = int(np.clip(round(float(uv[0])), 0, img_w - 1))
        row = int(np.clip(round(float(uv[1])), 0, img_h - 1))
        if np.isfinite(maps[i][row, col]) and not np.isclose(maps[i][row, col], depth_of_z(z)):
            collisions += 1
        maps[i][row, col] = depth_of_z(z)
    for i, fname in fnames.items():
        np.save(Path(out_dir) / (Path(fname).stem + ".npy"), maps[i])
    return collisions


def _run_depth_ba_full(data, depth_dir, fnames, depth_model="unimodal", auto_scale=False, **extra):
    """Run the real BundleAdjustmentOptimizer end-to-end with on-disk depth (sweep config path)."""
    options = BundleAdjustmentOptions(
        robust_ba_mode="NONE",
        depth_model=depth_model,
        depth_map_dir=str(depth_dir),
        depth_filename_template=None,  # sweep sets template=null + ext=.npy
        depth_ext=".npy",
        depth_min=0.1,
        depth_max=100.0,
        depth_auto_scale=auto_scale,
        **extra,
    )
    optimizer = options.to_optimizer(reproj_error_thresholds=[None], measurement_noise_sigma=1.0)
    n = data.number_images()
    optimized, filtered, _, _ = optimizer._run_ba_and_evaluate(
        data, [None] * n, {}, cameras_gt=[None] * n, image_fnames=fnames, verbose=False
    )
    return filtered, optimizer


def _run_depth_ba(data, depth_dir, fnames, depth_model="unimodal", auto_scale=False, **extra):
    filtered, optimizer = _run_depth_ba_full(data, depth_dir, fnames, depth_model, auto_scale, **extra)
    return filtered, optimizer._depth_factor_stats


def _camera_center_rms(data_a: GtsfmData, data_b: GtsfmData) -> float:
    deltas = []
    for i in data_a.get_valid_camera_indices():
        ca = data_a.get_camera(i).pose().translation()
        cb = data_b.get_camera(i).pose().translation()
        deltas.append(np.linalg.norm(np.asarray(ca) - np.asarray(cb)))
    return float(np.sqrt(np.mean(np.square(deltas))))


def _scene_scale(data: GtsfmData) -> float:
    centers = [np.asarray(data.get_camera(i).pose().translation()) for i in data.get_valid_camera_indices()]
    return float(np.median(np.linalg.norm(np.asarray(centers), axis=1)))


def _rescaled_scene(data: GtsfmData, factor: float) -> GtsfmData:
    """Globally rescale a scene (poses + points) — projections are invariant."""
    out = GtsfmData(number_images=data.number_images())
    for i in data.get_valid_camera_indices():
        cam = data.get_camera(i)
        pose = cam.pose()
        out.add_camera(i, PinholeCameraCal3Bundler(
            Pose3(pose.rotation(), np.asarray(pose.translation()) * factor), cam.calibration()))
    for j in range(data.number_tracks()):
        track = data.get_track(j)
        new_track = gtsam.SfmTrack(Point3(np.asarray(track.point3()) * factor))
        for m in range(track.numberMeasurements()):
            i, uv = track.measurement(m)
            new_track.addMeasurement(i, uv)
        out.add_track(new_track)
    return out


class TestEndToEndDepthPlumbing(unittest.TestCase):
    """Loading -> filename resolution -> sampling -> scaling -> factors -> optimization, in one shot."""

    @classmethod
    def setUpClass(cls):
        cls.data, cls.measurements, cls.img_hw = _build_synthetic_scene()
        cls.fnames = {i: f"DSC_{100 + i:04d}.JPG" for i in cls.data.get_valid_camera_indices()}
        assert cls.data.number_tracks() >= 40, "synthetic scene too sparse"
        per_cam = {i: 0 for i in cls.fnames}
        for _, i, _, _ in cls.measurements:
            per_cam[i] += 1
        assert min(per_cam.values()) >= 15, "need >= min_tracks_per_camera tracks per camera"

    def _fresh_depth_dir(self, depth_of_z=lambda z: z):
        depth_dir = Path(tempfile.mkdtemp())
        collisions = _write_depth_maps(self.data, self.measurements, self.img_hw, depth_dir, self.fnames, depth_of_z)
        self.assertEqual(collisions, 0, "rounded-pixel collision corrupts the fixed-point premise")
        return depth_dir

    def test_self_consistent_depth_is_a_fixed_point(self):
        """Depth == the recon's own z at every measurement -> BA must not move the solution."""
        result, stats = _run_depth_ba(self.data, self._fresh_depth_dir(), self.fnames)
        self.assertEqual(stats["skipped"], 0, "every measurement pixel was written; none may be skipped")
        self.assertEqual(stats["unimodal"], len(self.measurements))
        rms = _camera_center_rms(self.data, result)
        self.assertLess(rms, 1e-4, f"self-consistent depth moved the cameras by RMS {rms:.2e}")

    def test_wrong_filename_template_silently_reproduces_baseline(self):
        """With the dataclass default template, .npy maps are never found -> zero depth factors.

        This is the exact silent failure the per-run n_factors canary exists to catch.
        """
        depth_dir = self._fresh_depth_dir()
        result, stats = _run_depth_ba(
            self.data, depth_dir, self.fnames, depth_model="unimodal",
        )
        # sanity: correct config builds factors
        self.assertGreater(stats["unimodal"], 0)
        options = BundleAdjustmentOptions(
            robust_ba_mode="NONE", depth_model="unimodal", depth_map_dir=str(depth_dir),
            # default depth_filename_template="depth{:06d}.png" left in place on purpose
        )
        optimizer = options.to_optimizer(reproj_error_thresholds=[None], measurement_noise_sigma=1.0)
        n = self.data.number_images()
        optimizer._run_ba_and_evaluate(
            self.data, [None] * n, {}, cameras_gt=[None] * n, image_fnames=self.fnames, verbose=False
        )
        self.assertEqual(optimizer._depth_factor_stats["unimodal"], 0)
        self.assertEqual(optimizer._depth_factor_stats["bimodal"], 0)
        self.assertEqual(optimizer._depth_factor_stats["skipped"], len(self.measurements))

    def test_auto_scale_recovers_synthetic_gauge_exactly(self):
        """Scene shrunk 7x, depth kept metric: auto_scale must hold the recon at its own gauge."""
        s = 7.0
        shrunk = _rescaled_scene(self.data, 1.0 / s)
        result, stats = _run_depth_ba(shrunk, self._fresh_depth_dir(), self.fnames, auto_scale=True)
        self.assertEqual(stats["skipped"], 0)
        scale_ratio = _scene_scale(result) / _scene_scale(shrunk)
        self.assertAlmostEqual(scale_ratio, 1.0, delta=1e-3,
                               msg=f"auto_scale failed to neutralize the gauge (ratio {scale_ratio:.4f})")
        self.assertLess(_camera_center_rms(shrunk, result), 1e-4 / s * 10)

    def test_missing_scale_drags_recon_to_metric(self):
        """POSITIVE CONTROL: same shrunk scene with sf=1 must be pulled toward metric scale.

        Confirms the harness can detect scale mis-application: reprojection cost is invariant to
        global scaling, so metric depth factors with sf=1 should re-inflate the recon toward 7x.
        A pass here proves the fixed-point tests above have teeth.
        """
        s = 7.0
        shrunk = _rescaled_scene(self.data, 1.0 / s)
        result, _ = _run_depth_ba(shrunk, self._fresh_depth_dir(), self.fnames, auto_scale=False)
        scale_ratio = _scene_scale(result) / _scene_scale(shrunk)
        self.assertGreater(scale_ratio, 2.0, f"depth factors failed to act on scale (ratio {scale_ratio:.3f})")

    def test_depth_noise_degrades_monotonically(self):
        """Multiplicative log-normal depth noise at increasing sigma -> increasing camera drift."""
        rms = []
        for sigma_rel in [0.02, 0.1, 0.5]:
            rng = np.random.default_rng(42)
            depth_dir = Path(tempfile.mkdtemp())
            _write_depth_maps(
                self.data, self.measurements, self.img_hw, depth_dir, self.fnames,
                depth_of_z=lambda z: z * float(np.exp(sigma_rel * rng.standard_normal())),
            )
            result, stats = _run_depth_ba(self.data, depth_dir, self.fnames)
            self.assertGreater(stats["unimodal"], 0)
            gt_poses = {i: self.data.get_camera(i).pose() for i in self.data.get_valid_camera_indices()}
            aligned = result.align_via_sim3_and_transform(gt_poses)
            rms.append(_camera_center_rms(self.data, aligned))
        self.assertLess(rms[0], rms[1])
        self.assertLess(rms[1], rms[2])
        self.assertGreater(rms[2], 1e-3, "0.5-sigma depth noise should visibly degrade the recon")


class TestPerImageScaleAndRelativeSigma(unittest.TestCase):
    """Per-image profiled depth scale (sigma_a -> inf) + depth-proportional sigma.

    Convention: a_i is the CORRECTION applied to the measurement (z - a_i * d). Corrupting image
    i's depths by a factor a*_i therefore makes the refit converge to a_i = 1/a*_i; recovery is
    asserted log-symmetrically as |log(a_i * a*_i)| < 1%.
    """

    @classmethod
    def setUpClass(cls):
        cls.data, cls.measurements, cls.img_hw = _build_synthetic_scene()
        cls.fnames = {i: f"DSC_{100 + i:04d}.JPG" for i in cls.data.get_valid_camera_indices()}
        cls.a_star = {i: [0.85, 1.0, 1.25][i % 3] for i in cls.fnames}

    def _scaled_depth_dir(self, a_star=None):
        """Self-consistent depth maps with image i's depths multiplied by a*_i."""
        depth_dir = Path(tempfile.mkdtemp())
        collisions = _write_depth_maps(self.data, self.measurements, self.img_hw, depth_dir, self.fnames)
        self.assertEqual(collisions, 0)
        for i, fname in self.fnames.items():
            path = depth_dir / (Path(fname).stem + ".npy")
            np.save(path, np.load(path) * (a_star or self.a_star)[i])
        return depth_dir

    def test_per_image_scale_recovers_known_scales_and_is_fixed_point(self):
        """Known per-image corruptions a*_i in {0.85, 1, 1.25}: a_i -> 1/a*_i, solution does not move.

        Extends the self-consistent fixed-point test: after the profiled scales absorb the
        corruption, the depth factors are self-consistent again and BA must not move the solution.
        """
        result, optimizer = _run_depth_ba_full(
            self.data, self._scaled_depth_dir(), self.fnames,
            depth_per_image_scale=True, depth_pis_min_factors=10,
        )
        self.assertEqual(optimizer._depth_factor_stats["skipped"], 0)
        self.assertEqual(set(optimizer._pis_a), set(self.fnames))
        for i, a_i in optimizer._pis_a.items():
            self.assertLess(
                abs(np.log(a_i * self.a_star[i])), 0.01,
                f"image {i}: recovered a={a_i:.4f} vs injected a*={self.a_star[i]}",
            )
            self.assertFalse(optimizer._pis_last_info[i]["fallback"])
            self.assertFalse(optimizer._pis_last_info[i]["clamped"])
        rms = _camera_center_rms(self.data, result)
        self.assertLess(rms, 1e-4, f"profiled scales should restore the fixed point (RMS {rms:.2e})")

    def test_per_image_scale_off_solution_moves(self):
        """POSITIVE CONTROL: same corrupted depths without the profiled scale -> BA moves."""
        result, _ = _run_depth_ba(self.data, self._scaled_depth_dir(), self.fnames)
        rms = _camera_center_rms(self.data, result)
        self.assertGreater(rms, 1e-3, f"corrupted depths should move the solution (RMS {rms:.2e})")

    def test_per_image_scale_with_relative_sigma_fixed_point(self):
        """Both new flags on together (the _rs_pis sweep condition): fixed point still holds."""
        result, optimizer = _run_depth_ba_full(
            self.data, self._scaled_depth_dir(), self.fnames,
            depth_per_image_scale=True, depth_pis_min_factors=10,
            depth_relative_sigma=0.05, depth_sigma_floor=0.02,
        )
        for i, a_i in optimizer._pis_a.items():
            self.assertLess(abs(np.log(a_i * self.a_star[i])), 0.01)
        self.assertLess(_camera_center_rms(self.data, result), 1e-4)

    def _build_graph(self, records, sf=1.0, a=None, **options):
        opts = dict(robust_ba_mode="NONE", depth_model="unimodal")
        opts.update(options)
        optimizer = BundleAdjustmentOptions(**opts).to_optimizer()
        return optimizer._BundleAdjustmentOptimizer__build_depth_graph(records, sf, a)

    def test_relative_sigma_whitening_and_floor(self):
        """Whitened residual = raw / (rel * d); the floor engages when rel * d < floor."""
        rel, floor = 0.1, 0.02
        for d, sigma_expected in [(2.5, 0.1 * 2.5), (0.15, 0.02)]:  # rel*0.15 = 0.015 < floor
            graph = self._build_graph(
                [DepthFactorRecord(0, 0, "unimodal", (d,))],
                depth_relative_sigma=rel, depth_sigma_floor=floor,
            )
            factor = graph.at(0)
            for z in [d + 0.3, d * 1.5]:
                _, _, values = _scene_at_z(z)
                self.assertAlmostEqual(factor.error(values), 0.5 * ((z - d) / sigma_expected) ** 2, places=7)

    def test_mixture_composition_with_scale(self):
        """a_i = 1: bit-identical to the legacy mixture; a_i = 1.2: analytic mode selection on scaled modes."""
        rec = DepthFactorRecord(0, 0, "mixture", (2.0, 5.0), (0.2, 0.5), (0.0, 0.0))
        legacy = make_mixture_depth_factor(X(0), P(0), [2.0, 5.0], [0.2, 0.5], [0.0, 0.0], UNIT_NOISE)
        identity = self._build_graph([rec], a={0: 1.0}).at(0)
        scaled = self._build_graph([rec], a={0: 1.2}).at(0)
        mus, sigs = 1.2 * np.array(rec.depths), 1.2 * np.array(rec.sigmas)
        for z in [1.8, 2.4, 3.5, 4.2, 6.5]:
            _, _, values = _scene_at_z(z)
            self.assertEqual(identity.error(values), legacy.error(values), f"a=1 must be a no-op at z={z}")
            r = (z - mus) / sigs
            k = int(np.argmin(0.5 * r * r + np.log(sigs)))
            self.assertAlmostEqual(scaled.unwhitenedError(values)[0], r[k], places=7, msg=f"z={z}")
        # Ordering preserved: near mode stays near under any a > 0.
        self.assertLess(mus[0], mus[1])

    def _refit_values(self, points_by_image):
        """Values with identity poses; P(j) at (0, 0, z) so camera-frame depth == z."""
        values = Values()
        j = 0
        records = []
        for i, pairs in sorted(points_by_image.items()):
            values.insert(X(i), Pose3())
            for z, rec_args in pairs:
                values.insert(P(j), Point3(0.0, 0.0, z))
                records.append(DepthFactorRecord(i, j, *rec_args))
                j += 1
        return records, values

    def test_min_factors_fallback_and_clamp(self):
        """Too few factors -> a_i = 1 with fallback; a 5x-corrupted image hits |log a| == clamp."""
        clamp = 0.693
        few = [(2.0, ("unimodal", (2.0,)))] * 4  # 4 < min_factors=5
        corrupted = [(2.0, ("unimodal", (2.0 / 5.0,)))] * 25  # log ratio = log 5 > clamp
        records, values = self._refit_values({0: few, 1: corrupted})
        a, info = refit_depth_scales(records, values, {}, min_factors=5, clamp=clamp)
        self.assertEqual(a[0], 1.0)
        self.assertTrue(info[0]["fallback"])
        self.assertTrue(info[1]["clamped"])
        self.assertAlmostEqual(abs(info[1]["log_a"]), clamp, places=9)
        self.assertAlmostEqual(a[1], np.exp(clamp), places=6)

    def test_null_exclusion_in_refit(self):
        """A null-routed factor is excluded: the fitted a_i is the median over the survivors."""
        z = 2.0
        good_ratios = [1.05, 1.10, 1.20]
        pairs = [(z, ("mixture", (z / r,), (0.1,), (0.0,))) for r in good_ratios]
        pairs.append((z, ("mixture", (50.0,), (0.1,), (0.0,))))  # ~500 sigma off -> null-routed
        records, values = self._refit_values({0: pairs})
        a_null, info_null = refit_depth_scales(records, values, {}, min_factors=3, clamp=0.693, null_nsigma=5.0)
        self.assertEqual(info_null[0]["n_factors_used"], 3)
        self.assertAlmostEqual(a_null[0], 1.10, places=6, msg="median over the non-null factors")
        a_all, info_all = refit_depth_scales(records, values, {}, min_factors=3, clamp=0.693, null_nsigma=None)
        self.assertEqual(info_all[0]["n_factors_used"], 4)
        self.assertNotAlmostEqual(a_all[0], 1.10, places=4, msg="outlier must shift the median when not excluded")

    def test_none_baseline_unaffected_by_new_flags(self):
        """REGRESSION: depth_model='none' with all new flags on is bit-identical to plain 'none'."""
        depth_dir = self._scaled_depth_dir()
        plain, _ = _run_depth_ba(self.data, depth_dir, self.fnames, depth_model="none")
        flagged, _ = _run_depth_ba(
            self.data, depth_dir, self.fnames, depth_model="none",
            depth_per_image_scale=True, depth_relative_sigma=0.05,
        )
        for i in plain.get_valid_camera_indices():
            np.testing.assert_array_equal(
                plain.get_camera(i).pose().matrix(), flagged.get_camera(i).pose().matrix()
            )


if __name__ == "__main__":
    unittest.main()
