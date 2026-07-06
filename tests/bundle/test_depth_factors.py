"""Unit tests for the depth factors used in depth-aware bundle adjustment.

Covers the unimodal and bimodal (max-mixture) camera-frame-Z depth factors,
including a finite-difference check of the analytic Jacobians, and the
DepthProvider's largest-gap ambiguity analysis.

Authors: Adam Burhan
"""

import tempfile
import unittest
from pathlib import Path

import gtsam  # type: ignore
import numpy as np
from gtsam import Point3, Pose3, Rot3, Values
from gtsam.symbol_shorthand import A, P, X  # type: ignore

from gtsfm.bundle.bundle_adjustment import make_bimodal_depth_factor, make_depth_factor, make_log_depth_factor
from gtsfm.common.depth_provider import DepthProvider

DEPTH_NOISE = gtsam.noiseModel.Isotropic.Sigma(1, 0.1)


def _make_values(wTc: Pose3, point_w: np.ndarray) -> Values:
    values = Values()
    values.insert(X(0), wTc)
    values.insert(P(0), Point3(point_w))
    return values


class TestDepthFactors(unittest.TestCase):
    """Unit tests for the unimodal and bimodal depth factors."""

    def setUp(self):
        super().setUp()
        # Non-trivial pose so the test exercises rotation in transformTo.
        self.wTc = Pose3(Rot3.RzRyRx(0.1, -0.2, 0.3), np.array([0.5, -1.0, 2.0]))
        # A point 3m in front of the camera (camera-frame Z = 3.0).
        self.point_c = np.array([0.4, -0.2, 3.0])
        self.point_w = self.wTc.transformFrom(Point3(self.point_c))
        self.values = _make_values(self.wTc, self.point_w)

    def test_unimodal_residual(self):
        """Residual is z_pred - d."""
        factor = make_depth_factor(X(0), P(0), d=2.5, noise=DEPTH_NOISE)
        residual = factor.unwhitenedError(self.values)
        self.assertAlmostEqual(residual[0], 3.0 - 2.5, places=9)

    def test_bimodal_selects_closer_mode(self):
        """The bimodal factor's residual uses whichever hypothesis is closer to z_pred."""
        # z_pred = 3.0; d=1.0 (residual 2.0) vs d_alt=3.5 (residual -0.5) -> picks d_alt.
        factor = make_bimodal_depth_factor(X(0), P(0), d=1.0, d_alt=3.5, noise=DEPTH_NOISE)
        residual = factor.unwhitenedError(self.values)
        self.assertAlmostEqual(residual[0], 3.0 - 3.5, places=9)

        # Swapped hypotheses give the same result (selection is symmetric).
        factor_swapped = make_bimodal_depth_factor(X(0), P(0), d=3.5, d_alt=1.0, noise=DEPTH_NOISE)
        residual_swapped = factor_swapped.unwhitenedError(self.values)
        self.assertAlmostEqual(residual_swapped[0], 3.0 - 3.5, places=9)

    def test_bimodal_matches_unimodal_on_selected_mode(self):
        """Bimodal factor == unimodal factor built on the winning hypothesis (residual AND Jacobians)."""
        bimodal = make_bimodal_depth_factor(X(0), P(0), d=1.0, d_alt=3.5, noise=DEPTH_NOISE)
        unimodal = make_depth_factor(X(0), P(0), d=3.5, noise=DEPTH_NOISE)
        gf_bi = bimodal.linearize(self.values)
        gf_uni = unimodal.linearize(self.values)
        np.testing.assert_allclose(gf_bi.jacobian()[0], gf_uni.jacobian()[0], atol=1e-9)
        np.testing.assert_allclose(gf_bi.jacobian()[1], gf_uni.jacobian()[1], atol=1e-9)

    def test_jacobians_match_finite_differences(self):
        """Analytic Jacobians of the bimodal factor agree with central finite differences.

        Perturbations are applied on the tangent space (Pose3 retract / point addition),
        matching GTSAM's Jacobian convention.
        """
        factor = make_bimodal_depth_factor(X(0), P(0), d=1.0, d_alt=3.5, noise=DEPTH_NOISE)
        gf = factor.linearize(self.values)
        A, _ = gf.jacobian()
        # Columns: [pose (6), point (3)]; whitened by noise -> unwhiten for comparison.
        sigma = 0.1
        H_pose, H_point = A[:, :6] * sigma, A[:, 6:] * sigma

        eps = 1e-6

        def residual(wTc: Pose3, point_w: np.ndarray) -> float:
            return factor.unwhitenedError(_make_values(wTc, point_w))[0]

        for k in range(6):
            delta = np.zeros(6)
            delta[k] = eps
            r_plus = residual(self.wTc.retract(delta), self.point_w)
            r_minus = residual(self.wTc.retract(-delta), self.point_w)
            self.assertAlmostEqual(H_pose[0, k], (r_plus - r_minus) / (2 * eps), places=5)

        for k in range(3):
            delta = np.zeros(3)
            delta[k] = eps
            r_plus = residual(self.wTc, self.point_w + delta)
            r_minus = residual(self.wTc, self.point_w - delta)
            self.assertAlmostEqual(H_point[0, k], (r_plus - r_minus) / (2 * eps), places=5)


UNIT_NOISE = gtsam.noiseModel.Isotropic.Sigma(1, 1.0)


class TestLogDepthFactor(unittest.TestCase):
    """Unit tests for the log-space depth factor with a per-image alpha offset."""

    def setUp(self):
        super().setUp()
        self.wTc = Pose3(Rot3.RzRyRx(0.1, -0.2, 0.3), np.array([0.5, -1.0, 2.0]))
        self.point_c = np.array([0.4, -0.2, 3.0])
        self.point_w = self.wTc.transformFrom(Point3(self.point_c))
        self.alpha = 0.2
        self.values = self._make_values(self.wTc, self.point_w, self.alpha)

    @staticmethod
    def _make_values(wTc: Pose3, point_w: np.ndarray, alpha: float) -> Values:
        values = Values()
        values.insert(X(0), wTc)
        values.insert(P(0), Point3(point_w))
        values.insert(A(0), alpha)
        return values

    def test_unimodal_residual(self):
        """Residual is (log z_pred - log d - alpha) / rel_sigma."""
        d, rel_sigma = 2.5, 0.05
        factor = make_log_depth_factor(X(0), P(0), A(0), [d], [rel_sigma], [0.0], UNIT_NOISE)
        residual = factor.unwhitenedError(self.values)
        self.assertAlmostEqual(residual[0], (np.log(3.0) - np.log(d) - self.alpha) / rel_sigma, places=9)

    def test_alpha_shifts_mode_selection(self):
        """Both modes share alpha, so mode selection runs on bias-corrected residuals."""
        # z_pred = 3.0. With alpha = 0: log 3 - log 2 = 0.405 vs log 3 - log 4 = -0.288 -> far mode wins.
        factor = make_log_depth_factor(X(0), P(0), A(0), [2.0, 4.0], [0.1, 0.1], [0.0, 0.0], UNIT_NOISE)
        values = self._make_values(self.wTc, self.point_w, 0.0)
        self.assertAlmostEqual(factor.unwhitenedError(values)[0], (np.log(3.0) - np.log(4.0)) / 0.1, places=9)
        # With alpha = 0.405 (image depths biased small), the near mode wins instead.
        alpha = float(np.log(3.0) - np.log(2.0))
        values = self._make_values(self.wTc, self.point_w, alpha)
        self.assertAlmostEqual(factor.unwhitenedError(values)[0], 0.0, places=9)

    def test_null_hypothesis_opts_out(self):
        """If even the best mode is > null_nsigma away, residual and Jacobians are zero."""
        factor = make_log_depth_factor(X(0), P(0), A(0), [0.1], [0.01], [0.0], UNIT_NOISE, null_nsigma=3.0)
        self.assertAlmostEqual(factor.unwhitenedError(self.values)[0], 0.0, places=9)
        A_mat, b = factor.linearize(self.values).jacobian()
        np.testing.assert_allclose(A_mat, 0.0, atol=1e-12)

    def test_behind_camera_opts_out(self):
        """Non-positive predicted depth disables the factor instead of taking log of z <= 0."""
        point_c_behind = np.array([0.4, -0.2, -1.0])
        point_w_behind = self.wTc.transformFrom(Point3(point_c_behind))
        values = self._make_values(self.wTc, point_w_behind, self.alpha)
        factor = make_log_depth_factor(X(0), P(0), A(0), [2.5], [0.05], [0.0], UNIT_NOISE)
        self.assertAlmostEqual(factor.unwhitenedError(values)[0], 0.0, places=9)
        A_mat, b = factor.linearize(values).jacobian()
        np.testing.assert_allclose(A_mat, 0.0, atol=1e-12)

    def test_jacobians_match_finite_differences(self):
        """Analytic Jacobians (pose, point, alpha) agree with central finite differences."""
        factor = make_log_depth_factor(X(0), P(0), A(0), [1.0, 3.5], [0.05, 0.08], [0.0, 0.0], UNIT_NOISE)
        A_mat, _ = factor.linearize(self.values).jacobian()
        # Columns: [pose (6), point (3), alpha (1)]; unit noise, so A is the raw Jacobian.
        H_pose, H_point, H_alpha = A_mat[:, :6], A_mat[:, 6:9], A_mat[:, 9:]

        eps = 1e-6

        def residual(wTc: Pose3, point_w: np.ndarray, alpha: float) -> float:
            return factor.unwhitenedError(self._make_values(wTc, point_w, alpha))[0]

        for k in range(6):
            delta = np.zeros(6)
            delta[k] = eps
            r_plus = residual(self.wTc.retract(delta), self.point_w, self.alpha)
            r_minus = residual(self.wTc.retract(-delta), self.point_w, self.alpha)
            self.assertAlmostEqual(H_pose[0, k], (r_plus - r_minus) / (2 * eps), places=4)

        for k in range(3):
            delta = np.zeros(3)
            delta[k] = eps
            r_plus = residual(self.wTc, self.point_w + delta, self.alpha)
            r_minus = residual(self.wTc, self.point_w - delta, self.alpha)
            self.assertAlmostEqual(H_point[0, k], (r_plus - r_minus) / (2 * eps), places=4)

        r_plus = residual(self.wTc, self.point_w, self.alpha + eps)
        r_minus = residual(self.wTc, self.point_w, self.alpha - eps)
        self.assertAlmostEqual(H_alpha[0, 0], (r_plus - r_minus) / (2 * eps), places=4)


class TestDepthProviderHypotheses(unittest.TestCase):
    """Unit tests for DepthProvider's largest-gap ambiguity analysis."""

    def _make_provider(self, depth_map: np.ndarray, **kwargs) -> DepthProvider:
        tmp_dir = tempfile.mkdtemp()
        np.save(Path(tmp_dir) / "depth000000.npy", depth_map)
        defaults = dict(
            depth_min=0.1,
            depth_max=20.0,
            depth_filename_template="depth{:06d}.npy",
            compute_hypotheses=True,
            patch_radius=5,
            gap_thresh=0.15,
            ambiguity_thresh=0.20,
            min_valid=10,
        )
        defaults.update(kwargs)
        return DepthProvider(tmp_dir, image_fnames={0: "frame000000.jpg"}, **defaults)

    def test_flat_patch_not_ambiguous(self):
        """Uniform depth -> no discontinuity, sample is unambiguous."""
        provider = self._make_provider(np.full((50, 50), 2.0))
        sample = provider.get_depth(0, u=25.0, v=25.0)
        assert sample is not None
        self.assertFalse(sample.ambiguous)
        self.assertIsNone(sample.depth_alt)
        self.assertAlmostEqual(sample.depth, 2.0)

    def test_two_plane_patch_is_ambiguous(self):
        """Pixel on a fg/bg boundary -> ambiguous, with the far plane as the alternative."""
        depth_map = np.full((50, 50), 1.0)
        depth_map[:, 25:] = 3.0  # right half is background
        provider = self._make_provider(depth_map)
        # Sample on the near plane, right at the boundary -> balanced patch.
        sample = provider.get_depth(0, u=24.0, v=25.0)
        assert sample is not None
        self.assertTrue(sample.ambiguous)
        self.assertAlmostEqual(sample.depth, 1.0)
        self.assertAlmostEqual(sample.depth_alt, 3.0)
        self.assertGreater(sample.score, 0.20)

        # Sampling on the far side flips depth and the alternative.
        sample_far = provider.get_depth(0, u=26.0, v=25.0)
        assert sample_far is not None
        self.assertTrue(sample_far.ambiguous)
        self.assertAlmostEqual(sample_far.depth, 3.0)
        self.assertAlmostEqual(sample_far.depth_alt, 1.0)

    def test_unbalanced_modes_not_ambiguous(self):
        """A tiny far-depth region has a large gap but poor balance -> below score threshold."""
        depth_map = np.full((50, 50), 1.0)
        depth_map[25, 27] = 3.0  # single far pixel in the patch
        provider = self._make_provider(depth_map)
        sample = provider.get_depth(0, u=25.0, v=25.0)
        assert sample is not None
        self.assertFalse(sample.ambiguous)
        self.assertIsNone(sample.depth_alt)

    def test_hypotheses_disabled_returns_plain_sample(self):
        """With compute_hypotheses=False, even a boundary pixel is returned unanalyzed."""
        depth_map = np.full((50, 50), 1.0)
        depth_map[:, 25:] = 3.0
        provider = self._make_provider(depth_map, compute_hypotheses=False)
        sample = provider.get_depth(0, u=24.0, v=25.0)
        assert sample is not None
        self.assertFalse(sample.ambiguous)
        self.assertIsNone(sample.depth_alt)
        self.assertAlmostEqual(sample.depth, 1.0)

    def test_out_of_range_center_returns_none(self):
        """Center depth outside [depth_min, depth_max] -> None, as before."""
        provider = self._make_provider(np.zeros((50, 50)))
        self.assertIsNone(provider.get_depth(0, u=25.0, v=25.0))


if __name__ == "__main__":
    unittest.main()
