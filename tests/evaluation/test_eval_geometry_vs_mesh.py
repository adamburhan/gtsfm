"""Unit tests for point-based reconstruction metrics (eval_geometry_vs_mesh).

Authors: Adam Burhan
"""

import unittest

import numpy as np

from gtsfm.evaluation.eval_geometry import evaluate_points


class TestEvaluatePoints(unittest.TestCase):
    """Verify precision/recall/F-score on a synthetic scene with known floaters."""

    def setUp(self):
        super().setUp()
        # GT: dense grid on the z=0 plane, 1mm spacing over 1m x 1m.
        xs = np.linspace(0.0, 1.0, 1001)
        gx, gy = np.meshgrid(xs, xs)
        self.gt = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)

    def test_perfect_reconstruction(self):
        """Points on the GT surface -> precision 1, distances ~0."""
        rng = np.random.default_rng(0)
        recon = self.gt[rng.choice(len(self.gt), 1000, replace=False)]
        m = evaluate_points(recon, self.gt, taus=[0.05])
        self.assertAlmostEqual(m["precision@5cm"], 1.0)
        self.assertAlmostEqual(m["accuracy_median_m"], 0.0, places=6)

    def test_known_floater_fraction(self):
        """10% of points lifted 10cm off the surface -> precision@5cm = 0.9."""
        rng = np.random.default_rng(1)
        recon = self.gt[rng.choice(len(self.gt), 1000, replace=False)].copy()
        recon[:100, 2] = 0.10  # floaters at 10cm
        m = evaluate_points(recon, self.gt, taus=[0.05])
        self.assertAlmostEqual(m["precision@5cm"], 0.9)
        # p95 sits inside the floater population (top 10%).
        self.assertAlmostEqual(m["accuracy_p95_m"], 0.10, places=3)

    def test_fscore_formula(self):
        """F-score is the harmonic mean of precision and recall."""
        rng = np.random.default_rng(2)
        recon = self.gt[rng.choice(len(self.gt), 1000, replace=False)].copy()
        recon[:500, 2] = 1.0  # half the points far away
        m = evaluate_points(recon, self.gt, taus=[0.05])
        p, r, f = m["precision@5cm"], m["recall@5cm"], m["fscore@5cm"]
        self.assertAlmostEqual(f, 2 * p * r / (p + r), places=9)
        self.assertAlmostEqual(p, 0.5)

    def test_completeness_direction(self):
        """A single recon point cannot cover the GT plane -> recall near zero."""
        recon = np.array([[0.5, 0.5, 0.0]])
        m = evaluate_points(recon, self.gt, taus=[0.05])
        self.assertLess(m["recall@5cm"], 0.01)
        self.assertAlmostEqual(m["precision@5cm"], 1.0)


if __name__ == "__main__":
    unittest.main()
