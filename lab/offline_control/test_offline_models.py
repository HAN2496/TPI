"""Numerical contracts that protect value estimates and conservative training."""
import unittest

import numpy as np

from preference_loop.offline_models import (
    ConservativeRewardModels, FeatureRegressor, bounded_kernel_weights,
    continuous_values,
)


class OfflineModelTests(unittest.TestCase):
    def test_kernel_respects_physical_units_and_boundaries(self):
        actions = np.linspace(30., 300., 20001)
        weights = bounded_kernel_weights([30., 165., 300.], actions, (30., 300.), 20., 1 / 270.)
        np.testing.assert_allclose(np.trapezoid(weights / 270., actions, axis=1), 1., atol=1e-7)
        normalized = bounded_kernel_weights([0., .5, 1.], (actions - 30) / 270, (0., 1.), 20 / 270, 1.)
        np.testing.assert_allclose(weights, normalized, atol=1e-12)

    def test_correct_model_leaves_dr_unchanged(self):
        obs = np.arange(12.).reshape(6, 2)
        dm = np.array([[2., 3.], [5., 7.]])
        weights = np.ones((2, 6))
        value = continuous_values(dm, obs, obs.copy(), weights)
        np.testing.assert_array_equal(value["dr"], dm)
        np.testing.assert_array_equal(value["ess"], [6., 6.])

    def test_reward_projection_commutes_with_dr(self):
        rng = np.random.default_rng(0)
        obs, pred, dm = rng.normal(size=(3, 6, 2))
        weights = rng.uniform(size=(6, 6))
        theta = np.array([[-2., -1.], [-.5, -3.]]).T
        features = continuous_values(dm, obs, pred, weights)
        rewards = continuous_values(dm @ theta, obs @ theta, pred @ theta, weights)
        np.testing.assert_allclose(features["dr"] @ theta, rewards["dr"], atol=1e-12)

    def test_known_density_corrects_biased_model_on_independent_log(self):
        # Deterministic stratified uniform logging integrates the kernel. The
        # outcome model misses a constant offset; residual correction recovers
        # it even at the two gain boundaries.
        n = 20000
        actions = 30. + (np.arange(n) + .5) * 270. / n
        weights = bounded_kernel_weights([30., 165., 300.], actions, (30., 300.), 20., 1 / 270.)
        observed = np.full((n, 1), 7.)
        predicted = np.full((n, 1), 2.)
        result = continuous_values(np.full((3, 1), 2.), observed, predicted, weights)
        np.testing.assert_allclose(result["dr"], 7., atol=1e-7)

    def test_quadratic_ols_baseline_handles_constant_covariates(self):
        x = np.column_stack([np.linspace(30., 300., 50), np.ones(50)])
        y = np.column_stack([x[:, 0] / 300, (x[:, 0] / 300) ** 2])
        pred = FeatureRegressor().fit(x, y).predict(x)
        self.assertTrue(np.isfinite(pred).all())
        self.assertLess(np.mean((pred - y) ** 2), 1e-5)

    def test_conservative_training_and_checkpoint_are_finite(self):
        rng = np.random.default_rng(2)
        x = rng.uniform(size=(48, 2))
        rewards = np.column_stack([-x[:, 0] ** 2, -(1 - x[:, 0]) ** 2])
        model = ConservativeRewardModels(alpha=.1, steps=5, hidden=8, device="cpu").fit(x, rewards, (0., 1.))
        prediction = model.predict(x)
        self.assertEqual(prediction.shape, rewards.shape)
        self.assertTrue(np.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()
