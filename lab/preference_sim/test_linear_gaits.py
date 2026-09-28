"""Checks for the trajectory-reward and feedback contracts."""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.common import load_config
from pipeline.linear_env import FEATURE_NAMES, additive_features, make_env, rollout, weight_vector
from pipeline.linear_experiment import users


class LinearRewardTests(unittest.TestCase):
    def test_static_poses_cannot_collect_style_reward(self):
        signal = np.array([0., 1.08, 0., 0., 0., 0., 0., 0., 1., 1., 0., 0.])
        np.testing.assert_array_equal(additive_features(signal, True), [1., 0., 0., 0., 0., 0., 0., 0., 0.])
        np.testing.assert_array_equal(additive_features(signal, False), np.zeros(9))

    def test_speed_targets_and_style_decoupling(self):
        def features(speed, height=1.08, angle=0., airborne=0.):
            signal = np.array([speed, height, angle, .1, 1., airborne, .2, .5, 0., 0., 0., 0.])
            return additive_features(signal, True)
        self.assertGreater(features(1.2)[2], features(2.1)[2])
        self.assertGreater(features(2.1)[3], features(1.2)[3])
        self.assertGreater(features(2.1)[3], features(3.0)[3])
        self.assertEqual(features(1.2, angle=.4)[5], features(1.2)[5])
        self.assertEqual(features(2.1, airborne=1.)[3], features(2.1)[3])

    def test_touchdown_alternation_rejects_same_foot_and_simultaneous_events(self):
        env = make_env({"environment": {"horizon": 20, "feature_version": 5}}, {})
        try:
            env.reset(seed=1)
            env.steps = 10
            env._register_touchdowns(np.array([True, False]))
            env._register_touchdowns(np.array([False, False]))
            env.steps = 20
            env._register_touchdowns(np.array([True, False]))
            self.assertEqual(env.touchdowns, 2)
            self.assertEqual(env.alternating_touchdowns, 0)
            env._register_touchdowns(np.array([False, False]))
            env.steps = 30
            env._register_touchdowns(np.array([False, True]))
            self.assertEqual(env.alternating_touchdowns, 1)
            env._register_touchdowns(np.array([False, False]))
            env.steps = 40
            env._register_touchdowns(np.array([True, True]))
            self.assertEqual(env.touchdowns, 3)
        finally:
            env.close()

    def test_role_exchange_requires_both_legs_to_lead(self):
        env = make_env({"environment": {"horizon": 100, "feature_version": 5}}, {})
        try:
            env.reset(seed=2)
            env.steps = 10
            env._update_role_exchange(0.2, 1.0)
            env.steps = 20
            env._update_role_exchange(-0.2, 1.0)
            self.assertEqual(env.role_switches, 1)
            np.testing.assert_allclose(env.lead_time[0], env.lead_time[1])
            env.steps = 30
            env._update_role_exchange(-0.3, 1.0)
            self.assertEqual(env.role_switches, 1)
        finally:
            env.close()

    def test_real_mujoco_reward_telescopes_to_exact_dot_product(self):
        config = {"environment": {"horizon": 20, "feature_version": 5}}
        values = np.linspace(-1.3, 2.7, len(FEATURE_NAMES))
        weights = dict(zip(FEATURE_NAMES, values))
        env = make_env(config, weights)
        try:
            env.reset(seed=12)
            rng = np.random.default_rng(12)
            rewards = []
            final_phi = None
            for _ in range(20):
                _, reward, done, truncated, info = env.step(rng.uniform(-1., 1., 6))
                rewards.append(reward)
                final_phi = info["phi_prefix"]
                self.assertAlmostEqual(reward, weight_vector(weights) @ info["phi_delta"], places=12)
                if done or truncated:
                    break
            self.assertAlmostEqual(sum(rewards), weight_vector(weights) @ final_phi, places=11)
            self.assertTrue(np.all((final_phi >= 0) & (final_phi <= 1)))
        finally:
            env.close()

    def test_early_fall_uses_fixed_horizon_and_seeds_replay(self):
        env = make_env({"environment": {"horizon": 1000, "feature_version": 5}}, {})
        try:
            first = rollout(None, env, 9, 1000)
            second = rollout(None, env, 9, 1000)
            self.assertLess(first["length"], 1000)
            np.testing.assert_array_equal(first["signals"], second["signals"])
            np.testing.assert_allclose(first["phi"], first["step_phi"].sum(axis=0))
            self.assertLess(first["phi"][0], 1.)
        finally:
            env.close()

    def test_feedback_has_no_hidden_term_or_test_calibration(self):
        config, _ = load_config("configs/walker2d_role_exchange.yaml")
        config = copy.deepcopy(config)
        config["users"].update(n_train=3, n_test=2, labels_per_user=5)
        with tempfile.TemporaryDirectory(prefix="linear_gait_test_") as tmp:
            directory = Path(tmp)
            for name in ("tables", "rollouts", "exports", "reports"):
                (directory / name).mkdir()
            splits = np.repeat(["calibration", "context", "test"], 10)
            candidate = np.tile([True] * 8 + [False] * 2, 3)
            pd.DataFrame({"split": splits, "candidate": candidate,
                          "profile": np.where(candidate, "walker", "diagnostic")}).to_csv(
                              directory / "tables" / "episodes.csv", index=False)
            phi = np.random.default_rng(6).random((30, len(FEATURE_NAMES)))
            np.savez_compressed(directory / "rollouts" / "bank.npz", phi=phi)
            report = users(config, directory)
            self.assertEqual(report["n_train_users"], 3)
            self.assertEqual(report["n_test_users"], 2)
            self.assertEqual(report["n_probe_users"], 1)
            with np.load(directory / "exports" / "linear_preferences.npz") as result:
                theta = result["theta_true"].copy()
                thresholds = result["thresholds"].copy()
                center = result["feature_center"].copy()
                weights = result["weights_true"].copy()
                expected = float(result["beta"]) * (weights @ phi.T - thresholds[:, None])
                np.testing.assert_allclose(theta @ result["Z"].T, expected, atol=1e-12)
                self.assertTrue(np.all(splits[result["context_indices"]] == "context"))
                self.assertTrue(np.all(result["labels"][:, splits == "calibration"] == -1))
                self.assertEqual(result["user_split"][-1], "probe")
            phi[splits == "test"] *= 12
            np.savez_compressed(directory / "rollouts" / "bank.npz", phi=phi)
            users(config, directory)
            with np.load(directory / "exports" / "linear_preferences.npz") as result:
                np.testing.assert_array_equal(result["theta_true"], theta)
                np.testing.assert_array_equal(result["thresholds"], thresholds)
                np.testing.assert_array_equal(result["feature_center"], center)


if __name__ == "__main__":
    unittest.main()
