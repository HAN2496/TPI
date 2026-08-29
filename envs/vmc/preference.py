import numpy as np

from envs.base import Env

from .features import STEP
from .rollout import VMCEnv as VMCSimulator
from .users import (
    DEFAULT_THETA_COV,
    DEFAULT_THETA_MEAN,
    PARAMETER_NAMES,
    PREFERENCE_FEATURES,
    Users,
)


class VMCPreferenceEnv(Env):

    def __init__(self, controller_bounds, preferred_controller_bounds):
        super().__init__(
            parameter_names=PARAMETER_NAMES,
            parameter_labels=(
                "bias (-mu)",
                "pitch (-w_pitch)",
                "long (-w_long)",
            ),
            feature_groups=("bias", "pitch_rate", "long_accel"),
            controller_names=("kp",),
            controller_bounds=controller_bounds,
            preferred_controller_bounds=preferred_controller_bounds,
            population_mean=DEFAULT_THETA_MEAN,
            population_cov=DEFAULT_THETA_COV,
            bias_index=0,
        )
        self.simulator = VMCSimulator()

    def _sample_users(self, n, seed):
        users = Users.gaussian(
            n,
            seed=seed,
            theta_mean=self.population_mean,
            theta_cov=self.population_cov,
        )
        return users.names, users.parameter_matrix()

    def _sample_controller(self, rng):
        return rng.uniform(
            self.controller_bounds[:, 0],
            self.controller_bounds[:, 1],
        )

    def _rollout(self, controller, seed):
        return self.simulator.rollout(
            {"controller": "p", "kp": float(controller[0])},
            n=1,
            seed=seed,
        )[0]

    def _design_row(self, trajectory):
        features = [
            float(np.mean(STEP[name](trajectory)))
            for name in PREFERENCE_FEATURES
        ]
        return np.asarray([1.0, *features])

    def _episode_reward(self, theta, design_row):
        return design_row @ theta - theta[self.bias_index]

    def scenario_metadata(self, trajectory):
        scenario = trajectory.meta["scenario"]
        return {
            "env_seed": trajectory.meta["seed"],
            "initial_velocity_mps": scenario["initial_velocity_mps"],
            "bump_position_m": scenario["bump_position_m"],
            "bump_half_width_m": scenario["bump_half_width_m"],
            "bump_height_m": scenario["bump_height_m"],
        }


def make_env(cfg):
    return VMCPreferenceEnv(
        controller_bounds=cfg.controller_bounds,
        preferred_controller_bounds=cfg.preferred_controller_bounds,
    )
