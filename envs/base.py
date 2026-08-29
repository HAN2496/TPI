from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Trajectory:
    channels: dict
    dt: float
    meta: dict
    obs: np.ndarray = None
    action: np.ndarray = None

    def __post_init__(self):
        if not self.channels:
            raise ValueError("Trajectory channels cannot be empty")
        lengths = {len(values) for values in self.channels.values()}
        if len(lengths) != 1:
            raise ValueError("All trajectory channels must have equal length")
        if self.dt <= 0.0:
            raise ValueError("Trajectory dt must be positive")

    def __len__(self):
        return len(next(iter(self.channels.values())))


class Env(ABC):

    def __init__(
        self,
        parameter_names,
        parameter_labels,
        feature_groups,
        controller_names,
        controller_bounds,
        preferred_controller_bounds,
        population_mean,
        population_cov,
        bias_index=0,
    ):
        self.parameter_names = tuple(parameter_names)
        self.parameter_labels = tuple(parameter_labels)
        self.feature_groups = tuple(feature_groups)
        self.controller_names = tuple(controller_names)
        self.controller_bounds = np.asarray(controller_bounds, dtype=float)
        self.preferred_controller_bounds = np.asarray(
            preferred_controller_bounds, dtype=float,
        )
        self.population_mean = np.asarray(population_mean, dtype=float)
        self.population_cov = np.asarray(population_cov, dtype=float)
        self.bias_index = int(bias_index)

        d = len(self.parameter_names)
        p = len(self.controller_names)
        if not d or not p:
            raise ValueError("Parameter and controller dimensions must be positive")
        if len(self.parameter_labels) != d or len(self.feature_groups) != d:
            raise ValueError("Parameter names, labels, and feature groups must match")
        if self.controller_bounds.shape != (p, 2):
            raise ValueError(f"controller_bounds must have shape ({p}, 2)")
        if self.preferred_controller_bounds.shape != (p, 2):
            raise ValueError(
                f"preferred_controller_bounds must have shape ({p}, 2)"
            )
        if np.any(self.controller_bounds[:, 0] >= self.controller_bounds[:, 1]):
            raise ValueError("Controller lower bounds must be below upper bounds")
        if np.any(
            self.preferred_controller_bounds[:, 0] < self.controller_bounds[:, 0]
        ) or np.any(
            self.preferred_controller_bounds[:, 1] > self.controller_bounds[:, 1]
        ):
            raise ValueError("Preferred controller bounds must lie inside bounds")
        if self.population_mean.shape != (d,):
            raise ValueError(f"population_mean must have shape ({d},)")
        if self.population_cov.shape != (d, d):
            raise ValueError(f"population_cov must have shape ({d}, {d})")
        if not 0 <= self.bias_index < d:
            raise ValueError("bias_index is outside the parameter vector")

    @property
    def parameter_dim(self):
        return len(self.parameter_names)

    @property
    def controller_dim(self):
        return len(self.controller_names)

    def sample_users(self, n, seed):
        names, theta = self._sample_users(n, seed)
        names = tuple(names)
        theta = np.asarray(theta, dtype=float)
        if len(names) != n or len(set(names)) != n:
            raise ValueError("Environment must return n unique user names")
        if theta.shape != (n, self.parameter_dim):
            raise ValueError(
                f"User parameters must have shape ({n}, {self.parameter_dim})"
            )
        return names, theta

    def sample_controller(self, rng):
        controller = np.asarray(self._sample_controller(rng), dtype=float)
        return self._controller(controller)

    def rollout(self, controller, seed):
        controller = self._controller(controller)
        trajectory = self._rollout(controller, int(seed))
        if not isinstance(trajectory, Trajectory):
            raise TypeError("Environment rollout must return one Trajectory")
        return trajectory

    def design_row(self, trajectory):
        row = np.asarray(self._design_row(trajectory), dtype=float)
        if row.shape != (self.parameter_dim,):
            raise ValueError(
                f"Preference feature row must have shape ({self.parameter_dim},)"
            )
        if not np.all(np.isfinite(row)):
            raise FloatingPointError("Preference feature row contains non-finite values")
        return row

    def episode_reward(self, theta, design_row):
        theta = np.asarray(theta, dtype=float)
        design_row = np.asarray(design_row, dtype=float)
        if theta.shape != (self.parameter_dim,):
            raise ValueError(f"theta must have shape ({self.parameter_dim},)")
        if design_row.shape != (self.parameter_dim,):
            raise ValueError(
                f"design_row must have shape ({self.parameter_dim},)"
            )
        return float(self._episode_reward(theta, design_row))

    def format_controller(self, controller):
        controller = self._controller(controller)
        return ", ".join(
            f"{name}={value:.1f}"
            for name, value in zip(self.controller_names, controller)
        )

    def _controller(self, controller):
        controller = np.asarray(controller, dtype=float)
        if controller.shape != (self.controller_dim,):
            raise ValueError(
                f"Controller must have shape ({self.controller_dim},)"
            )
        if np.any(controller < self.controller_bounds[:, 0]) or np.any(
            controller > self.controller_bounds[:, 1]
        ):
            raise ValueError("Controller is outside environment bounds")
        return controller

    @abstractmethod
    def _sample_users(self, n, seed):
        raise NotImplementedError

    @abstractmethod
    def _sample_controller(self, rng):
        raise NotImplementedError

    @abstractmethod
    def _rollout(self, controller, seed):
        raise NotImplementedError

    @abstractmethod
    def _design_row(self, trajectory):
        raise NotImplementedError

    @abstractmethod
    def _episode_reward(self, theta, design_row):
        raise NotImplementedError

    @abstractmethod
    def scenario_metadata(self, trajectory):
        raise NotImplementedError
