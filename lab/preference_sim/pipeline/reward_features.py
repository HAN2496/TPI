from __future__ import annotations

from collections.abc import Mapping

import gymnasium as gym
import mujoco
import numpy as np


STYLE_NAMES = (
    "neutral",
    "sprint",
    "strider",
    "shuffler",
    "bouncer",
    "glider",
    "hopper",
    "croucher",
    "smooth",
)

SIGNAL_NAMES = (
    "forward_speed",
    "control_energy",
    "torso_angle",
    "torso_height",
    "vertical_speed",
    "action_delta",
    "contact_force",
    "foot_contact_right",
    "foot_contact_left",
    "foot_height_right",
    "foot_height_left",
    "fall_indicator",
)

FEATURE_NAMES = (
    "speed_mean",
    "energy_mean",
    "posture_mean",
    "height_mean",
    "bounce_rms",
    "clearance_mean",
    "airtime_frac",
    "sync_frac",
    "cadence",
    "smoothness_rms",
    "impact_p95",
    "survival",
)

DEFAULT_STYLE_PARAMS = {
    "v_ref": 1.0,
    "v_fast": 3.0,
    "foot_rest_height": 0.15,
    "clearance_ref": 0.25,
    "bounce_sigma": 0.5,
    "crouch_height": 1.0,
    "crouch_sigma": 0.15,
    "smooth_sigma": 0.5,
}


def ordered_style_weights(values: Mapping[str, float]) -> np.ndarray:
    unknown = set(values) - set(STYLE_NAMES)
    if unknown:
        raise KeyError(f"Unknown style components: {sorted(unknown)}")
    return np.asarray([float(values.get(name, 0.0)) for name in STYLE_NAMES])


class Walker2dStyleReward(gym.Wrapper):
    """Scalarize per-step progress-gated style components.

    Every component has the form v_gate * g(state) with both factors in [0, 1],
    so any convex weight combination still requires competent forward walking:
    standing still scores zero and falling forfeits all future reward.
    """

    style_names = STYLE_NAMES
    signal_names = SIGNAL_NAMES

    def __init__(
        self,
        env: gym.Env,
        style_weights: Mapping[str, float],
        style_params: Mapping[str, float] | None = None,
        contact_force_scale: float = 1000.0,
        reward_mode: str = "competence_style",
        style_scale: float = 1.0,
        healthy_bonus: float = 0.0,
        fall_penalty: float = 0.0,
    ) -> None:
        super().__init__(env)
        if reward_mode not in {"original", "competence_style", "style_only"}:
            raise ValueError(f"Unknown reward mode: {reward_mode}")
        self.style_weight_map = {
            name: float(style_weights.get(name, 0.0)) for name in STYLE_NAMES
        }
        self.style_weights = ordered_style_weights(style_weights)
        self.params = {**DEFAULT_STYLE_PARAMS, **dict(style_params or {})}
        self.contact_force_scale = float(contact_force_scale)
        self.reward_mode = reward_mode
        self.style_scale = float(style_scale)
        self.healthy_bonus = float(healthy_bonus)
        self.fall_penalty = float(fall_penalty)
        model = self.env.unwrapped.model
        self._floor_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self._foot_geom_right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "foot_geom")
        self._foot_geom_left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "foot_left_geom")
        self._previous_action = np.zeros(self.action_space.shape, dtype=np.float64)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._previous_action.fill(0.0)
        return observation, info

    def _foot_contacts(self) -> tuple[float, bool, bool]:
        base = self.env.unwrapped
        force = np.zeros(6, dtype=np.float64)
        total_normal_force = 0.0
        contact_right = contact_left = False
        for contact_index in range(base.data.ncon):
            contact = base.data.contact[contact_index]
            mujoco.mj_contactForce(base.model, base.data, contact_index, force)
            total_normal_force += max(float(force[0]), 0.0)
            pair = {int(contact.geom1), int(contact.geom2)}
            if self._floor_geom in pair:
                contact_right = contact_right or self._foot_geom_right in pair
                contact_left = contact_left or self._foot_geom_left in pair
        return total_normal_force, contact_right, contact_left

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        observation, _original_reward, terminated, truncated, info = self.env.step(action)
        base = self.env.unwrapped
        p = self.params

        speed = float(info.get("x_velocity", base.data.qvel[0]))
        energy_cost = float(np.mean(np.square(action)))
        torso_angle = float(base.data.qpos[2])
        torso_height = float(base.data.qpos[1])
        vertical_speed = float(base.data.qvel[1])
        action_delta = float(np.sqrt(np.mean(np.square(action - self._previous_action))))
        contact_force, contact_right, contact_left = self._foot_contacts()
        foot_height_right = float(base.data.geom_xpos[self._foot_geom_right, 2])
        foot_height_left = float(base.data.geom_xpos[self._foot_geom_left, 2])
        fall = float(bool(terminated))

        v_gate = float(np.clip(speed / p["v_ref"], 0.0, 1.0))
        swing_height = max(foot_height_right, foot_height_left)
        clearance = float(
            np.clip((swing_height - p["foot_rest_height"]) / p["clearance_ref"], 0.0, 1.0)
        )
        components = np.asarray(
            [
                v_gate,
                float(np.clip(speed / p["v_fast"], 0.0, 1.0)),
                v_gate * clearance,
                v_gate * (1.0 - clearance),
                v_gate * float(not contact_right and not contact_left),
                v_gate * float(contact_right or contact_left)
                * float(np.exp(-((vertical_speed / p["bounce_sigma"]) ** 2))),
                v_gate * float(contact_right == contact_left),
                v_gate * float(np.exp(-(((torso_height - p["crouch_height"]) / p["crouch_sigma"]) ** 2))),
                v_gate * float(np.exp(-((action_delta / p["smooth_sigma"]) ** 2))),
            ],
            dtype=np.float64,
        )
        signals = np.asarray(
            [
                speed, energy_cost, torso_angle, torso_height, vertical_speed, action_delta,
                contact_force, float(contact_right), float(contact_left),
                foot_height_right, foot_height_left, fall,
            ],
            dtype=np.float64,
        )
        style_reward = float(self.style_weights @ components)
        if self.reward_mode == "original":
            reward = float(_original_reward)
        elif self.reward_mode == "competence_style":
            reward = (
                float(_original_reward)
                + self.style_scale * style_reward
                + self.healthy_bonus * float(not terminated)
                - self.fall_penalty * float(terminated)
            )
        else:
            reward = style_reward
        self._previous_action = action.copy()

        info = dict(info)
        info["preference_components"] = components.astype(np.float32)
        info["preference_signals"] = signals.astype(np.float32)
        info["original_reward"] = float(_original_reward)
        info["style_reward"] = style_reward
        info["competence_bonus"] = (
            self.healthy_bonus * float(not terminated)
            - self.fall_penalty * float(terminated)
        )
        info["preference_reward"] = reward
        return observation, reward, terminated, truncated, info


def make_walker2d_env(
    config: dict,
    style_weights: Mapping[str, float],
    reward_mode: str | None = None,
) -> gym.Env:
    env_cfg = config["environment"]
    reward_cfg = config.get("reward", {})
    env = gym.make(
        env_cfg.get("id", "Walker2d-v5"),
        render_mode=None,
        max_episode_steps=int(env_cfg.get("horizon", 500)),
        terminate_when_unhealthy=bool(env_cfg.get("terminate_when_unhealthy", True)),
    )
    return Walker2dStyleReward(
        env,
        style_weights,
        style_params=config.get("styles", {}).get("params"),
        contact_force_scale=float(env_cfg.get("contact_force_scale", 1000.0)),
        reward_mode=reward_mode or str(reward_cfg.get("mode", "competence_style")),
        style_scale=float(reward_cfg.get("style_scale", 1.0)),
        healthy_bonus=float(reward_cfg.get("healthy_bonus", 0.0)),
        fall_penalty=float(reward_cfg.get("fall_penalty", 0.0)),
    )
