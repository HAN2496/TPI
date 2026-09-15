"""One trajectory feature basis for PPO rewards and user utilities.

The environment reward at a step is the change in the trajectory-prefix
utility. Consequently, the undiscounted episode return telescopes exactly to
``weights @ final_phi``. Gymnasium's native reward is never added.
"""
from __future__ import annotations

import gymnasium as gym
import mujoco
import numpy as np


FEATURE_NAMES = (
    "survival", "progress", "walk_speed", "run_speed", "upright", "crouch",
    "straight_legs", "smooth", "flight", "alternation", "role_exchange",
    "stance_balance", "push_balance",
)
SIGNAL_NAMES = (
    "speed", "height", "torso_abs_angle", "knee_abs_angle", "knee_abs_speed",
    "airborne", "action_delta", "energy", "foot_contact_right", "foot_contact_left",
    "foot_x_difference", "right_ahead", "left_ahead",
    "right_only_contact", "left_only_contact", "action_energy_right", "action_energy_left",
    "forward_force_right", "forward_force_left",
)


def weight_vector(weights: dict) -> np.ndarray:
    unknown = set(weights) - set(FEATURE_NAMES)
    if unknown:
        raise ValueError(f"Unknown linear reward features: {sorted(unknown)}")
    values = np.asarray([weights.get(name, 0.0) for name in FEATURE_NAMES], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Reward weights must be finite")
    return values


class PeriodicMirrorPolicy:
    """Alternate a sagittal Walker controller with its exact leg-label mirror."""

    def __init__(self, policy, period):
        self.policy = policy
        self.period = int(period)
        if self.period <= 0:
            raise ValueError("Periodic mirror period must be positive")
        self.steps = 0

    def reset(self):
        self.steps = 0

    def predict(self, observation, deterministic=True):
        mirrored = (self.steps // self.period) % 2 == 1
        self.steps += 1
        if not mirrored:
            return self.policy.predict(observation, deterministic=deterministic)
        swapped = np.array(observation, copy=True)
        swapped[2:5], swapped[5:8] = observation[5:8], observation[2:5]
        swapped[11:14], swapped[14:17] = observation[14:17], observation[11:14]
        action = self.policy.predict(swapped, deterministic=deterministic)[0]
        return np.r_[action[3:6], action[0:3]], None


def transform_policy(policy, transform):
    if not transform:
        return policy
    if transform.get("type") == "periodic_mirror":
        return PeriodicMirrorPolicy(policy, transform["period"])
    raise ValueError(f"Unknown policy transform: {transform.get('type')}")


def additive_features(signals: np.ndarray, alive: bool) -> np.ndarray:
    """Nine per-step contributions, before fixed-horizon normalization."""
    speed, height, angle, knee, knee_speed, airborne, delta, *_ = signals
    move = float(alive) * float(np.clip(speed / 0.8, 0.0, 1.0))
    return np.asarray([
        float(alive),
        move,
        move * np.exp(-np.square((speed - 1.2) / 0.55)),
        move * np.exp(-np.square((speed - 2.1) / 0.45)),
        move * np.exp(-np.square(angle / 0.18)),
        move * np.exp(-np.square((height - 1.08) / 0.15)),
        move * np.exp(-knee / 0.12) * np.exp(-knee_speed / 6.0),
        move * np.exp(-np.square(delta / 0.5)),
        move * float(airborne),
    ], dtype=np.float64)


class LinearGaitEnv(gym.Wrapper):
    """Walker2d with stateful, fixed-horizon trajectory features."""

    def __init__(self, env: gym.Env, weights: dict, feature_version=5, horizon=1000,
                 touchdown_rate=2.0, touchdown_refractory=8, role_margin=0.08,
                 role_switch_rate=1.5, role_switch_refractory=10):
        super().__init__(env)
        if int(feature_version) != 5:
            raise ValueError("This experiment implements only trajectory feature version 5")
        self.weights = weight_vector(weights)
        self.feature_version = int(feature_version)
        self.horizon = int(horizon)
        self.touchdown_rate = float(touchdown_rate)
        self.touchdown_refractory = int(touchdown_refractory)
        self.role_margin = float(role_margin)
        self.role_switch_rate = float(role_switch_rate)
        self.role_switch_refractory = int(role_switch_refractory)
        model = self.unwrapped.model
        self.floor = model.geom("floor").id
        self.feet = (model.geom("foot_geom").id, model.geom("foot_left_geom").id)
        joints = [model.joint(name) for name in ("leg_joint", "leg_left_joint")]
        self.knee_qpos = [int(j.qposadr[0]) for j in joints]
        self.knee_qvel = [int(j.dofadr[0]) for j in joints]
        self.previous_action = np.zeros(self.action_space.shape)
        self._reset_feature_state()

    def _reset_feature_state(self):
        self.steps = 0
        self.alive_steps = 0
        self.additive_sum = np.zeros(9, dtype=np.float64)
        self.previous_phi = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
        self.previous_contacts = np.zeros(2, dtype=bool)
        self.last_touchdown_step = np.full(2, -10_000, dtype=int)
        self.last_touchdown_foot = -1
        self.touchdowns = 0
        self.alternating_touchdowns = 0
        self.stance_time = np.zeros(2, dtype=np.float64)
        self.forward_impulse = np.zeros(2, dtype=np.float64)
        self.lead_time = np.zeros(2, dtype=np.float64)
        self.lead_foot = -1
        self.role_switches = 0
        self.last_role_switch_step = -10_000

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous_action.fill(0)
        self._reset_feature_state()
        return obs, info

    def _foot_contacts_and_forces(self):
        data, model = self.unwrapped.data, self.unwrapped.model
        contacts = np.zeros(2, dtype=bool)
        forward_force = np.zeros(2, dtype=np.float64)
        contact_force = np.zeros(6, dtype=np.float64)
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            if contact.dist > 0:
                continue
            pair = (int(contact.geom1), int(contact.geom2))
            if self.floor not in pair:
                continue
            for foot_index, foot in enumerate(self.feet):
                if foot not in pair:
                    continue
                contacts[foot_index] = True
                mujoco.mj_contactForce(model, data, contact_id, contact_force)
                world_force = np.asarray(contact.frame).reshape(3, 3).T @ contact_force[:3]
                # The sign makes this the ground force acting on the foot even
                # if geom order changes in a future Walker2d XML.
                sign = 1.0 if pair[0] == self.floor else -1.0
                forward_force[foot_index] += max(0.0, sign * float(world_force[0]))
        return contacts, forward_force

    def _register_touchdowns(self, contacts):
        rising = contacts & ~self.previous_contacts
        # Simultaneous landings are not evidence of left-right alternation.
        if int(rising.sum()) == 1:
            foot = int(np.flatnonzero(rising)[0])
            if self.steps - self.last_touchdown_step[foot] >= self.touchdown_refractory:
                if self.last_touchdown_foot >= 0 and foot != self.last_touchdown_foot:
                    self.alternating_touchdowns += 1
                self.touchdowns += 1
                self.last_touchdown_foot = foot
                self.last_touchdown_step[foot] = self.steps
        self.previous_contacts = contacts.copy()

    def _prefix_phi(self):
        phi = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
        phi[:9] = self.additive_sum / self.horizon
        progress = phi[1]
        elapsed_seconds = max(self.alive_steps * float(self.unwrapped.dt), 1e-12)
        required = max(self.touchdown_rate * elapsed_seconds, 1.0)
        coverage = min(1.0, self.touchdowns / required)
        ratio = self.alternating_touchdowns / max(self.touchdowns - 1, 1)
        phi[9] = progress * ratio * coverage
        lead_total = float(self.lead_time.sum())
        if lead_total > 1e-12:
            lead_balance = 2.0 * float(self.lead_time.min()) / lead_total
            required_switches = max(self.role_switch_rate * elapsed_seconds, 1.0)
            switch_coverage = min(1.0, self.role_switches / required_switches)
            phi[10] = progress * lead_balance * switch_coverage
        stance_total = float(self.stance_time.sum())
        push_total = float(self.forward_impulse.sum())
        if stance_total > 1e-12:
            phi[11] = progress * 2.0 * float(self.stance_time.min()) / stance_total
        if push_total > 1e-12:
            phi[12] = progress * 2.0 * float(self.forward_impulse.min()) / push_total
        return phi

    def _update_role_exchange(self, difference, move):
        lead = 0 if difference > self.role_margin else (1 if difference < -self.role_margin else -1)
        if lead < 0:
            return
        self.lead_time[lead] += move * float(self.unwrapped.dt)
        if self.lead_foot >= 0 and lead != self.lead_foot:
            if self.steps - self.last_role_switch_step >= self.role_switch_refractory:
                self.role_switches += 1
                self.last_role_switch_step = self.steps
        self.lead_foot = lead

    def step(self, action):
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        data = self.unwrapped.data
        contacts, forward_force = self._foot_contacts_and_forces()
        foot_x_difference = float(data.geom_xpos[self.feet[0], 0] - data.geom_xpos[self.feet[1], 0])
        action = np.asarray(action)
        signal = np.asarray([
            float(info["x_velocity"]), float(data.qpos[1]), abs(float(data.qpos[2])),
            np.abs(data.qpos[self.knee_qpos]).mean(),
            np.abs(data.qvel[self.knee_qvel]).mean(), float(not any(contacts)),
            np.sqrt(np.square(action - self.previous_action).mean()),
            np.square(action).mean(), *map(float, contacts), foot_x_difference,
            float(foot_x_difference > self.role_margin), float(foot_x_difference < -self.role_margin),
            float(contacts[0] and not contacts[1]), float(contacts[1] and not contacts[0]),
            np.square(action[:3]).mean(), np.square(action[3:]).mean(), *forward_force,
        ], dtype=np.float64)
        alive = not terminated
        self.steps += 1
        self.alive_steps += int(alive)
        additive = additive_features(signal, alive)
        self.additive_sum += additive
        move = additive[1]
        exclusive = contacts & ~contacts[::-1]
        self.stance_time += move * exclusive.astype(float) * float(self.unwrapped.dt)
        self.forward_impulse += move * forward_force * float(self.unwrapped.dt)
        if alive:
            self._register_touchdowns(contacts)
            self._update_role_exchange(foot_x_difference, move)
        else:
            self.previous_contacts = contacts.copy()
        phi_prefix = self._prefix_phi()
        phi_delta = phi_prefix - self.previous_phi
        reward = float(self.weights @ phi_delta)
        self.previous_phi = phi_prefix.copy()
        self.previous_action = action.copy()
        return obs, reward, terminated, truncated, {
            **info, "phi": phi_delta, "phi_delta": phi_delta, "phi_prefix": phi_prefix,
            "signals": signal, "original_reward_unused": float(original_reward),
            "touchdowns": self.touchdowns,
            "alternating_touchdowns": self.alternating_touchdowns,
            "role_switches": self.role_switches,
        }


def make_env(config: dict, weights: dict, render_mode=None, horizon=None):
    env_cfg = config["environment"]
    horizon = int(horizon or env_cfg["horizon"])
    kwargs = dict(env_cfg.get("kwargs", {}))
    if render_mode:
        kwargs.update(width=360, height=360)
    env = gym.make(
        "Walker2d-v5", render_mode=render_mode, max_episode_steps=horizon,
        terminate_when_unhealthy=True, **kwargs,
    )
    return LinearGaitEnv(
        env, weights, feature_version=env_cfg.get("feature_version", 5), horizon=horizon,
        touchdown_rate=env_cfg.get("touchdown_rate", 2.0),
        touchdown_refractory=env_cfg.get("touchdown_refractory", 8),
        role_margin=env_cfg.get("role_margin", 0.08),
        role_switch_rate=env_cfg.get("role_switch_rate", 1.5),
        role_switch_refractory=env_cfg.get("role_switch_refractory", 10),
    )


def rollout(policy, env, seed: int, horizon: int, noise: float = 0.0, video=False):
    if policy is not None and hasattr(policy, "reset"):
        policy.reset()
    obs, _ = env.reset(seed=int(seed))
    rng = np.random.default_rng(seed)
    phis, signals, observations, actions, frames = [], [], [], [], []
    if video:
        frames.append(env.render())
    terminated = truncated = False
    final_phi = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    for _ in range(horizon):
        action = np.zeros(env.action_space.shape) if policy is None else policy.predict(obs, deterministic=True)[0]
        if noise:
            action = action + rng.normal(0, noise, size=action.shape)
        action = np.clip(action, env.action_space.low, env.action_space.high)
        observations.append(obs.copy())
        obs, reward, terminated, truncated, info = env.step(action)
        phis.append(info["phi_delta"])
        final_phi = info["phi_prefix"]
        signals.append(info["signals"])
        actions.append(action)
        if video:
            frames.append(env.render())
        if terminated or truncated:
            break
    step_phi = np.asarray(phis)
    np.testing.assert_allclose(step_phi.sum(axis=0), final_phi, atol=1e-10)
    return {
        "phi": final_phi, "step_phi": step_phi, "signals": np.asarray(signals),
        "observations": np.asarray(observations), "actions": np.asarray(actions),
        "length": len(phis), "terminated": bool(terminated), "frames": frames,
    }
