import numpy as np

from envs.base import Trajectory

STATE_KEYS = (
    "dz_com", "dtheta", "dz_us_f", "dz_us_r", "dx_com",
    "z_com", "theta", "z_us_f", "z_us_r", "x_com",
)
ACCEL_KEYS = ("ddz_com", "ddtheta", "ddz_us_f", "ddz_us_r", "ddx_com")
JERK_KEYS = ("dddz_com", "dddtheta", "dddz_us_f", "dddz_us_r", "dddx_com")
CHANNEL_KEYS = STATE_KEYS + ACCEL_KEYS + JERK_KEYS + (
    "u_eride", "u_human", "env_reward", "bump_detected",
)


class VmcRollout:
    def __init__(self, make_env, make_policy, dt, max_steps):
        self.make_env = make_env
        self.make_policy = make_policy
        self.dt = float(dt)
        self.max_steps = int(max_steps)

    def rollout(self, params, n, seed):
        env = self.make_env(params)
        trajectories = []
        try:
            for episode in range(int(n)):
                trajectories.append(self._episode(env, params, seed + episode))
        finally:
            close = getattr(env, "close", None)
            if close is not None:
                close()
        return trajectories

    def _episode(self, env, params, seed):
        obs, _ = env.reset(seed=seed)
        policy = self.make_policy(params)
        reset = getattr(policy, "reset", None)
        if reset is not None:
            reset()

        buffers = {key: np.empty(self.max_steps, dtype=np.float32) for key in CHANNEL_KEYS}
        terminated = truncated = False
        size = 0

        while not (terminated or truncated) and size < self.max_steps:
            action = self._action(policy, obs)
            obs, reward, terminated, truncated, info = env.step(action)
            trace = info.get("trace")
            if trace is not None:
                count = min(len(next(iter(trace.values()))), self.max_steps - size)
                for key in CHANNEL_KEYS:
                    buffers[key][size:size + count] = trace[key][:count]
                size += count
            else:
                inner_infos = info.get("inner_infos") or [info]
                for inner in inner_infos:
                    if size == self.max_steps:
                        break
                    self._record(buffers, size, inner, reward)
                    size += 1

        channels = {key: value[:size].copy() for key, value in buffers.items()}
        return Trajectory(
            channels=channels,
            dt=self.dt,
            meta={
                "seed": seed,
                "params": dict(params),
                "horizon": self.max_steps,
                "terminated": bool(terminated),
            },
        )

    @staticmethod
    def _action(policy, obs):
        predict = getattr(policy, "predict", None)
        if predict is not None:
            action, _ = predict(obs, deterministic=True)
            return np.asarray(action, dtype=np.float32)
        return np.asarray(policy(obs), dtype=np.float32)

    @staticmethod
    def _record(buffers, index, info, fallback_reward):
        state = info.get("state", {})
        for key in STATE_KEYS + ACCEL_KEYS + JERK_KEYS:
            buffers[key][index] = state.get(key, 0.0)
        buffers["u_eride"][index] = info.get("u_eride", state.get("u_eride", 0.0))
        buffers["u_human"][index] = info.get("u_human", 0.0)
        buffers["env_reward"][index] = info.get("reward", fallback_reward)
        buffers["bump_detected"][index] = info.get("bump_detected", False)
