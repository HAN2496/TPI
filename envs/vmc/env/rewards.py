import numpy as np
import torch


STATE_KEYS = ["dz_com", "dtheta", "dz_us_f", "dz_us_r", "dx_com", "z_com", "theta", "z_us_f", "z_us_r", "x_com"]
ACCEL_KEYS = ["ddz_com", "ddtheta", "ddz_us_f", "ddz_us_r", "ddx_com"]
JERK_KEYS = ["dddz_com", "dddtheta", "dddz_us_f", "dddz_us_r", "dddx_com"]


class EnvReward:
    def __init__(self, base_reward_fn=None, learned_reward_fn=None, mix=None):
        self.base_reward_fn = base_reward_fn
        self.learned_reward_fn = learned_reward_fn
        self.mix = mix
        if self.base_reward_fn is None and (self.learned_reward_fn is None or self.mix is not None):
            self.base_reward_fn = create_weighted_reward()
        self.reward_info = {"reward_env": 0.0, "reward_model": 0.0, "reward_mixed": 0.0}

    def reset(self):
        if self.base_reward_fn is not None:
            self.base_reward_fn.reset()
        if self.learned_reward_fn is not None:
            self.learned_reward_fn.reset()
        self.reward_info = {"reward_env": 0.0, "reward_model": 0.0, "reward_mixed": 0.0}

    def __call__(self, state, state_ddot, state_dddot, u_eride):
        reward_env = 0.0
        reward_model = 0.0

        if self.learned_reward_fn is not None:
            reward_model = self.learned_reward_fn(state, state_ddot, state_dddot, u_eride)
            if self.mix is None:
                reward = reward_model
            else:
                reward_env = self.base_reward_fn(state, state_ddot, state_dddot, u_eride)
                reward = self.mix * reward_env + (1.0 - self.mix) * reward_model
        else:
            reward_env = self.base_reward_fn(state, state_ddot, state_dddot, u_eride)
            reward = reward_env

        self.reward_info = {
            "reward_env": float(reward_env),
            "reward_model": float(reward_model),
            "reward_mixed": float(reward),
        }
        return float(reward)

    def batch_from_unified_states(self, states):
        if not states:
            return np.empty(0, dtype=np.float32)

        reward_env = np.zeros(len(states), dtype=np.float64)
        reward_model = np.zeros(len(states), dtype=np.float64)

        if self.learned_reward_fn is not None:
            reward_model = np.asarray(self.learned_reward_fn.batch_from_unified_states(states), dtype=np.float64)

        if self.learned_reward_fn is None or self.mix is not None:
            batch_fn = getattr(self.base_reward_fn, "_batch_from_unified_states", None)
            if batch_fn is not None:
                reward_env = np.asarray(batch_fn(states), dtype=np.float64)
            else:
                for index, unified in enumerate(states):
                    state = {key: unified[key] for key in STATE_KEYS}
                    acceleration = {key: unified[key] for key in ACCEL_KEYS}
                    jerk = {key: unified[key] for key in JERK_KEYS}
                    reward_env[index] = self.base_reward_fn(state, acceleration, jerk, unified["u_eride"])

        if self.learned_reward_fn is None:
            rewards = reward_env
        elif self.mix is None:
            rewards = reward_model
        else:
            rewards = self.mix * reward_env + (1.0 - self.mix) * reward_model

        self.reward_info = {
            "reward_env": float(reward_env[-1]),
            "reward_model": float(reward_model[-1]),
            "reward_mixed": float(rewards[-1]),
        }
        return rewards


class BaseReward:
    def __init__(self, w_pitch=1.0, w_accel=0.0, w_control=0.0, w_bounce=0.0, w_exp=0.0,
                 w_exp_threshold=0.5, w_delta_u=0.0):
        self.w_pitch = float(w_pitch)
        self.w_accel = float(w_accel)
        self.w_control = float(w_control)
        self.w_bounce = float(w_bounce)
        self.w_exp = float(w_exp)
        self.w_exp_threshold = float(w_exp_threshold)
        self.w_delta_u = float(w_delta_u)
        self.prev_u = 0.0
        self.reward_info = {"reward_env": 0.0}

    def reset(self):
        self.prev_u = 0.0
        self.reward_info = {"reward_env": 0.0}

    def __call__(self, state, state_ddot, state_dddot, u_eride):
        reward = 0.0
        if self.w_pitch:
            reward -= self.w_pitch * state["dtheta"] ** 2
        if self.w_accel:
            reward -= self.w_accel * state_ddot["ddx_com"] ** 2
        if self.w_control:
            reward -= self.w_control * u_eride ** 2
        if self.w_bounce:
            reward -= self.w_bounce * state_ddot["ddz_com"] ** 2
        if self.w_exp:
            pitch_accel = state_ddot["ddtheta"]
            reward -= self.w_exp * (np.exp(max(0.0, self.w_exp_threshold - abs(pitch_accel))) - 1.0) * u_eride ** 2
        if self.w_delta_u:
            reward -= self.w_delta_u * (u_eride - self.prev_u) ** 2
        self.prev_u = float(u_eride)
        self.reward_info = {"reward_env": float(reward)}
        return float(reward)

    def _batch_from_unified_states(self, states):
        controls = np.fromiter((state["u_eride"] for state in states), dtype=np.float64, count=len(states))
        rewards = np.zeros(len(states), dtype=np.float64)

        if self.w_pitch:
            values = np.fromiter((state["dtheta"] for state in states), dtype=np.float64, count=len(states))
            rewards -= self.w_pitch * values ** 2
        if self.w_accel:
            values = np.fromiter((state["ddx_com"] for state in states), dtype=np.float64, count=len(states))
            rewards -= self.w_accel * values ** 2
        if self.w_control:
            rewards -= self.w_control * controls ** 2
        if self.w_bounce:
            values = np.fromiter((state["ddz_com"] for state in states), dtype=np.float64, count=len(states))
            rewards -= self.w_bounce * values ** 2
        if self.w_exp:
            values = np.fromiter((state["ddtheta"] for state in states), dtype=np.float64, count=len(states))
            rewards -= self.w_exp * (np.exp(np.maximum(0.0, self.w_exp_threshold - np.abs(values))) - 1.0) * controls ** 2
        if self.w_delta_u:
            previous = np.empty_like(controls)
            previous[0] = self.prev_u
            previous[1:] = controls[:-1]
            rewards -= self.w_delta_u * (controls - previous) ** 2

        self.prev_u = float(controls[-1])
        self.reward_info = {"reward_env": float(rewards[-1])}
        return rewards


def create_weighted_reward(w_pitch=1.0, w_accel=0.0, w_control=0.0, w_bounce=0.0, w_exp=0.0,
                           w_exp_threshold=0.5, w_delta_u=0.0):
    return BaseReward(w_pitch, w_accel, w_control, w_bounce, w_exp, w_exp_threshold, w_delta_u)


class LearnedReward:
    def __init__(self, reward_model, state_keys, base_reward_fn=None, device="cpu"):
        self.reward_model = reward_model
        self.state_keys = list(state_keys)
        self.base_reward_fn = base_reward_fn
        self.device = device
        self.reward_info = {"reward_model": 0.0}

    def reset(self):
        self.reward_info = {"reward_model": 0.0}

    def __call__(self, state, state_ddot, state_dddot, u_eride):
        unified = {**state, **state_ddot, **state_dddot, "u_eride": u_eride}
        return float(self.batch_from_unified_states([unified])[0])

    def batch_from_unified_states(self, states):
        features = np.asarray([[state[key] for key in self.state_keys] for state in states], dtype=np.float32)
        return self.batch_from_features(features)

    def batch_from_features(self, features):
        tensor = torch.as_tensor(features, dtype=torch.float32, device=self.device).unsqueeze(1)
        with torch.no_grad():
            rewards = self.reward_model(tensor).detach().cpu().numpy().reshape(-1)
        self.reward_info = {"reward_model": float(rewards[-1])}
        return rewards


class OracleReward:
    def __init__(self, oracle):
        from ..configs import Environment_Parameters
        config = Environment_Parameters()
        self.oracle = oracle
        steps = int(round(config.t_observe / config.dt_inner))
        self.episode_scale = steps if oracle.aggregation == "mean" else 1
        self.reward_info = {"reward_oracle": 0.0}

    def reset(self):
        self.oracle.reset()
        self.reward_info = {"reward_oracle": 0.0}

    def __call__(self, state, state_ddot, state_dddot, u_eride):
        unified = {**state, **state_ddot, **state_dddot, "u_eride": u_eride}
        reward = self.oracle.step_reward(unified)
        for term in self.oracle.episode_terms:
            reward -= self.oracle.weights[term.name] * term.step(unified) * self.episode_scale
        self.reward_info = {"reward_oracle": float(reward)}
        return float(reward)
