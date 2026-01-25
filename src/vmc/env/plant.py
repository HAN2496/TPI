import numpy as np
import gymnasium

from src.vmc.controller import HumanController
from src.vmc.env.components import Bump, compile_vehicle_model
from src.vmc.configs import Environment_Parameters, Vehicle_Parameters

STATE_KEYS = ["dz_com", "dtheta", "dz_us_f", "dz_us_r", "dx_com", 
              "z_com", "theta", "z_us_f", "z_us_r", "x_com"]
ACCEL_KEYS = ["ddz_com", "ddtheta", "ddz_us_f", "ddz_us_r", "ddx_com"]
JERK_KEYS = ["dddz_com", "dddtheta", "dddz_us_f", "dddz_us_r", "dddx_com"]

class SuspensionEnv(gymnasium.Env):
    def __init__(self, human_controller=None, vehicle=None, reward_fn=None, obs_keys=None):
        super().__init__()

        self.obs_keys = obs_keys if obs_keys is not None else ["theta", "dtheta", "ddtheta", "dx_com", "ddx_com", "dz_com"]

        self.action_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.obs_keys),), dtype=np.float32)

        self.config = Environment_Parameters()
        self.vehicle_params = Vehicle_Parameters()
        self.bump = Bump()
        self.vehicle = compile_vehicle_model(self.vehicle_params) if vehicle is None else vehicle

        self.human_controller = human_controller if human_controller is not None else HumanController()

        if reward_fn is None:
            from src.vmc.env.rewards import create_weighted_reward
            self.reward_fn = create_weighted_reward()
        else:
            self.reward_fn = reward_fn

        self.time = 0.0
        self.max_time = 10.0

        self.eride_time = 0.0
        self.eride_decay = 1.0
        self.eride_duration = 3.0
        self.bump_detected = False
        self.u_eride = 0.0

        self.x0 = np.array(self.config.x0, dtype=np.float32)
        self._reset_state()
        self.dt = self.config.dt_inner

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.time = 0.0
        self.eride_time = 0.0
        self.bump_detected = False
        self.u_eride = 0.0

        self.bump.set_rng(self.np_random)
        self.bump.reset()

        if hasattr(self.reward_fn, 'reset'):
            self.reward_fn.reset()

        v_ref = self.np_random.uniform(20/3.6, 40/3.6)
        self.human_controller.v_ref = v_ref
        self.x0[STATE_KEYS.index("dx_com")] = v_ref
        self._reset_state()

        theta0 = float(self.state["theta"])
        x_front = (float(self.state["x_com"]) + self.vehicle_params.l_f * np.cos(theta0))
        hw = float(self.bump.bump_specs[0, 1])
        new_pos = x_front + hw + 1e-3
        self.bump.set_x(new_pos)

        info = self._get_info(z=np.zeros(2, dtype=np.float32), reward=0.0, u_eride=0.0, u_human=0.0)
        return self.obs.copy(), info

    def _reset_state(self):
        self.state = {key: 0.0 for key in STATE_KEYS}
        for i, key in enumerate(STATE_KEYS):
            self.state[key] = self.x0[i]

        self.state_ddot = {key: 0.0 for key in ACCEL_KEYS}
        self.state_dddot = {key: 0.0 for key in JERK_KEYS}
        self.obs, self.obs_dict = self._get_obs()

        x = np.array([self.state[key] for key in STATE_KEYS], dtype=np.float32)
        z = self.calculate_disturbance(self.state.copy())
        u_human = self.human_controller(self.obs_dict, self.state)
        dx = np.asarray(self.vehicle(x, u_human, z), dtype=np.float32)

        for i, key in enumerate(ACCEL_KEYS):
            self.state_ddot[key] = dx[i]

        self.state_dddot = {key: 0.0 for key in JERK_KEYS}

        self.obs, self.obs_dict = self._get_obs()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(1)
        # 1. State, disturbance
        x = np.array([self.state[key] for key in STATE_KEYS], dtype=np.float32)
        z = self.calculate_disturbance(self.state.copy())

        # 2. Calculate Human Control input
        u_human = self.human_controller(self.obs_dict, self.state)

        # 3. Bump detection
        if not self.bump_detected:
            self.detect_bump(x, u_human, z)
            if self.bump_detected:
                self.eride_time = 0.0

        # 4. Calculate total control input
        self.u_eride = 0.0
        if self.bump_detected:
            self.u_eride = float(action[0])
        u = u_human + self.u_eride

        # 5. Vehicle dynamics
        dx = np.asarray(self.vehicle(x, u, z), dtype=np.float32)
        x_next = x + dx * self.dt

        # 6. Update
        for i, key in enumerate(STATE_KEYS):
            self.state[key] = x_next[i]

        prev_ddot = self.state_ddot.copy()
        for i, key in enumerate(ACCEL_KEYS):
            self.state_ddot[key] = dx[i]
        for i, key in enumerate(JERK_KEYS):
            accel_key = ACCEL_KEYS[i]
            self.state_dddot[key] = (self.state_ddot[accel_key] - prev_ddot[accel_key]) / self.dt

        self.obs, self.obs_dict = self._get_obs()

        # 7. Calculate reward
        reward = self._get_reward(self.u_eride)

        # 8. Update time
        if self.bump_detected:
            self.eride_time += self.dt
            if self.eride_time >= self.eride_duration:
                self.eride_time = 0.0
                self.bump_detected = False

        self.time += self.dt

        truncated = True if self.time >= self.max_time else False
        if truncated:
            if hasattr(self.reward_fn, 'get_episode_reward'):
                reward += self.reward_fn.get_episode_reward()
            print(f"time: {self.time:.2f}")
            print(f"eride time: {self.eride_time:.2f}")
            print(f"Bump detected: {self.bump_detected}")
        info = self._get_info(z, reward, self.u_eride, u_human)
        return self.obs.copy(), reward, False, truncated, info

    def _get_obs(self):
        obs_values = []
        obs_dict = {}
        for key in self.obs_keys:
            if key == 'u_eride':
                value = self.u_eride
            elif key in self.state:
                value = self.state[key]
            elif key in self.state_ddot:
                value = self.state_ddot[key]
            elif key in self.state_dddot:
                value = self.state_dddot[key]
            else:
                raise KeyError(f"Key '{key}' not found in state, state_ddot, or state_dddot.")
            obs_values.append(value)
            obs_dict[key] = value
        obs_array = np.array(obs_values, dtype=np.float32)
        return obs_array, obs_dict

    def _get_info(self, z, reward, u_eride, u_human):
        unified_state = {}
        unified_state.update(self.state)
        unified_state.update(self.state_ddot)
        unified_state.update(self.state_dddot)
        unified_state['u_eride'] = u_eride
        info = {
            "time": self.time, "bump_detected": self.bump_detected, "disturbance": z,
            "state": unified_state, "reward": reward, "u_eride": u_eride, "u_human": u_human
        }
        info.update(self.reward_fn.reward_info)
        return info

    def _get_reward(self, u_eride):
        return self.reward_fn(self.state, self.state_ddot, self.state_dddot, u_eride)

    def detect_bump(self, x, u, z):
        x_pred = self.vehicle(x, u, z=[0, 0]).copy()
        x_real = self.vehicle(x, u, z=z)
        residual1 = np.abs(x_pred[2] - x_real[2])  # dz_us_f
        residual2 = np.abs(x_pred[3] - x_real[3])  # dz_us_r
        self.bump_detected = residual1 > 0.05 or residual2 > 0.05

    def calculate_disturbance(self, state):
        x_front = state["x_com"] + self.vehicle_params.l_f * np.cos(state["theta"])
        x_rear = state["x_com"] - self.vehicle_params.l_r * np.cos(state["theta"])
        return np.array([self.bump(x_front), self.bump(x_rear)], dtype=np.float32)

class ErideEnv(SuspensionEnv):
    def __init__(self, eride_controller, human_controller=None, vehicle=None, reward_fn=None, mode="pure", action_dim=1, obs_keys=None):
        assert mode in ("schedule", "pure")
        super().__init__(human_controller=human_controller, vehicle=vehicle, reward_fn=reward_fn, obs_keys=obs_keys)

        self.mode = mode
        self.eride_controller = eride_controller

        self.t_observe    = self.config.t_observe
        self.observe_step = int(self.t_observe / self.config.dt_inner)
        self.current_step = 0

        ratio = self.config.dt_outer / self.config.dt_inner
        self.ctrl_period_step = int(round(ratio))
        assert self.ctrl_period_step >= 1
        assert abs(self.ctrl_period_step * self.config.dt_inner - self.config.dt_outer) < 1e-6, \
            "dt_outer must be an integer multiple of dt_inner"

        self.action_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32) if mode=="schedule" \
                            else gymnasium.spaces.Box(low=0.0, high=0.0, shape=(0,), dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.eride_controller.reset()
        self.current_step = 0
        return obs, info

    def step(self, action):
        if self.mode == "schedule":
            self.eride_controller.schedule(np.asarray(action, dtype=np.float32))
        u_eride = self.eride_controller(self.obs_dict, self.state)

        if self.ctrl_period_step <= 1:
            obs, r, term, trunc, info = super().step(np.array([u_eride], dtype=np.float32))
            self.current_step += 1
            if self.current_step >= self.observe_step:
                if not (term or trunc):
                    trunc = True # Use truncated for time limit
                    if hasattr(self.reward_fn, 'get_episode_reward'):
                        r += self.reward_fn.get_episode_reward()
            return obs, r, term, trunc, info

        total_reward = 0.0
        terminated = False
        truncated = False
        last_info = None
        obs = self.obs.copy()
        for _ in range(self.ctrl_period_step):
            obs, r, term, trunc, info = super().step(np.array([u_eride], dtype=np.float32))

            total_reward += r

            last_info = info
            self.current_step += 1
            if self.current_step >= self.observe_step:
                truncated = True # Changed to truncated to be consistent
                if hasattr(self.reward_fn, 'get_episode_reward'):
                    total_reward += self.reward_fn.get_episode_reward()
                break

            if term or trunc:
                terminated, truncated = term, trunc
                break
        last_info = dict(last_info)

        last_info.update({"controller": self.eride_controller.kind, "u_eride": float(u_eride)})
        last_info.update({f"ctrl_{k}": v for k, v in self.eride_controller.get_params().items()})
        return obs, total_reward, terminated, truncated, last_info