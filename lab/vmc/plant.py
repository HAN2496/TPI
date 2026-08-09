import gymnasium
import numpy as np


STATE_KEYS = (
    "dz_com", "dtheta", "dz_us_f", "dz_us_r", "dx_com",
    "z_com", "theta", "z_us_f", "z_us_r", "x_com",
)
ACCEL_KEYS = ("ddz_com", "ddtheta", "ddz_us_f", "ddz_us_r", "ddx_com")
JERK_KEYS = ("dddz_com", "dddtheta", "dddz_us_f", "dddz_us_r", "dddx_com")
TRACE_KEYS = STATE_KEYS + ACCEL_KEYS + JERK_KEYS + (
    "u_eride", "u_human", "env_reward", "bump_detected",
)
STATE_INDEX = {key: i for i, key in enumerate(STATE_KEYS)}
ACCEL_INDEX = {key: i for i, key in enumerate(ACCEL_KEYS)}
JERK_INDEX = {key: i for i, key in enumerate(JERK_KEYS)}


class Bump:
    def __init__(self, position=3.0, half_width=1.25, height=0.09):
        self.base = np.array([position, half_width, height], dtype=np.float64)
        self.spec = self.base.copy()
        self.rng = np.random.default_rng()

    def set_rng(self, rng):
        self.rng = rng

    def reset(self):
        self.spec = self.base.copy()
        self.spec[1] += self.rng.uniform(-0.75, 0.75)
        self.spec[2] += self.rng.uniform(-0.03, 0.03)

    def set_x(self, position):
        self.base[0] = position
        self.spec[0] = position

    def __call__(self, x):
        distance = float(x) - self.spec[0]
        if abs(distance) > self.spec[1]:
            return 0.0
        return float(self.spec[2] * (1.0 + np.cos(np.pi * distance / self.spec[1])) / 2.0)


class PitchReward:
    def __init__(self, weight=1.0):
        self.weight = float(weight)
        self.reward_info = {"reward_env": 0.0}

    def reset(self):
        self.reward_info = {"reward_env": 0.0}

    def __call__(self, state, state_ddot, state_dddot, u_eride):
        reward = -self.weight * float(state["dtheta"] ** 2)
        self.reward_info = {"reward_env": reward}
        return reward


def _defaults():
    from configs import Environment_Parameters, Vehicle_Parameters
    from envs.vmc.controller import HumanController
    from envs.vmc.env.components import compile_vehicle_model
    return Environment_Parameters(), Vehicle_Parameters(), HumanController, compile_vehicle_model


class SuspensionEnv(gymnasium.Env):
    def __init__(
        self,
        config=None,
        vehicle_params=None,
        vehicle=None,
        bump=None,
        human_controller=None,
        reward_fn=None,
        obs_keys=None,
        use_human=False,
        control_limit=145.0,
    ):
        super().__init__()
        defaults = None
        if config is None or vehicle_params is None or vehicle is None or human_controller is None:
            defaults = _defaults()
        if config is None:
            config = defaults[0]
        if vehicle_params is None:
            vehicle_params = defaults[1]
        if human_controller is None:
            human_controller = defaults[2]()
        if vehicle is None:
            vehicle = defaults[3](vehicle_params)

        self.config = config
        self.vehicle_params = vehicle_params
        self.vehicle = vehicle
        self.bump = bump or Bump()
        self.human_controller = human_controller
        self.reward_fn = reward_fn or PitchReward()
        self.use_human = bool(use_human)
        self.control_limit = float(control_limit)
        self.obs_keys = tuple(obs_keys or ("theta", "dtheta", "ddtheta", "dx_com", "ddx_com", "dz_com"))
        self.dt = float(config.dt_inner)
        self.observe_steps = max(1, int(round(float(config.t_observe) / self.dt)))
        self.max_steps = int(getattr(config, "max_steps", self.observe_steps))
        self.action_space = gymnasium.spaces.Box(-1.0, 1.0, (1,), np.float32)
        self.observation_space = gymnasium.spaces.Box(-np.inf, np.inf, (len(self.obs_keys),), np.float32)
        self._obs_source = tuple(self._source(key) for key in self.obs_keys)
        self.x = np.zeros(10, dtype=np.float64)
        self.accel = np.zeros(5, dtype=np.float64)
        self.jerk = np.zeros(5, dtype=np.float64)
        self.road = np.zeros(2, dtype=np.float64)
        self.zero_road = np.zeros(2, dtype=np.float64)
        self.obs = np.zeros(len(self.obs_keys), dtype=np.float32)
        self.time = 0.0
        self.steps = 0
        self.eride_time = 0.0
        self.eride_duration = 3.0
        self.bump_detected = False
        self.u_eride = 0.0

    def _source(self, key):
        if key in STATE_INDEX:
            return 0, STATE_INDEX[key]
        if key in ACCEL_INDEX:
            return 1, ACCEL_INDEX[key]
        if key in JERK_INDEX:
            return 2, JERK_INDEX[key]
        if key == "u_eride":
            return 3, 0
        raise KeyError(key)

    def _update_obs(self):
        for i, (group, index) in enumerate(self._obs_source):
            self.obs[i] = (self.x, self.accel, self.jerk)[group][index] if group < 3 else self.u_eride

    def _state_dict(self):
        state = dict(zip(STATE_KEYS, self.x.tolist()))
        state.update(zip(ACCEL_KEYS, self.accel.tolist()))
        state.update(zip(JERK_KEYS, self.jerk.tolist()))
        state["u_eride"] = self.u_eride
        return state

    def _obs_dict(self):
        return dict(zip(self.obs_keys, self.obs.tolist()))

    def _road(self):
        front = self.x[9] + self.vehicle_params.l_f * np.cos(self.x[6])
        rear = self.x[9] - self.vehicle_params.l_r * np.cos(self.x[6])
        self.road[0] = self.bump(front)
        self.road[1] = self.bump(rear)
        return self.road

    def _set_initial_accel(self):
        u = self.human_controller(self._obs_dict(), self._state_dict()) if self.use_human else 0.0
        dx = np.asarray(self.vehicle(self.x, u, self._road()), dtype=np.float64)
        self.accel[:] = dx[:5]
        self.jerk.fill(0.0)
        self._update_obs()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.x[:] = np.asarray(self.config.x0, dtype=np.float64)
        self.x[4] = self.np_random.uniform(20.0 / 3.6, 40.0 / 3.6)
        self.time = 0.0
        self.steps = 0
        self.eride_time = 0.0
        self.bump_detected = False
        self.u_eride = 0.0
        self.bump.set_rng(self.np_random)
        self.bump.reset()
        if hasattr(self.human_controller, "v_ref"):
            self.human_controller.v_ref = self.x[4]
        front = self.x[9] + self.vehicle_params.l_f * np.cos(self.x[6])
        self.bump.set_x(front + self.bump.spec[1] + 1e-3)
        self._update_obs()
        self._set_initial_accel()
        self.reward_fn.reset()
        return self.obs.copy(), self._info(np.zeros(2), 0.0, 0.0)

    def _detect_bump(self, u, road):
        nominal = self.vehicle(self.x, u, self.zero_road)
        front = float(nominal[2])
        rear = float(nominal[3])
        disturbed = self.vehicle(self.x, u, road)
        self.bump_detected = abs(front - disturbed[2]) > 0.05 or abs(rear - disturbed[3]) > 0.05

    def step(self, action, compute_reward=True):
        action = float(np.asarray(action, dtype=np.float64).reshape(-1)[0])
        state = self._state_dict() if self.use_human else None
        obs = self._obs_dict() if self.use_human else None
        u_human = self.human_controller(obs, state) if self.use_human else 0.0
        road = self._road()
        if not self.bump_detected:
            self._detect_bump(u_human, road)
            if self.bump_detected:
                self.eride_time = 0.0
        self.u_eride = action if self.bump_detected else 0.0
        u = u_human + self.u_eride
        dx = np.asarray(self.vehicle(self.x, u, road), dtype=np.float64)
        self.x += dx * self.dt
        self.jerk[:] = (dx[:5] - self.accel) / self.dt
        self.accel[:] = dx[:5]
        self._update_obs()
        reward = self._get_reward() if compute_reward else 0.0
        self.steps += 1
        self.time += self.dt
        if self.bump_detected:
            self.eride_time += self.dt
            if self.eride_time >= self.eride_duration:
                self.bump_detected = False
                self.eride_time = 0.0
        truncated = self.steps >= self.max_steps
        return self.obs.copy(), reward, False, truncated, self._info(road, reward, u_human)

    def _get_reward(self):
        return float(self.reward_fn(self._state_dict(), dict(zip(ACCEL_KEYS, self.accel)), dict(zip(JERK_KEYS, self.jerk)), self.u_eride))

    def _info(self, road, reward, u_human):
        info = {
            "time": self.time,
            "bump_detected": self.bump_detected,
            "disturbance": np.asarray(road, dtype=np.float32).copy(),
            "reward": float(reward),
            "u_eride": float(self.u_eride),
            "u_human": float(u_human),
        }
        info.update(self.reward_fn.reward_info)
        return info

class ErideEnv(SuspensionEnv):
    def __init__(self, eride_controller, mode="schedule", action_dim=1, control_dt=None, **kwargs):
        super().__init__(**kwargs)
        if mode not in {"schedule", "pure"}:
            raise ValueError(mode)
        self.eride_controller = eride_controller
        self.mode = mode
        self.control_dt = float(control_dt or eride_controller.control_dt)
        self.ctrl_period = max(1, int(round(self.control_dt / self.dt)))
        if not np.isclose(self.ctrl_period * self.dt, self.control_dt):
            raise ValueError("control_dt must be an integer multiple of dt_inner")
        self.action_space = gymnasium.spaces.Box(-1.0, 1.0, (action_dim,), np.float32) if mode == "schedule" else gymnasium.spaces.Box(0.0, 0.0, (0,), np.float32)
        self.current_step = 0

    def reset(self, **kwargs):
        self.eride_controller.reset()
        self.current_step = 0
        return super().reset(**kwargs)

    def step(self, action):
        if self.mode == "schedule":
            self.eride_controller.schedule(np.asarray(action, dtype=np.float32))
        u_eride = float(self.eride_controller(self._obs_dict(), self._state_dict()))
        traces = {key: np.empty(self.ctrl_period, dtype=np.float32) for key in TRACE_KEYS}
        total_reward = 0.0
        terminated = truncated = False
        size = 0
        for _ in range(self.ctrl_period):
            _, reward, terminated, truncated, info = super().step([u_eride])
            total_reward += reward
            values = np.concatenate((self.x, self.accel, self.jerk))
            for key, value in zip(STATE_KEYS + ACCEL_KEYS + JERK_KEYS, values):
                traces[key][size] = value
            traces["u_eride"][size] = self.u_eride
            traces["u_human"][size] = info["u_human"]
            traces["env_reward"][size] = reward
            traces["bump_detected"][size] = self.bump_detected
            size += 1
            self.current_step += 1
            if self.current_step >= self.observe_steps:
                truncated = True
                break
            if terminated or truncated:
                break
        info["trace"] = {key: value[:size].copy() for key, value in traces.items()}
        info["controller"] = self.eride_controller.kind
        info.update({f"ctrl_{key}": value for key, value in self.eride_controller.get_params().items()})
        return self.obs.copy(), total_reward, terminated, truncated, info
