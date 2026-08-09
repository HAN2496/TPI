import numpy as np
import gymnasium
from numba import njit

from ..controller import HumanController
from .components import Bump, compile_vehicle_model
from ..configs import Environment_Parameters, Vehicle_Parameters


STATE_KEYS = ["dz_com", "dtheta", "dz_us_f", "dz_us_r", "dx_com", "z_com", "theta", "z_us_f", "z_us_r", "x_com"]
ACCEL_KEYS = ["ddz_com", "ddtheta", "ddz_us_f", "ddz_us_r", "ddx_com"]
JERK_KEYS = ["dddz_com", "dddtheta", "dddz_us_f", "dddz_us_r", "dddx_com"]


@njit(cache=True, fastmath=True)
def _integrate(state, acceleration, jerk, derivative, dt):
    for index in range(5):
        jerk[index] = (derivative[index] - acceleration[index]) / dt
        acceleration[index] = derivative[index]
    for index in range(10):
        state[index] += derivative[index] * dt


class SuspensionEnv(gymnasium.Env):
    def __init__(self, human_controller=None, vehicle=None, reward_fn=None, obs_keys=None, use_human=False):
        super().__init__()
        self.use_human = use_human
        self.obs_keys = obs_keys or ["theta", "dtheta", "ddtheta", "dx_com", "ddx_com", "dz_com"]
        self.action_space = gymnasium.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gymnasium.spaces.Box(-np.inf, np.inf, shape=(len(self.obs_keys),), dtype=np.float32)
        self.config = Environment_Parameters()
        self.vehicle_params = Vehicle_Parameters()
        self.bump = Bump()
        self.vehicle = compile_vehicle_model(self.vehicle_params) if vehicle is None else vehicle
        self.human_controller = HumanController() if human_controller is None else human_controller

        if reward_fn is None:
            from .rewards import create_weighted_reward
            reward_fn = create_weighted_reward()

        self.reward_fn = reward_fn
        self.dt = float(self.config.dt_inner)
        self.max_time = 10.0
        self.eride_duration = 3.0
        self.time = 0.0
        self.eride_time = 0.0
        self.bump_detected = False
        self.u_eride = 0.0
        self.u_human = 0.0
        self.state_vector = np.asarray(self.config.x0, dtype=np.float64).copy()
        self.initial_state = self.state_vector.copy()
        self.acceleration = np.zeros(5, dtype=np.float64)
        self.jerk = np.zeros(5, dtype=np.float64)
        self.disturbance = np.zeros(2, dtype=np.float64)
        self.state = dict.fromkeys(STATE_KEYS, 0.0)
        self.state_ddot = dict.fromkeys(ACCEL_KEYS, 0.0)
        self.state_dddot = dict.fromkeys(JERK_KEYS, 0.0)
        self.obs = np.empty(len(self.obs_keys), dtype=np.float32)
        self.obs_dict = dict.fromkeys(self.obs_keys, 0.0)
        self.bump_gain_front = self.vehicle_params.k_us_f / self.vehicle_params.m_us_f
        self.bump_gain_rear = self.vehicle_params.k_us_r / self.vehicle_params.m_us_r
        self._reset_state()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.time = 0.0
        self.eride_time = 0.0
        self.bump_detected = False
        self.u_eride = 0.0
        self.u_human = 0.0
        self.bump.set_rng(self.np_random)
        self.bump.reset()

        velocity = self.np_random.uniform(20 / 3.6, 40 / 3.6)
        self.human_controller.v_ref = velocity
        self.initial_state[4] = velocity

        pitch = self.initial_state[6]
        position = self.initial_state[9]
        front_position = position + self.vehicle_params.l_f * np.cos(pitch)
        self.bump.set_x(front_position + self.bump.bump_specs[0, 1] + 1e-3)
        self._reset_state()
        return self.obs.copy(), self._get_info(self.disturbance, 0.0, self.u_eride, self.u_human)

    def _reset_state(self):
        self.state_vector[:] = self.initial_state
        self.acceleration.fill(0.0)
        self.jerk.fill(0.0)
        self._sync_dicts()
        self._update_obs()
        self._update_disturbance()
        self.u_human = self.human_controller(self.obs_dict, self.state, self.state_vector) if self.use_human else 0.0
        derivative = self.vehicle(self.state_vector, self.u_human, self.disturbance)
        self.acceleration[:] = derivative[:5]
        self._sync_dicts()
        self._update_obs()

    def step(self, action, compute_reward=True, build_info=True):
        action_value = float(np.asarray(action).reshape(-1)[0]) if np.ndim(action) else float(action)
        self._update_disturbance()
        self.u_human = self.human_controller(self.obs_dict, self.state, self.state_vector) if self.use_human else 0.0

        if not self.bump_detected:
            self.detect_bump(self.disturbance)

        self.u_eride = action_value if self.bump_detected else 0.0
        derivative = self.vehicle(self.state_vector, self.u_human + self.u_eride, self.disturbance)
        _integrate(self.state_vector, self.acceleration, self.jerk, derivative, self.dt)
        self._sync_dicts()
        self._update_obs()

        reward = self._get_reward(self.u_eride) if compute_reward else 0.0

        if self.bump_detected:
            self.eride_time += self.dt
            if self.eride_time >= self.eride_duration:
                self.eride_time = 0.0
                self.bump_detected = False

        self.time += self.dt
        truncated = self.time >= self.max_time
        info = self._get_info(self.disturbance, reward, self.u_eride, self.u_human) if build_info else None
        return self.obs.copy(), reward, False, truncated, info

    def _sync_dicts(self):
        self.state.update(zip(STATE_KEYS, self.state_vector))
        self.state_ddot.update(zip(ACCEL_KEYS, self.acceleration))
        self.state_dddot.update(zip(JERK_KEYS, self.jerk))

    def _update_obs(self):
        for index, key in enumerate(self.obs_keys):
            if key == "u_eride":
                value = self.u_eride
            elif key == "kp":
                value = self.eride_controller.kp
            elif key == "q_dz":
                value = self.eride_controller.q_diag[0]
            elif key == "q_dtheta":
                value = self.eride_controller.q_diag[1]
            elif key == "q_theta":
                value = self.eride_controller.q_diag[6]
            elif key == "q_dx":
                value = self.eride_controller.q_diag[4]
            elif key in self.state:
                value = self.state[key]
            elif key in self.state_ddot:
                value = self.state_ddot[key]
            else:
                value = self.state_dddot[key]
            self.obs[index] = value
            self.obs_dict[key] = value

    def _update_disturbance(self):
        pitch = self.state_vector[6]
        position = self.state_vector[9]
        offset = np.cos(pitch)
        self.disturbance[0] = self.bump(position + self.vehicle_params.l_f * offset)
        self.disturbance[1] = self.bump(position - self.vehicle_params.l_r * offset)

    def _get_unified_state(self):
        state = self.state.copy()
        state.update(self.state_ddot)
        state.update(self.state_dddot)
        state["u_eride"] = self.u_eride
        return state

    def _get_info(self, disturbance, reward, u_eride, u_human):
        info = {
            "time": self.time,
            "bump_detected": self.bump_detected,
            "disturbance": disturbance.copy(),
            "state": self._get_unified_state(),
            "reward": reward,
            "u_eride": u_eride,
            "u_human": u_human,
        }
        info.update(self.reward_fn.reward_info)
        return info

    def _get_reward(self, u_eride):
        return self.reward_fn(self.state, self.state_ddot, self.state_dddot, u_eride)

    def detect_bump(self, disturbance):
        front_residual = abs(self.bump_gain_front * disturbance[0])
        rear_residual = abs(self.bump_gain_rear * disturbance[1])
        self.bump_detected = front_residual > 0.05 or rear_residual > 0.05

    def calculate_disturbance(self, state):
        pitch = state["theta"]
        position = state["x_com"]
        offset = np.cos(pitch)
        return np.array([
            self.bump(position + self.vehicle_params.l_f * offset),
            self.bump(position - self.vehicle_params.l_r * offset),
        ])


class ErideEnv(SuspensionEnv):
    def __init__(self, eride_controller, human_controller=None, vehicle=None, reward_fn=None, mode="pure",
                 action_dim=1, obs_keys=None, use_human=False, record_inner=False, schedule_dt=None):
        if mode not in ("schedule", "pure"):
            raise ValueError("mode must be schedule or pure")

        self.eride_controller = eride_controller
        self.mode = mode
        self.record_inner = record_inner
        super().__init__(human_controller, vehicle, reward_fn, obs_keys, use_human)

        self.observe_step = int(round(self.config.t_observe / self.config.dt_inner))
        self.current_step = 0

        control_ratio = self.eride_controller.control_dt / self.config.dt_inner
        self.ctrl_period_step = int(round(control_ratio))
        if self.ctrl_period_step < 1 or not np.isclose(self.ctrl_period_step * self.config.dt_inner, self.eride_controller.control_dt):
            raise ValueError("control_dt must be an integer multiple of dt_inner")

        self.schedule_dt = self.eride_controller.control_dt if schedule_dt is None else float(schedule_dt)
        schedule_ratio = self.schedule_dt / self.eride_controller.control_dt
        self.schedule_period_step = int(round(schedule_ratio))
        if self.schedule_period_step < 1 or not np.isclose(self.schedule_period_step * self.eride_controller.control_dt, self.schedule_dt):
            raise ValueError("schedule_dt must be an integer multiple of control_dt")

        if mode == "schedule":
            self.action_space = gymnasium.spaces.Box(-1.0, 1.0, shape=(action_dim,), dtype=np.float32)
        else:
            self.action_space = gymnasium.spaces.Box(0.0, 0.0, shape=(0,), dtype=np.float32)

    def reset(self, **kwargs):
        self.eride_controller.reset()
        self.reward_fn.reset()
        self.current_step = 0
        return super().reset(**kwargs)

    def step(self, action):
        if self.mode == "schedule":
            self.eride_controller.schedule(np.asarray(action, dtype=np.float32))

        batch_reward = getattr(self.reward_fn, "batch_from_unified_states", None)
        deferred_reward = callable(batch_reward)
        inner_states = [] if deferred_reward else None
        inner_infos = [] if self.record_inner else None
        total_reward = 0.0
        last_reward = 0.0
        controller_output = 0.0
        terminated = False
        truncated = False
        obs = self.obs.copy()

        for _ in range(self.schedule_period_step):
            controller_output = self.eride_controller(self.obs_dict, self.state, self.state_vector)

            for _ in range(self.ctrl_period_step):
                obs, reward, terminated, truncated, info = super().step(
                    controller_output,
                    compute_reward=not deferred_reward,
                    build_info=self.record_inner,
                )

                if deferred_reward:
                    inner_states.append(info["state"] if self.record_inner else self._get_unified_state())
                else:
                    total_reward += reward
                    last_reward = reward

                if self.record_inner:
                    inner_infos.append(info)

                self.current_step += 1
                if self.current_step >= self.observe_step:
                    truncated = True
                    break
                if terminated or truncated:
                    break

            if terminated or truncated:
                break

        if deferred_reward:
            rewards = batch_reward(inner_states)
            if len(rewards):
                total_reward = float(np.sum(rewards))
                last_reward = float(rewards[-1])

        info = self._get_info(self.disturbance, last_reward, self.u_eride, self.u_human)
        info["controller"] = self.eride_controller.kind
        info["controller_output"] = float(controller_output)
        info.update({f"ctrl_{key}": value for key, value in self.eride_controller.get_params().items()})

        if self.record_inner:
            info["inner_infos"] = inner_infos

        return obs, total_reward, terminated, truncated, info
