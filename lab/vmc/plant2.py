import math
import numpy as np
from numba import njit
from configs import Vehicle_Parameters

import numpy as np
from numba import njit
import math

from configs import Vehicle_Parameters

@njit(cache=True, fastmath=True)
def _compute_dynamics(x_data, u, z_data, params_data, out_data):
    # Unpack constants
    k_s_f, k_s_r          = params_data[0], params_data[1]
    c_s_f, c_s_r          = params_data[2], params_data[3] 
    k_us_f, k_us_r        = params_data[4], params_data[5]
    l_f, l_r              = params_data[6], params_data[7]
    rolling_f, rolling_r  = params_data[8], params_data[9] # precompute
    m_tot, m_s            = params_data[10], params_data[11]
    m_us_f, m_us_r        = params_data[12], params_data[13]
    I                     = params_data[14]
    drive_gain            = params_data[15]
    air_drag_gain         = params_data[16]
    traction_limit        = params_data[17]
    h_f, h_r              = params_data[18], params_data[19]
    tan_phi_f, tan_phi_r  = params_data[20], params_data[21]

    # state variables
    dz_com, dtheta, dz_us_f, dz_us_r, dx_com = x_data[0:5]
    z_com, theta, z_us_f, z_us_r, x_com      = x_data[5:10]

    # Precompute
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    sign_dx_com = 1.0 if dx_com > 0 else (-1.0 if dx_com < 0 else 0.0)


    # Suspension
    del_f = l_f * sin_theta + z_com - z_us_f
    del_r = -l_r * sin_theta + z_com - z_us_r
    ddel_f = l_f * dtheta * cos_theta + dz_com - dz_us_f
    ddel_r = -l_r * dtheta * cos_theta + dz_com - dz_us_r

    spring_force_f = k_s_f * del_f
    spring_force_r = k_s_r * del_r
    damper_force_f = c_s_f * ddel_f
    damper_force_r = c_s_r * ddel_r

    # Unsprung force
    unsprung_force_f = k_us_f * (z_us_f - z_data[0])
    unsprung_force_r = k_us_r * (z_us_r - z_data[1])

    traction = min(max(drive_gain * u, -traction_limit), traction_limit)
    air_drag = air_drag_gain * dx_com * dx_com * sign_dx_com
    rolling_force_f = rolling_f * sign_dx_com
    rolling_force_r = rolling_r * sign_dx_com

    # Tire forces
    tire_long_f = -rolling_force_f
    tire_long_r = traction - rolling_force_r
    tire_vert_f = -tire_long_f * tan_phi_f
    tire_vert_r = tire_long_r * tan_phi_r

    # Pitch moment
    pitch_moment_extra = (tire_long_f * tan_phi_f * l_f +
                          tire_long_r * tan_phi_r * l_r) * cos_theta

    # Compute accelerations
    out_data[0] = (-(spring_force_f + spring_force_r + damper_force_f + damper_force_r) + 
                   (tire_vert_f + tire_vert_r)) / m_s

    out_data[1] = (-(spring_force_f + damper_force_f) * l_f * cos_theta + 
                   (spring_force_r + damper_force_r) * l_r * cos_theta + 
                   traction * h_r - rolling_force_f * h_f - rolling_force_r * h_r + 
                   pitch_moment_extra) / I

    out_data[2] = (spring_force_f + damper_force_f - unsprung_force_f - tire_vert_f) / m_us_f
    out_data[3] = (spring_force_r + damper_force_r - unsprung_force_r - tire_vert_r) / m_us_r
    out_data[4] = (traction - air_drag - rolling_force_f - rolling_force_r) / m_tot

    # velocities
    out_data[5] = dz_com
    out_data[6] = dtheta
    out_data[7] = dz_us_f  
    out_data[8] = dz_us_r
    out_data[9] = dx_com


class VehicleModel:
    __slots__ = ("params", "derivative")
    def __init__(self, params=None):
        if params is None:
            params = Vehicle_Parameters()

        mass_sprung = params.m_tot - params.m_us_f - params.m_us_r
        wheelbase = params.l_f + params.l_r
        normal_front = mass_sprung * 9.81 * params.l_r / wheelbase + params.m_us_f * 9.81
        normal_rear = mass_sprung * 9.81 * params.l_f / wheelbase + params.m_us_r * 9.81
        self.params = np.array([
            params.k_s_f, params.k_s_r, params.c_s_f, params.c_s_r,
            params.k_us_f, params.k_us_r,
            params.l_f, params.l_r,
            params.C_r_f * normal_front, params.C_r_r * normal_rear,
            params.m_tot, mass_sprung, params.m_us_f, params.m_us_r, params.I,
            params.eta_drive * params.r_transmission / params.r_wheel,
            0.5 * params.rho_air * params.A * params.C_d,
            params.mu * normal_rear,
            params.h_f, params.h_r, np.tan(params.phi_f), np.tan(params.phi_r)
            ], dtype=np.float64)
        self.derivative = np.empty(10, dtype=np.float64)

    def __call__(self, x, u, z):
        return self.system_dynamics(x, u, z)

    def system_dynamics(self, state, control, road):
        _compute_dynamics(state, float(control), road, self.params, self.derivative)
        return self.derivative

    def get_ABmatrix(self, xeq, ueq, zeq):
        def f(x,u):
            dX = self.system_dynamics(x, np.asarray(u).item(), zeq)
            return np.array(dX)

        n = 10; m = 1; eps = 1e-1

        # A 
        A = np.zeros((n,n))
        for i in range(n):
            dx = np.zeros(n); dx[i] = eps
            A[:,i] = (f(xeq+dx, ueq) - f(xeq-dx, ueq)) / (2*eps)

        # B
        B = np.zeros((n,m))
        for j in range(m):
            du = np.zeros(m); du[j] = eps
            B[:,j] = (f(xeq, ueq+du) - f(xeq, ueq-du)) / (2*eps)
        return A, B


# Warm-up compilation function
def compile_vehicle_model(params=None):
    model = VehicleModel(params=params)
    test_state = np.zeros(10)
    test_state[4] = 10.0  # velocity
    test_control = 0.0
    test_road = np.zeros(2)

    # Trigger JIT compilation
    model(test_state, test_control, test_road)
    return model


import math
import numpy as np
from numba import njit
from configs import Vehicle_Parameters

@njit(cache=True, fastmath=True)
def _compute_dynamics(x_data, u, z_data, params_data, out_data):
    k_s_f, k_s_r = params_data[0], params_data[1]
    c_s_f, c_s_r = params_data[2], params_data[3]
    k_us_f, k_us_r = params_data[4], params_data[5]
    l_f, l_r = params_data[6], params_data[7]
    rolling_f, rolling_r = params_data[8], params_data[9]
    m_tot, m_s = params_data[10], params_data[11]
    m_us_f, m_us_r = params_data[12], params_data[13]
    I = params_data[14]
    drive_gain = params_data[15]
    air_drag_gain = params_data[16]
    traction_limit = params_data[17]
    h_f, h_r = params_data[18], params_data[19]
    tan_phi_f, tan_phi_r = params_data[20], params_data[21]
    dz_com, dtheta, dz_us_f, dz_us_r, dx_com = x_data[:5]
    z_com, theta, z_us_f, z_us_r, x_com = x_data[5:]
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    sign_dx_com = 1.0 if dx_com > 0 else (-1.0 if dx_com < 0 else 0.0)
    del_f = l_f * sin_theta + z_com - z_us_f
    del_r = -l_r * sin_theta + z_com - z_us_r
    ddel_f = l_f * dtheta * cos_theta + dz_com - dz_us_f
    ddel_r = -l_r * dtheta * cos_theta + dz_com - dz_us_r
    spring_force_f = k_s_f * del_f
    spring_force_r = k_s_r * del_r
    damper_force_f = c_s_f * ddel_f
    damper_force_r = c_s_r * ddel_r
    unsprung_force_f = k_us_f * (z_us_f - z_data[0])
    unsprung_force_r = k_us_r * (z_us_r - z_data[1])
    traction = min(max(drive_gain * u, -traction_limit), traction_limit)
    air_drag = air_drag_gain * dx_com * dx_com * sign_dx_com
    rolling_force_f = rolling_f * sign_dx_com
    rolling_force_r = rolling_r * sign_dx_com
    tire_long_f = -rolling_force_f
    tire_long_r = traction - rolling_force_r
    tire_vert_f = -tire_long_f * tan_phi_f
    tire_vert_r = tire_long_r * tan_phi_r
    pitch_moment_extra = (tire_long_f * tan_phi_f * l_f + tire_long_r * tan_phi_r * l_r) * cos_theta
    out_data[0] = (-(spring_force_f + spring_force_r + damper_force_f + damper_force_r) + tire_vert_f + tire_vert_r) / m_s
    out_data[1] = (-(spring_force_f + damper_force_f) * l_f * cos_theta + (spring_force_r + damper_force_r) * l_r * cos_theta + traction * h_r - rolling_force_f * h_f - rolling_force_r * h_r + pitch_moment_extra) / I
    out_data[2] = (spring_force_f + damper_force_f - unsprung_force_f - tire_vert_f) / m_us_f
    out_data[3] = (spring_force_r + damper_force_r - unsprung_force_r - tire_vert_r) / m_us_r
    out_data[4] = (traction - air_drag - rolling_force_f - rolling_force_r) / m_tot
    out_data[5] = dz_com
    out_data[6] = dtheta
    out_data[7] = dz_us_f
    out_data[8] = dz_us_r
    out_data[9] = dx_com


class VehicleModel:
    __slots__ = ("params", "derivative")

    def __init__(self, params=None):
        params = Vehicle_Parameters() if params is None else params
        mass_sprung = params.m_tot - params.m_us_f - params.m_us_r
        wheelbase = params.l_f + params.l_r
        normal_front = mass_sprung * 9.81 * params.l_r / wheelbase + params.m_us_f * 9.81
        normal_rear = mass_sprung * 9.81 * params.l_f / wheelbase + params.m_us_r * 9.81
        self.params = np.array([
            params.k_s_f, params.k_s_r,
            params.c_s_f, params.c_s_r,
            params.k_us_f, params.k_us_r,
            params.l_f, params.l_r,
            params.C_r_f * normal_front, params.C_r_r * normal_rear,
            params.m_tot, mass_sprung,
            params.m_us_f, params.m_us_r, params.I, params.eta_drive * params.r_transmission / params.r_wheel, 0.5 * params.rho_air * params.A * params.C_d, params.mu * normal_rear, params.h_f, params.h_r, np.tan(params.phi_f), np.tan(params.phi_r)], dtype=np.float64)
        self.derivative = np.empty(10, dtype=np.float64)

    def __call__(self, state, control, road):
        return self.system_dynamics(state, control, road)

    def system_dynamics(self, state, control, road):
        _compute_dynamics(state, float(control), road, self.params, self.derivative)
        return self.derivative

    def get_ABmatrix(self, state_eq, control_eq, road_eq):
        state_eq = np.asarray(state_eq, dtype=np.float64)
        road_eq = np.asarray(road_eq, dtype=np.float64)
        control_eq = float(np.asarray(control_eq).item())
        epsilon = 1e-1

        def dynamics(state, control):
            return self.system_dynamics(state, control, road_eq).copy()
        A = np.empty((10, 10))
        for index in range(10):
            delta = np.zeros(10)
            delta[index] = epsilon
            A[:, index] = ( dynamics(state_eq + delta, control_eq) - dynamics(state_eq - delta, control_eq) ) / (2 * epsilon)
        B = ( dynamics(state_eq, control_eq + epsilon) - dynamics(state_eq, control_eq - epsilon) )[:, None] / (2 * epsilon)
        return A, B

def compile_vehicle_model(params=None):
    model = VehicleModel(params)
    state = np.zeros(10)
    state[4] = 10.0
    model(state, 0.0, np.zeros(2))
    return model
import math
import numpy as np


class Bump:
    __slots__ = ("base_specs", "bump_specs", "rng")

    def __init__(self):
        self.base_specs = np.array([[3.0, 1.25, 0.09]])
        self.bump_specs = self.base_specs.copy()

    def set_rng(self, rng):
        self.rng = rng

    def reset(self):
        self.bump_specs[:] = self.base_specs
        self.bump_specs[0, 1] += self.rng.uniform(-0.75, 0.75)
        self.bump_specs[0, 2] += self.rng.uniform(-0.03, 0.03)

    def __call__(self, position):
        bump_position, half_width, height = self.bump_specs[0]
        distance = position - bump_position
        if abs(distance) > half_width:
            return 0.0
        return height * ( 1.0 + math.cos(math.pi * distance / half_width) ) * 0.5

    def set_x(self, position):
        self.base_specs[0, 0] = position
        self.bump_specs[0, 0] = position
import numpy as np
import gymnasium
from numba import njit
from src.controller import HumanController
from src.env.components import Bump, compile_vehicle_model
from configs import Environment_Parameters, Vehicle_Parameters
STATE_KEYS = [ "dz_com", "dtheta", "dz_us_f", "dz_us_r", "dx_com", "z_com", "theta", "z_us_f", "z_us_r", "x_com" ]
ACCEL_KEYS = [ "ddz_com", "ddtheta", "ddz_us_f", "ddz_us_r", "ddx_com" ]
JERK_KEYS = [ "dddz_com", "dddtheta", "dddz_us_f", "dddz_us_r", "dddx_com" ]

@njit(cache=True, fastmath=True)
def _integrate(state, acceleration, jerk, derivative, dt):
    for index in range(5):
        jerk[index] = (derivative[index] - acceleration[index]) / dt
        acceleration[index] = derivative[index]
    for index in range(10):
        state[index] += derivative[index] * dt


class SuspensionEnv(gymnasium.Env):

    def __init__( self, human_controller=None, vehicle=None, reward_fn=None, obs_keys=None, use_human=False, ):
        super().__init__()
        self.use_human = use_human
        self.obs_keys = obs_keys or [ "theta", "dtheta", "ddtheta", "dx_com", "ddx_com", "dz_com", ]
        self.action_space = gymnasium.spaces.Box( -1.0, 1.0, shape=(1,), dtype=np.float32, )
        self.observation_space = gymnasium.spaces.Box( -np.inf, np.inf, shape=(len(self.obs_keys),), dtype=np.float32, )
        self.config = Environment_Parameters()
        self.vehicle_params = Vehicle_Parameters()
        self.bump = Bump()
        self.vehicle = ( compile_vehicle_model(self.vehicle_params) if vehicle is None else vehicle )
        self.human_controller = ( HumanController() if human_controller is None else human_controller )
        if reward_fn is None:
            from src.env.rewards import create_weighted_reward
            reward_fn = create_weighted_reward()
        self.reward_fn = reward_fn
        self.dt = self.config.dt_inner
        self.max_time = 10.0
        self.eride_duration = 3.0
        self.time = 0.0
        self.eride_time = 0.0
        self.bump_detected = False
        self.u_eride = 0.0
        self.u_human = 0.0
        self.state_vector = np.asarray( self.config.x0, dtype=np.float64, ).copy()
        self.initial_state = self.state_vector.copy()
        self.acceleration = np.zeros(5)
        self.jerk = np.zeros(5)
        self.disturbance = np.zeros(2)
        self.state = dict.fromkeys(STATE_KEYS, 0.0)
        self.state_ddot = dict.fromkeys(ACCEL_KEYS, 0.0)
        self.state_dddot = dict.fromkeys(JERK_KEYS, 0.0)
        self.obs = np.empty(len(self.obs_keys), dtype=np.float32)
        self.obs_dict = dict.fromkeys(self.obs_keys, 0.0)
        self.bump_gain_front = ( self.vehicle_params.k_us_f / self.vehicle_params.m_us_f )
        self.bump_gain_rear = ( self.vehicle_params.k_us_r / self.vehicle_params.m_us_r )
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
        front_position = ( position + self.vehicle_params.l_f * np.cos(pitch) )
        bump_position = ( front_position + self.bump.bump_specs[0, 1] + 1e-3 )
        self.bump.set_x(bump_position)
        self._reset_state()
        info = self._get_info( self.disturbance, 0.0, 0.0, self.u_human, )
        return self.obs.copy(), info

    def _reset_state(self):
        self.state_vector[:] = self.initial_state
        self.acceleration.fill(0.0)
        self.jerk.fill(0.0)
        self._sync_dicts()
        self._update_obs()
        self._update_disturbance()
        self.u_human = ( self.human_controller(self.obs_dict, self.state) if self.use_human else 0.0 )
        derivative = self.vehicle( self.state_vector, self.u_human, self.disturbance, )
        self.acceleration[:] = derivative[:5]
        self._sync_dicts()
        self._update_obs()

    def step(self, action, compute_reward=True, build_info=True):
        action_value = ( float(action[0]) if isinstance(action, np.ndarray) else float(action) )
        self._update_disturbance()
        self.u_human = ( self.human_controller(self.obs_dict, self.state) if self.use_human else 0.0 )
        if not self.bump_detected:
            self.detect_bump(self.disturbance)
        self.u_eride = ( action_value if self.bump_detected else 0.0 )
        control = self.u_human + self.u_eride
        derivative = self.vehicle( self.state_vector, control, self.disturbance, )
        _integrate( self.state_vector, self.acceleration, self.jerk, derivative, self.dt, )
        self._sync_dicts()
        self._update_obs()
        reward = ( self._get_reward(self.u_eride) if compute_reward else 0.0 )
        if self.bump_detected:
            self.eride_time += self.dt
            if self.eride_time >= self.eride_duration:
                self.eride_time = 0.0
                self.bump_detected = False
        self.time += self.dt
        truncated = self.time >= self.max_time
        info = ( self._get_info( self.disturbance, reward, self.u_eride, self.u_human, ) if build_info else None )
        return ( self.obs.copy(), reward, False, truncated, info, )

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
        front_position = ( position + self.vehicle_params.l_f * offset )
        rear_position = ( position - self.vehicle_params.l_r * offset )
        self.disturbance[0] = self.bump(front_position)
        self.disturbance[1] = self.bump(rear_position)

    def _get_unified_state(self):
        state = self.state.copy()
        state.update(self.state_ddot)
        state.update(self.state_dddot)
        state["u_eride"] = self.u_eride
        return state

    def _get_info(self, disturbance, reward, u_eride, u_human):
        info = { "time": self.time, "bump_detected": self.bump_detected, "disturbance": disturbance.copy(), "state": self._get_unified_state(), "reward": reward, "u_eride": u_eride, "u_human": u_human, }
        info.update(self.reward_fn.reward_info)
        return info

    def _get_reward(self, u_eride):
        return self.reward_fn( self.state, self.state_ddot, self.state_dddot, u_eride, )

    def detect_bump(self, disturbance):
        front_residual = abs( self.bump_gain_front * disturbance[0] )
        rear_residual = abs( self.bump_gain_rear * disturbance[1] )
        self.bump_detected = ( front_residual > 0.05 or rear_residual > 0.05 )

    def calculate_disturbance(self, state):
        pitch = state["theta"]
        position = state["x_com"]
        offset = np.cos(pitch)
        return np.array([ self.bump( position + self.vehicle_params.l_f * offset ), self.bump( position - self.vehicle_params.l_r * offset ), ])


class ErideEnv(SuspensionEnv):

    def __init__( self, eride_controller, human_controller=None, vehicle=None, reward_fn=None, mode="pure", action_dim=1, obs_keys=None, use_human=False, record_inner=False, ):
        assert mode in ("schedule", "pure")
        self.eride_controller = eride_controller
        self.mode = mode
        self.record_inner = record_inner
        super().__init__( human_controller, vehicle, reward_fn, obs_keys, use_human, )
        self.observe_step = int( round( self.config.t_observe / self.config.dt_inner ) )
        self.current_step = 0
        control_ratio = ( self.eride_controller.control_dt / self.config.dt_inner )
        self.ctrl_period_step = int(round(control_ratio))
        assert self.ctrl_period_step >= 1
        assert abs( self.ctrl_period_step * self.config.dt_inner - self.eride_controller.control_dt ) < 1e-6
        if mode == "schedule":
            self.action_space = gymnasium.spaces.Box( -1.0, 1.0, shape=(action_dim,), dtype=np.float32, )
        else:
            self.action_space = gymnasium.spaces.Box( 0.0, 0.0, shape=(0,), dtype=np.float32, )

    def reset(self, **kwargs):
        self.eride_controller.reset()
        self.reward_fn.reset()
        self.current_step = 0
        return super().reset(**kwargs)

    def step(self, action):
        if self.mode == "schedule":
            self.eride_controller.schedule( np.asarray(action, dtype=np.float32) )
        control = self.eride_controller( self.obs_dict, self.state, )
        batch_reward = getattr( self.reward_fn, "batch_from_unified_states", None, )
        deferred_reward = callable(batch_reward)
        inner_states = [] if deferred_reward else None
        inner_infos = [] if self.record_inner else None
        total_reward = 0.0
        terminated = False
        truncated = False
        last_reward = 0.0
        for _ in range(self.ctrl_period_step):
            obs, reward, terminated, truncated, info = super().step( control, compute_reward=not deferred_reward, build_info=self.record_inner, )
            if deferred_reward:
                inner_states.append(self._get_unified_state())
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
        if deferred_reward:
            rewards = batch_reward(inner_states)
            total_reward = float(np.sum(rewards))
            last_reward = float(rewards[-1])
        info = self._get_info( self.disturbance, last_reward, self.u_eride, self.u_human, )
        info["controller"] = self.eride_controller.kind
        info["u_eride"] = float(control)
        info.update({ f"ctrl_{key}": value for key, value in self.eride_controller.get_params().items() })
        if self.record_inner:
            info["inner_infos"] = inner_infos
        return ( obs, total_reward, terminated, truncated, info, )
