import math
import numpy as np
from numba import njit

from ...configs import Vehicle_Parameters


@njit(cache=True, fastmath=True)
def _compute_dynamics(state, control, road, params, derivative):
    ks_f, ks_r           = params[0], params[1]
    cs_f, cs_r           = params[2], params[3]
    k_tire_f, k_tire_r   = params[4], params[5]
    l_f, l_r             = params[6], params[7]
    rolling_f, rolling_r = params[8], params[9]
    m_tot, m_s           = params[10], params[11]
    m_us_f, m_us_r       = params[12], params[13]
    pitch_inertia        = params[14]
    drive_gain           = params[15]
    air_drag_gain        = params[16]
    traction_limit       = params[17]
    h_f, h_r             = params[18], params[19]
    tan_phi_f, tan_phi_r = params[20], params[21]

    dz_com, pitch_rate, dz_front, dz_rear, velocity = state[:5]
    z_com, pitch, z_front, z_rear, _ = state[5:]

    sin_pitch = math.sin(pitch)
    cos_pitch = math.cos(pitch)
    direction = 1.0 if velocity > 0 else (-1.0 if velocity < 0 else 0.0)

    front_deflection = l_f * sin_pitch + z_com - z_front
    rear_deflection = -l_r * sin_pitch + z_com - z_rear
    front_deflection_rate = l_f * pitch_rate * cos_pitch + dz_com - dz_front
    rear_deflection_rate = -l_r * pitch_rate * cos_pitch + dz_com - dz_rear

    front_suspension = ks_f * front_deflection + cs_f * front_deflection_rate
    rear_suspension = ks_r * rear_deflection + cs_r * rear_deflection_rate
    front_tire = k_tire_f * (z_front - road[0])
    rear_tire = k_tire_r * (z_rear - road[1])

    traction = min(max(drive_gain * control, -traction_limit), traction_limit)
    air_drag = air_drag_gain * velocity * velocity * direction
    front_rolling = rolling_f * direction
    rear_rolling = rolling_r * direction

    front_longitudinal = -front_rolling
    rear_longitudinal = traction - rear_rolling
    front_vertical = -front_longitudinal * tan_phi_f
    rear_vertical = rear_longitudinal * tan_phi_r

    extra_pitch_moment = (
        front_longitudinal * tan_phi_f * l_f
        + rear_longitudinal * tan_phi_r * l_r
    ) * cos_pitch

    derivative[0] = (-front_suspension - rear_suspension + front_vertical + rear_vertical) / m_s
    derivative[1] = (
        -front_suspension * l_f * cos_pitch
        + rear_suspension * l_r * cos_pitch
        + traction * h_r
        - front_rolling * h_f
        - rear_rolling * h_r
        + extra_pitch_moment
    ) / pitch_inertia
    derivative[2] = (front_suspension - front_tire - front_vertical) / m_us_f
    derivative[3] = (rear_suspension - rear_tire - rear_vertical) / m_us_r
    derivative[4] = (traction - air_drag - front_rolling - rear_rolling) / m_tot
    derivative[5] = dz_com
    derivative[6] = pitch_rate
    derivative[7] = dz_front
    derivative[8] = dz_rear
    derivative[9] = velocity


class VehicleModel:
    __slots__ = ("params", "derivative")

    def __init__(self, params=None):
        params = Vehicle_Parameters() if params is None else params

        mass_sprung = params.m_tot - params.m_us_f - params.m_us_r
        wheelbase = params.l_f + params.l_r
        normal_front = mass_sprung * 9.81 * params.l_r / wheelbase + params.m_us_f * 9.81
        normal_rear = mass_sprung * 9.81 * params.l_f / wheelbase + params.m_us_r * 9.81

        self.params = np.array([
            params.k_s_f,
            params.k_s_r,
            params.c_s_f,
            params.c_s_r,
            params.k_us_f,
            params.k_us_r,
            params.l_f,
            params.l_r,
            params.C_r_f * normal_front,
            params.C_r_r * normal_rear,
            params.m_tot,
            mass_sprung,
            params.m_us_f,
            params.m_us_r,
            params.I,
            params.eta_drive * params.r_transmission / params.r_wheel,
            0.5 * params.rho_air * params.A * params.C_d,
            params.mu * normal_rear,
            params.h_f,
            params.h_r,
            np.tan(params.phi_f),
            np.tan(params.phi_r),
        ], dtype=np.float64)

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
        state_step = 1e-5 * np.maximum(1.0, np.abs(state_eq))
        control_step = 1e-5 * max(1.0, abs(control_eq))

        A = np.empty((10, 10), dtype=np.float64)
        for index in range(10):
            step = state_step[index]
            delta = np.zeros(10, dtype=np.float64)
            delta[index] = step
            forward = self.system_dynamics(state_eq + delta, control_eq, road_eq).copy()
            backward = self.system_dynamics(state_eq - delta, control_eq, road_eq).copy()
            A[:, index] = (forward - backward) / (2.0 * step)

        forward = self.system_dynamics(state_eq, control_eq + control_step, road_eq).copy()
        backward = self.system_dynamics(state_eq, control_eq - control_step, road_eq).copy()
        B = ((forward - backward) / (2.0 * control_step))[:, None]
        return A, B


def compile_vehicle_model(params=None):
    model = VehicleModel(params)
    state = np.zeros(10, dtype=np.float64)
    state[4] = 10.0
    model(state, 0.0, np.zeros(2, dtype=np.float64))
    return model
