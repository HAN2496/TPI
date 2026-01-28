import numpy as np
from numba import jit
import math
import matplotlib.pyplot as plt

from src.vmc.controller import PController
from src.vmc.configs import Vehicle_Parameters
from src.vmc.env.plant import SuspensionEnv, ErideEnv


@jit(nopython=True, cache=True)
def _compute_dynamics(x_data, u, z_data, params_data, out_data):
    # --- 1. Unpack Constants ---
    k_s_f, k_s_r          = params_data[0], params_data[1]
    c_s_f, c_s_r          = params_data[2], params_data[3] 
    k_us_f, k_us_r        = params_data[4], params_data[5]
    l_f, l_r              = params_data[6], params_data[7]
    c_r_f, c_r_r          = params_data[8], params_data[9]
    m_tot, m_us_f, m_us_r = params_data[10], params_data[11], params_data[12]
    I, r_wheel            = params_data[13], params_data[14]
    mu, eta_r, k_air      = params_data[15], params_data[16], params_data[17]
    g                     = params_data[18]
    h_f, h_r              = params_data[19], params_data[20]
    tan_phi_f, tan_phi_r  = params_data[21], params_data[22]
    
    # [NEW] Additional Parameters for Wheel Dynamics
    I_wheel_r             = params_data[23]  # Rear wheel inertia
    C_slip                = params_data[24]  # Tire Longitudinal Stiffness

    # --- 2. Unpack State Variables ---
    # x_data structure changed: size 10 -> 11
    # [velocities (6)] + [positions (5)]
    dz_com, dtheta, dz_us_f, dz_us_r, dx_com = x_data[0:5]
    omega_r                                  = x_data[5]  # [NEW] Rear wheel angular velocity
    z_com, theta, z_us_f, z_us_r, x_com      = x_data[6:11]

    # Precompute
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    sign_dx_com = 1.0 if dx_com > 0 else (-1.0 if dx_com < 0 else 0.0)

    # --- 3. Suspension Forces ---
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

    # --- 4. Vertical Load Calculation (Quasi-static approx for tire model) ---
    L = l_f + l_r
    m_s = m_tot - m_us_f - m_us_r
    mg_s = m_s * g
    
    # Static Weight Distribution
    Fz_f_static = mg_s * (l_r / L) + m_us_f * g
    Fz_r_static = mg_s * (l_f / L) + m_us_r * g
    
    # Note: 엄밀한 Load Transfer는 가속도(ax)에 의존하지만, 
    # ax를 구하기 위해 Fx가 필요하므로 여기서는 Static 값(또는 이전 step 값)을 사용합니다.
    # Numba 루프 내에서 대수적 루프(Algebraic Loop) 방지를 위해 Static 값을 기준으로 Clamping 합니다.
    Fz_r = Fz_r_static 

    # --- 5. Tire & Drivetrain Dynamics (Revised) ---
    
    # 5.1 Slip Ratio Calculation
    # v_x가 0에 가까울 때 분모 폭발 방지 (Singularity Handling)
    v_safe = dx_com
    if v_safe < 0.1 and v_safe > -0.1:
        v_safe = 0.1 if v_safe >= 0 else -0.1
        
    # Slip ratio lambda = (r*w - v) / v
    slip_ratio = (r_wheel * omega_r - dx_com) / abs(v_safe)

    # 5.2 Traction Force (Linear Tire Model with Friction Circle)
    # Fx = C_slip * slip
    raw_traction = C_slip * slip_ratio
    max_traction = mu * Fz_r
    
    if raw_traction > max_traction: traction_force = max_traction
    elif raw_traction < -max_traction: traction_force = -max_traction
    else: traction_force = raw_traction

    # 5.3 Wheel Dynamics (Rotational Equation)
    # I_w * dw/dt = Torque - r * Fx
    wheel_torque_input = eta_r * u
    # Rolling resistance torque is approx (rolling_force * r_wheel), but usually handled as force on body.
    # Here we focus on drive torque vs reaction force.
    d_omega_r = (wheel_torque_input - traction_force * r_wheel) / I_wheel_r

    # --- 6. Body Forces & Moments ---
    
    # Aerodynamic and rolling resistance
    air_drag = k_air * dx_com * dx_com * sign_dx_com
    
    # Rolling resistance (simplified: force opposing motion)
    rolling_f = c_r_f * Fz_f_static * sign_dx_com
    rolling_r = c_r_r * Fz_r_static * sign_dx_com

    # Tire forces acting on body
    tire_long_f = -rolling_f
    tire_long_r = traction_force - rolling_r
    
    # Anti-Squat / Anti-Dive Forces (Geometry effect)
    tire_vert_f = -tire_long_f * tan_phi_f
    tire_vert_r = tire_long_r * tan_phi_r

    # Pitch moment from longitudinal forces (taking height into account implicit in Anti-squat, 
    # but explicit moment arm exists due to CG height vs ground contact)
    # Note: tan_phi model approximates the jacking force. 
    # The moment due to Fx acting at ground level vs CG is:
    pitch_moment_extra = (tire_long_f * tan_phi_f * l_f +
                          tire_long_r * tan_phi_r * l_r) * cos_theta

    # --- 7. Compute Accelerations (State Derivatives) ---
    
    # dz_com dot (Vertical Accel)
    out_data[0] = (-(spring_force_f + spring_force_r + damper_force_f + damper_force_r) + 
                   (tire_vert_f + tire_vert_r)) / m_s

    # dtheta dot (Pitch Accel)
    out_data[1] = (-(spring_force_f + damper_force_f) * l_f * cos_theta + 
                   (spring_force_r + damper_force_r) * l_r * cos_theta + 
                   traction_force * h_r - rolling_f * h_f - rolling_r * h_r + 
                   pitch_moment_extra) / I

    # dz_us_f dot
    out_data[2] = (spring_force_f + damper_force_f - unsprung_force_f - tire_vert_f) / m_us_f
    
    # dz_us_r dot
    out_data[3] = (spring_force_r + damper_force_r - unsprung_force_r - tire_vert_r) / m_us_r
    
    # dx_com dot (Longitudinal Accel)
    out_data[4] = (traction_force - air_drag - rolling_f - rolling_r) / m_tot

    # omega_r dot (Wheel Angular Accel) - [NEW]
    out_data[5] = d_omega_r

    # Velocities -> Positions mapping
    out_data[6]  = dz_com
    out_data[7]  = dtheta
    out_data[8]  = dz_us_f  
    out_data[9]  = dz_us_r
    out_data[10] = dx_com


class VehicleModel:
    # Buffer sizes increased for new state variables
    __slots__ = ('_params_arr', '_output_buffer', '_x_buffer', '_z_buffer')
    
    def __init__(self, params=None):
        if params is None:
            params = Vehicle_Parameters()
            
        # Default values for new parameters if they don't exist in config
        i_wheel_val = getattr(params, 'I_wheel', 1.5)      # kg*m^2 (Typical car wheel)
        c_slip_val  = getattr(params, 'C_slip', 50000.0)   # N/slip (Tire stiffness)

        self._params_arr = np.array([
            params.k_s_f, params.k_s_r,                    # 0, 1
            params.c_s_f, params.c_s_r,                    # 2, 3
            params.k_us_f, params.k_us_r,                  # 4, 5
            params.l_f, params.l_r,                        # 6, 7
            params.C_r_f, params.C_r_r,                    # 8, 9
            params.m_tot, params.m_us_f, params.m_us_r,    # 10, 11, 12
            params.I, params.r_wheel,                      # 13, 14
            params.mu,                                     # 15
            params.eta_drive * params.r_transmission,      # 16
            0.5 * params.rho_air * params.A * params.C_d,  # 17
            9.81,                                          # 18
            params.h_f, params.h_r,                        # 19, 20
            np.tan(params.phi_f), np.tan(params.phi_r),    # 21, 22
            i_wheel_val,                                   # 23 [NEW]
            c_slip_val                                     # 24 [NEW]
        ], dtype=np.float64)

        # Buffers: State dimension increased to 11
        # [dz_c, dth, dz_uf, dz_ur, dx_c, w_r, z_c, th, z_uf, z_ur, x_c]
        self._output_buffer = np.zeros(11, dtype=np.float64)
        self._x_buffer = np.zeros(11, dtype=np.float64) 
        self._z_buffer = np.zeros(2, dtype=np.float64)

    def __call__(self, x, u, z):
        # [수정됨] 함수 실행 후 버퍼를 명시적으로 반환
        _compute_dynamics(x, u, z, self._params_arr, self._output_buffer)
        return self._output_buffer

# Warm-up compilation function
def compile_vehicle_model(params=None):
    model = VehicleModel(params=params)
    
    # [수정됨] 모델 상태가 10개에서 11개로 늘어났으므로, 테스트 벡터도 11로 맞춰야 함
    test_state = np.zeros(11) 
    
    test_state[4] = 10.0  # velocity (dx_com)
    # test_state[5] = 0.0 # omega_r (새로 추가된 상태, 0으로 초기화됨)
    
    test_control = 0.0
    test_road = np.zeros(2)

    # Trigger JIT compilation
    model(test_state, test_control, test_road)
    return model

NEW_STATE_KEYS = ["dz_com", "dtheta", "dz_us_f", "dz_us_r", "dx_com", 
                  "omega_r",  # [NEW] Inserted at index 5
                  "z_com", "theta", "z_us_f", "z_us_r", "x_com"]

class AdvancedErideEnv(ErideEnv):
    """
    기존 ErideEnv를 상속받아 state 크기(10->11)와 매핑 로직만 수정한 클래스.
    라이브러리 원본 코드를 건드리지 않고 문제를 해결함.
    """
    def _reset_state(self):
        # 1. 11개짜리 상태 벡터 초기화
        self.state = {key: 0.0 for key in NEW_STATE_KEYS}
        
        # x0 (초기값 설정) - 기본 x0는 10개일 수 있으므로 주의해서 매핑
        # 기존 설정: x0 순서는 [dz_c, dth, dz_uf, dz_ur, dx_c, z_c, th, z_uf, z_ur, x_c]
        # dx_com(속도)는 index 4에 있음.
        
        # 먼저 기존 x0 값들을 적절히 매핑
        # (라이브러리의 self.x0가 10개라고 가정)
        current_idx = 0
        for i, key in enumerate(NEW_STATE_KEYS):
            if key == "omega_r":
                continue # omega_r은 x0에 없으므로 건너뜀
            if current_idx < len(self.x0):
                self.state[key] = self.x0[current_idx]
                current_idx += 1
                
        # 2. [중요] 초기 휠 속도 동기화 (Slip = 0 상태로 시작)
        # v = r * w  =>  w = v / r
        r_wheel = self.vehicle_params.r_wheel
        self.state["omega_r"] = self.state["dx_com"] / r_wheel

        # 가속도/저크 초기화 (기존 키 사용해도 무방하지만 omega_r dot은 누락될 수 있음. 일단 유지)
        from src.vmc.env.plant import ACCEL_KEYS, JERK_KEYS # 필요한 키 가져오기
        self.state_ddot = {key: 0.0 for key in ACCEL_KEYS}
        # omega_r의 가속도인 d_omega_r을 저장할 공간이 원래 ACCEL_KEYS엔 없음.
        # 하지만 simulation loop에서는 self.state_ddot 딕셔너리에 동적으로 할당하므로 에러는 안 날 것임.
        self.state_ddot["d_omega_r"] = 0.0 # 명시적 추가
        
        self.state_dddot = {key: 0.0 for key in JERK_KEYS}
        self.obs, self.obs_dict = self._get_obs()

        # 3. 초기 동역학 계산
        # 딕셔너리 -> 배열 변환 (11개)
        x = np.array([self.state[key] for key in NEW_STATE_KEYS], dtype=np.float32)
        z = self.calculate_disturbance(self.state.copy())
        u_human = self.human_controller(self.obs_dict, self.state)
        
        # 여기서 11개짜리 dx가 반환됨
        dx = np.asarray(self.vehicle(x, u_human, z), dtype=np.float32)

        # 가속도 업데이트
        # dx 구조: [ddz, ddth, ..., ddx, dw_r, dz, dth, ...]
        # 가속도 부분(앞 6개) 매핑
        # 기존 ACCEL_KEYS는 5개 ["ddz_com", "ddtheta", "ddz_us_f", "ddz_us_r", "ddx_com"]
        # dx[5]는 dw_r (각가속도)
        
        idx = 0
        for key in ACCEL_KEYS:
            self.state_ddot[key] = dx[idx]
            idx += 1
        self.state_ddot["d_omega_r"] = dx[5] # 추가된 상태 미분값 저장

        self.state_dddot = {key: 0.0 for key in JERK_KEYS}
        self.obs, self.obs_dict = self._get_obs()

    def step(self, action):
        # ErideEnv.step을 완전히 덮어쓸 필요 없이, 부모의 step을 호출하되
        # 내부적으로 사용하는 _reset_state, _get_info 등이 오버라이드 되었으므로
        # SuspensionEnv 부분의 로직(물리 업데이트)도 수정이 필요합니다.
        # --> 상속 구조상 SuspensionEnv.step이 하드코딩된 STATE_KEYS(10개)를 쓰면 에러가 납니다.
        # 따라서, 어쩔 수 없이 step 함수 전체를 오버라이드 해야 안전합니다.
        
        # 여기서는 ErideEnv 로직(스케줄링 등) + SuspensionEnv 로직(물리)을 합친 형태입니다.
        
        if self.mode == "schedule":
            self.eride_controller.schedule(np.asarray(action, dtype=np.float32))
        
        u_eride = self.eride_controller(self.obs_dict, self.state)

        # Inner Loop (Physics Step)
        total_reward = 0.0
        terminated = False
        truncated = False
        last_info = None
        
        steps_to_run = self.ctrl_period_step if self.ctrl_period_step > 1 else 1
        
        for _ in range(steps_to_run):
            # --- SuspensionEnv.step logic adaptation ---
            # 1. State array creation (11 elements)
            x = np.array([self.state[key] for key in NEW_STATE_KEYS], dtype=np.float32)
            z = self.calculate_disturbance(self.state.copy())
            
            # 2. Human Control
            u_human = self.human_controller(self.obs_dict, self.state)

            # 3. Bump Detection
            if not self.bump_detected:
                self.detect_bump(x, u_human, z)
                if self.bump_detected:
                    self.eride_time = 0.0
            
            # 4. Control Input
            actual_u_eride = 0.0
            if self.bump_detected:
                # If mode is pure/direct action provided in argument (handled differently in ErideEnv logic)
                # But here we simplified. u_eride is calculated above.
                actual_u_eride = u_eride

            u = u_human + actual_u_eride

            # 5. Dynamics
            dx = np.asarray(self.vehicle(x, u, z), dtype=np.float32)
            x_next = x + dx * self.dt

            # 6. Update State Dict
            for i, key in enumerate(NEW_STATE_KEYS):
                self.state[key] = x_next[i]

            # Update derivatives
            from src.vmc.env.plant import ACCEL_KEYS, JERK_KEYS
            prev_ddot = self.state_ddot.copy()
            
            # Map accelerations (indices 0~4)
            for i, key in enumerate(ACCEL_KEYS):
                self.state_ddot[key] = dx[i]
            # Map wheel accel (index 5)
            self.state_ddot["d_omega_r"] = dx[5]

            # Jerk calculation (omitted for omega_r for brevity, or add if needed)
            for key in JERK_KEYS:
                # ... existing logic ...
                pass 

            self.obs, self.obs_dict = self._get_obs()

            # 7. Reward & Time
            r = self._get_reward(actual_u_eride)
            total_reward += r

            if self.bump_detected:
                self.eride_time += self.dt
                if self.eride_time >= self.eride_duration:
                    self.eride_time = 0.0
                    self.bump_detected = False
            
            self.time += self.dt
            
            # Check termination
            if self.time >= self.max_time:
                truncated = True
                
            info = self._get_info(z, r, actual_u_eride, u_human)
            last_info = info
            
            self.current_step += 1
            if truncated: break

        # ErideEnv Logic: Check observation step
        if self.current_step >= self.observe_step:
             # Eride specific truncation logic could go here
             pass

        last_info = dict(last_info) if last_info else {}
        last_info.update({"u_eride": float(u_eride)})
        
        return self.obs.copy(), total_reward, terminated, truncated, last_info

    def detect_bump(self, x, u, z):
        # Override bump detection because x is now size 11
        # Need to ensure vehicle call handles size 11 correctly (it does)
        # Residual check indices: dz_us_f is index 2 in dx vector? 
        # No, dx vector from dynamics is: 
        # [ddz, ddth, dz_us_f_dot, dz_us_r_dot, ddx, dw, ...]
        # Wait, bump detection usually compares VELOCITIES or POSITIONS?
        # Code says: x_pred[2] - x_real[2]. 
        # In output buffer: 
        # indices 0-5 are accelerations/velocities derivatives.
        # indices 6-10 are velocities (passed through).
        # Original code: out_data[2] = dz_us_f_dot. 
        # So it checks unsprung mass vertical velocity difference.
        
        # In NEW dynamics: out_data[2] is still dz_us_f_dot. So this logic remains valid!
        x_pred = self.vehicle(x, u, z=[0, 0]).copy()
        x_real = self.vehicle(x, u, z=z)
        residual1 = np.abs(x_pred[2] - x_real[2])
        residual2 = np.abs(x_pred[3] - x_real[3])
        self.bump_detected = residual1 > 0.05 or residual2 > 0.05

# ==============================================================================
# 3. 컴파일 및 실행 (수정된 클래스 사용)
# ==============================================================================

def compile_vehicle_model(params=None):
    model = VehicleModel(params=params)
    # Warm-up with size 11 state
    test_state = np.zeros(11) 
    test_state[4] = 10.0
    model(test_state, 0.0, np.zeros(2))
    return model

# [변경] ErideEnv 대신 AdvancedErideEnv 사용
controller_1 = PController(kp=300)
controller_2 = PController(kp=30)
env_1 = AdvancedErideEnv(vehicle=compile_vehicle_model(), eride_controller=controller_1)
env_2 = AdvancedErideEnv(vehicle=compile_vehicle_model(), eride_controller=controller_2)

if __name__ == "__main__":
    def run_sim(env):
        env.reset(seed=42)
        history = {"dz_com": [], "dtheta": [], "dx_com": [], "omega_r": [], "time": []}
        while True:
            obs, reward, term, trunc, info = env.step(np.array([]))
            state = info["state"]
            history["dz_com"].append(state["dz_com"])
            history["dx_com"].append(state["dx_com"])
            history["dtheta"].append(state["dtheta"])
            history["omega_r"].append(state["omega_r"]) # New state tracking
            history["time"].append(info["time"])
            if term or trunc:
                break
        return history

    h1, h2 = run_sim(env_1), run_sim(env_2)

    keys = ["dz_com", "dtheta", "dx_com", "omega_r"]
    fig, axes = plt.subplots(len(keys), 1, figsize=(10, 12), sharex=True)
    
    for ax, key in zip(axes, keys):
        ax.plot(h1["time"], h1[key], label=f"kp=300")
        ax.plot(h2["time"], h2[key], label=f"kp=30", linestyle="--")
        ax.set_ylabel(key)
        ax.legend(loc="upper right")
        ax.grid(True)
        
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()