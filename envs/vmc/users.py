"""VMC 가상 사용자 — 설계된 참 보상, step(dense)·episode(sparse) 항의 명시적 분리.

  r(s_t) = -Σ_ch w·c_t²                              dense — CoPL류 step 추정의 GT 대응물
  R(τ)   = mean_t r(s_t) - Σ_ch w·p2p(c) - threshold  p2p 항은 에피소드에 한 번 (sparse)

threshold는 oracle.calibrate(users, bank)로 뱅크 기준 라벨 균형을 맞춘다.
가중치 스케일 근거(kp=50 뱅크 실측): mean dtheta²≈0.013, ddx²≈0.019, dddx²≈0.50,
p2p(dtheta)≈0.46, p2p(ddx)≈0.31 — 각 사용자의 주 항이 O(1)이 되게 잡았다.
run 스크립트만 이 모듈을 import한다 (control 금지).
"""
import numpy as np


class User:
    def __init__(self, name, step=None, p2p=None, threshold=0.0):
        self.name = name
        self.step = dict(step or {})     # {채널: w} — 매 스텝 -w·c_t²
        self.p2p = dict(p2p or {})       # {채널: w} — 에피소드에 한 번 -w·(max-min)
        self.threshold = threshold

    def r_step(self, tau):
        r = np.zeros(len(tau))
        for ch, w in self.step.items():
            r -= w * tau.channels[ch] ** 2
        return r

    def R(self, tau):
        r = float(self.r_step(tau).mean())
        for ch, w in self.p2p.items():
            r -= w * (tau.channels[ch].max() - tau.channels[ch].min())
        return r - self.threshold


USERS = {
    "pitch_averse": User("pitch_averse", step={"dtheta": 75.0, "ddx_com": 10.0}),
    "long_averse": User("long_averse", step={"dtheta": 15.0, "ddx_com": 50.0}),
    "jerk_averse": User("jerk_averse", step={"dtheta": 15.0, "dddx_com": 2.0}),
    "peak_averse": User("peak_averse", step={"dtheta": 15.0}, p2p={"dtheta": 2.0, "ddx_com": 1.5}),
    "balanced": User("balanced", step={"dtheta": 37.0, "ddx_com": 25.0}),
}

MANUAL_STATS = {                     # 추정 모델이 보는 (채널, 통계) — GT 설계와 다른 기저
    "dtheta": ["rms", "p2p", "abs_peak", "rms_deriv", "vdv"],
    "ddx_com": ["rms", "p2p", "abs_peak", "rms_deriv", "vdv"],
}
