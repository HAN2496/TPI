"""Reward 프로토콜 — R(tau) -> float 를 구현하면 Reward다.

선택 능력: draws(tau) -> (S,) posterior 불확실성, r_step(states) -> (T,) step 보상(CoPL).
"""


class EnvReward:
    """env 고유 보상 — 래퍼가 기록한 env_reward 채널의 합."""

    def R(self, tau):
        return float(tau.channels["env_reward"].sum())


class Blend:
    """④가 최적화하는 최종 보상: R = user + lam * base. 기본 lam=0 = base 미사용."""

    def __init__(self, user, base=None, lam=0.0):
        self.user = user
        self.base = base or EnvReward()
        self.lam = lam

    def R(self, tau):
        return self.user.R(tau) + self.lam * self.base.R(tau)
