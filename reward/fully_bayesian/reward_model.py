"""학습된 (phi, UserReward posterior)를 Trajectory 위의 Reward로 감싸는 어댑터."""
import numpy as np


class PosteriorReward:
    """R(τ) = posterior 평균 θ @ φ(τ), logit 스케일 — σ(R) ≈ p(good).

    φ 통계는 phi.fs 기준으로 계산되므로 τ의 샘플링 주기가 학습 데이터와 같아야 한다.
    """

    def __init__(self, phi, user):
        self.phi = phi
        self.user = user                # UserReward: .theta (S, d), .reward(Z) -> (S, N)

    def Z(self, tau):
        X = np.stack([tau.channels[ch] for ch in self.phi.channels], axis=-1)[None]
        return self.phi.transform(X).astype(np.float64)

    def R(self, tau):
        return float(self.user.reward(self.Z(tau)).mean())

    def draws(self, tau):
        return self.user.reward(self.Z(tau))[:, 0]
