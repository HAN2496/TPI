import numpy as np


class LinearReward:
    """사용자의 실제 보상함수: R(τ) = θ @ standardize(phi(τ)) - threshold.

    phi는 환경 래퍼가 소유한 에피소드 feature 함수 (Trajectory -> dict[str, float]).
    mean/sd/threshold는 기준 궤적 뱅크에서 적합해 R을 단일 τ의 함수로 고정한다.
    """

    def __init__(self, weights, phi, mean, sd, threshold=0.0):
        self.names = list(weights)
        self.theta = np.asarray([weights[n] for n in self.names])
        self.phi = phi
        self.mean = mean
        self.sd = sd
        self.threshold = threshold

    def features(self, tau):
        f = self.phi(tau)
        return np.asarray([f[n] for n in self.names])

    def R(self, tau):
        return float(self.theta @ ((self.features(tau) - self.mean) / self.sd) - self.threshold)
