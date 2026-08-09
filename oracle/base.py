import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


class Oracle:
    """가상 사용자: 참 보상 R을 σ(β·R)로 확률화해 binary feedback을 준다.

    beta=합리성(낮을수록 랜덤 응답), flip=라벨 노이즈 — 피드백 품질 개입은 이 둘뿐.
    """

    def __init__(self, reward, beta=1.0, flip=0.0, name=""):
        self.reward = reward
        self.beta = beta
        self.flip = flip
        self.name = name

    def p_good(self, taus):
        return sigmoid(self.beta * np.asarray([self.reward.R(tau) for tau in taus]))

    def feedback(self, taus, rng):
        p = self.p_good(taus)
        y = (rng.random(len(p)) < p).astype(np.int8)
        return np.where(rng.random(len(p)) < self.flip, 1 - y, y)
