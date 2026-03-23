import numpy as np

class Term:
    name: str
    def __call__(self, state):
        raise NotImplementedError

class PitchTerm(Term):
    name = "pitch"
    def __call__(self, state):
        dtheta = np.asarray(state["dtheta"])
        return (2.0 * dtheta) ** 2

class LongitudinalTerm(Term):
    name = "longitudinal"
    def __call__(self, state):
        ddx = np.asarray(state["ddx_com"])
        return ddx ** 2

class DdxThresholdTerm(Term):
    name = "ddx_threshold"
    def __init__(self, threshold: float, exp_scale: float):
        self.threshold = threshold
        self.exp_scale = exp_scale
    def __call__(self, state):
        ddx = np.asarray(state["ddx_com"])
        a = np.abs(ddx)
        excess = np.maximum(a - self.threshold, 0.0)
        base = np.exp(self.exp_scale * excess) -    1.0
        mask = (a > self.threshold).astype(base.dtype)
        return base * mask

class DthetaThresholdTerm(Term):
    name = "dtheta_threshold"
    def __init__(self, threshold: float, exp_scale: float):
        self.threshold = threshold
        self.exp_scale = exp_scale
    def __call__(self, state):
        dth = np.asarray(state["dtheta"])
        a = np.abs(dth)
        excess = np.maximum(a - self.threshold, 0.0)
        base = np.exp(self.exp_scale * excess) - 1.0
        mask = (a > self.threshold).astype(base.dtype)
        return base * mask