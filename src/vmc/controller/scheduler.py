import numpy as np

"""
Action shape
    Scalar: Scalar parameter (P)
    Vector: Vector parameter (LQR, MPC)
Logic
    Direct| Direct action                            | k * a
    Deriv | Derivative of action based on current    | current + delta * a
"""

class BaseMapper:
    def __init__(self, clip=None):
        self.clip = clip
        self.param = np.array(self.initial_value(), np.float32)

    def initial_value(self):
        raise NotImplementedError

    def compute_target(self, action, current):
        raise NotImplementedError

    def reset(self):
        self.param = np.array(self.initial_value(), np.float32)

    def update(self, action):
        target = self.compute_target(action, current=self.param)
        if self.clip is not None:
            if isinstance(self.clip[0], (list, tuple, np.ndarray)):
                for i, (lo, hi) in enumerate(self.clip):
                    target[i] = np.clip(target[i], lo, hi)
            else:
                lo, hi = self.clip
                target = np.clip(target, lo, hi)
        self.param = np.array(target, np.float32)
        return self.param

class ScalarBaseMapper(BaseMapper):
    def __init__(self, **kwargs):
        self.action_dim = 1
        super().__init__(**kwargs)
        
    def update(self, action):
        target = self.compute_target(action, current=self.param)
        if self.clip is not None:
            lo, hi = self.clip
            target = np.clip(target, lo, hi)
        self.param = np.array(target, np.float32)
        return self.param

# ---- Scalar ----
class ScalarDirectMap(ScalarBaseMapper):
    def __init__(self, scale, clip=None, init=None):
        self.scale = float(scale)
        self.init = init if init is not None else 0.0
        super().__init__(clip=clip)

    def initial_value(self): 
        return self.init
    
    def compute_target(self, action, current):
        a = float(np.asarray(action).reshape(()))
        return self.scale * a

class ScalarDerivMap(ScalarBaseMapper):
    def __init__(self, base_value, delta_per_action, clip=None):
        self.base_value = float(base_value)
        self.delta = float(delta_per_action)
        super().__init__(clip=clip)

    def initial_value(self):
        return self.base_value

    def compute_target(self, action, current):
        a = float(np.asarray(action).reshape(()))
        return current + self.delta * a


class VectorBaseMapper(BaseMapper):
    def __init__(self, dim, indices=None, **kwargs):
        self.N = int(dim)
        self.idxs = list(range(self.N)) if indices is None else list(indices)
        self.action_dim = len(self.idxs)
        self.initial_val = np.zeros(self.N, np.float32)
        super().__init__(**kwargs)

    def update(self, action):
        target = self.compute_target(action, current=self.param)
        if self.clip is not None:
            for k, (lo, hi) in enumerate(self.clip):
                idx = self.idxs[k]
                target[idx] = np.clip(target[idx], lo, hi)
        self.param = np.array(target, np.float32)
        return self.param

# ---- Vector ----
class VectorDirectMap(VectorBaseMapper):
    def __init__(self, dim, indices=None, direct_gain=1.0, init=None, clip=None):
        super().__init__(dim=dim, indices=indices, clip=clip)

        g = np.asarray(direct_gain, np.float32)
        if g.size == 1: g = np.full(len(self.idxs), float(g), np.float32)
        assert g.size == len(self.idxs), "direct_gain length mismatch"
        self.gain = g
        self.initial_val = np.zeros(self.N, np.float32) if init is None else np.asarray(init, np.float32)

    def initial_value(self): 
        return self.initial_val.copy()

    def compute_target(self, action, current):
        a = np.asarray(action, np.float32).reshape(-1)
        assert a.size == len(self.idxs), "action length must match indices"
        tgt = np.array(current, np.float32)
        for k, i in enumerate(self.idxs):
            tgt[i] = self.gain[k] * a[k]
        return tgt

class VectorDerivMap(VectorBaseMapper):
    def __init__(self, dim, indices=None, base_value=1.0, delta_per_action=0.1, clip=None):
        super().__init__(indices=indices, dim=dim, clip=clip)

        # For Q matrix scheduling: base_value should be Q diagonal values
        b = np.asarray(base_value, np.float32)
        if b.size == 1: b = np.full(self.N, float(b), np.float32)
        assert b.size == self.N, "base_value length mismatch"
        self.initial_val = b.copy()

        d = np.asarray(delta_per_action, np.float32)
        if d.size == 1: d = np.full(len(self.idxs), float(d), np.float32)
        assert d.size == len(self.idxs), "delta_per_action length mismatch"
        self.delta = d

    def initial_value(self):
        return self.initial_val.copy()

    def compute_target(self, action, current):
        a = np.asarray(action, np.float32).reshape(-1)
        assert a.size == len(self.idxs), "action length must match indices"
        tgt = np.array(current, np.float32)
        for k, i in enumerate(self.idxs):
            tgt[i] = current[i] + self.delta[k] * a[k]
        return tgt