import numpy as np

from .features import EPISODE, STEP


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


class User:

    def __init__(self, name, step=None, episode=None, mu=-1.0, T=0.25, flip=0.0, archetype=None):
        step, episode = dict(step or {}), dict(episode or {})
        total = sum(abs(w) for w in list(step.values()) + list(episode.values())) or 1.0
        self.name = name
        self.step = {n: w / total for n, w in step.items()}
        self.episode = {n: w / total for n, w in episode.items()}
        self.mu = mu
        self.T = T
        self.flip = flip
        self.archetype = archetype or name

    def r_step(self, tau):
        r = np.zeros(len(tau))
        for name, w in self.step.items():
            r -= w * STEP[name](tau)
        return r

    def R(self, taus):
        if not hasattr(taus, "channels"):
            return np.asarray([self.R(t) for t in taus], dtype=float)
        r = float(self.r_step(taus).mean())
        for name, w in self.episode.items():
            r -= w * EPISODE[name](taus)
        return r

    def z(self, taus):
        return (self.R(taus) - self.mu) / self.T

    def p_good(self, taus):
        return sigmoid(self.z(taus))

    def feedback(self, taus, rng):
        p = np.atleast_1d(self.p_good(taus))
        y = (rng.random(len(p)) < p).astype(np.int8)
        return np.where(rng.random(len(p)) < self.flip, 1 - y, y)

    def __repr__(self):
        return f"User({self.name}, mu={self.mu:+.2f}, T={self.T:.2f})"


def _jitter(weights, rng, sd):
    return {name: float(w * np.exp(rng.normal(0.0, sd))) for name, w in weights.items()}


def _corr(a, b):
    return float(np.mean((a - a.mean()) * (b - b.mean())) / max(a.std() * b.std(), 1e-12))


class Users:

    def __init__(self, users):
        self.users = {u.name: u for u in users}

    @classmethod
    def designed(cls, specs=None, flip=0.0):
        specs = ARCHETYPES if specs is None else specs
        return cls([User(name, flip=flip, **spec) for name, spec in specs.items()])

    @classmethod
    def sample(cls, n, seed, specs=None, w_sd=0.15, mu_sd=0.1, T_sd=0.15, flip=0.0):
        specs = ARCHETYPES if specs is None else specs
        rng = np.random.default_rng(seed)
        names = list(specs)
        users = []
        for i in range(n):
            a = names[i % len(names)]
            spec = specs[a]
            users.append(User(
                f"user_{i:03d}_{a}", archetype=a, flip=flip,
                step=_jitter(spec.get("step", {}), rng, w_sd),
                episode=_jitter(spec.get("episode", {}), rng, w_sd),
                mu=float(spec.get("mu", -1.0) + rng.normal(0.0, mu_sd)),
                T=float(spec.get("T", 0.25) * np.exp(rng.normal(0.0, T_sd)))))
        return cls(users)

    def R(self, taus):
        return np.stack([u.R(taus) for u in self])

    def p_good(self, taus):
        return np.stack([u.p_good(taus) for u in self])

    def feedback(self, taus, rng):
        return np.stack([u.feedback(taus, rng) for u in self])

    def filter(self, **kw):
        def ok(u):
            return all(getattr(u, k) in v if isinstance(v, (list, tuple, set)) else getattr(u, k) == v
                       for k, v in kw.items())
        return Users(u for u in self if ok(u))

    def summary(self, taus, ref=None):
        out = {}
        ref = None if ref is None else np.asarray(ref, dtype=float)
        for u in self:
            R = u.R(taus)
            p = sigmoid((R - u.mu) / u.T)
            m = dict(R_mean=float(R.mean()), R_std=float(R.std()), p_good=float(p.mean()),
                     informative=float(np.mean((0.05 < p) & (p < 0.95))))
            if ref is not None:
                m["corr"] = _corr(R, ref)
            out[u.name] = m
            print(f"  {u.name:22s} mu={u.mu:+.2f} T={u.T:.2f}  R={m['R_mean']:+.2f}±{m['R_std']:.2f}"
                  f"  p_good={m['p_good']:.2f}  informative={m['informative']:.2f}"
                  + (f"  corr={m['corr']:+.2f}" if ref is not None else ""))
        return out

    @property
    def names(self):
        return list(self.users)

    def __getitem__(self, key):
        return self.users[key] if isinstance(key, str) else list(self.users.values())[key]

    def __iter__(self):
        return iter(self.users.values())

    def __len__(self):
        return len(self.users)

    def __repr__(self):
        head = ", ".join(self.names[:4])
        return f"Users({len(self)}: {head}{' …' if len(self) > 4 else ''})"


ARCHETYPES = {
    "pitch_averse": dict(step={"pitch_rate_sq": 0.85, "long_accel_sq": 0.15},
                         mu=-1.00, T=0.18),
    "long_averse": dict(step={"pitch_rate_sq": 0.25, "long_accel_sq": 0.75},
                        mu=-1.00, T=0.25),
    "rms_averse": dict(episode={"pitch_rate_rms": 0.70, "long_accel_rms": 0.30},
                       mu=-1.00, T=0.10),
    "peak_averse": dict(episode={"pitch_rate_p2p": 0.60, "long_accel_p2p": 0.40},
                        mu=-0.85, T=0.12),
    "balanced": dict(step={"pitch_rate_sq": 0.40, "long_accel_sq": 0.30},
                     episode={"pitch_rate_p2p": 0.30}, mu=-1.10, T=0.30),
}

MANUAL_STATS = {
    "dtheta": ["rms", "p2p", "abs_peak", "rms_deriv", "vdv"],
    "ddx_com": ["rms", "p2p", "abs_peak", "rms_deriv", "vdv"],
}
