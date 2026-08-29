import numpy as np

from .features import EPISODE, STEP


PREFERENCE_FEATURES = ("pitch_rate_sq", "long_accel_sq")
PARAMETER_NAMES = ("bias",) + PREFERENCE_FEATURES

# Reward parameters seen by the fully Bayesian Bernoulli-logit model:
#   theta = [-mu, -w_pitch, -w_long]
#   p(good | tau, user) = sigmoid([1, features(tau)] @ theta_user)
# Temperature is omitted because binary feedback identifies only mu/T and w/T.
# The negative non-bias means encode aversion to the corresponding discomfort.
# Anchor the mean weights at the squared-feature Oracle-A-level fit, then set
# the population bias for balanced labels under independently randomized gain,
# bump, and initial-velocity scenarios. The covariance axes separately vary
# preferred gain, parameter scale, and feedback bias while theta remains one MVN.
RESPONSE_SCALE = 0.14
GAIN_AXIS_SCALE = 0.7
DEFAULT_THETA_MEAN = RESPONSE_SCALE * np.asarray(
    [38.0, -33.70208881, -5.50478316], dtype=float,
)


def _default_theta_cov():
    gain_axis = GAIN_AXIS_SCALE * np.asarray([0.0, -3.5, 1.3])
    scale_axis = np.asarray([4.5, -4.5, -0.735])
    bias_axis = np.asarray([2.5, 0.0, 0.0])
    loading = np.column_stack((gain_axis, scale_axis, bias_axis))
    return RESPONSE_SCALE ** 2 * (loading @ loading.T)


DEFAULT_THETA_COV = _default_theta_cov()


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


class User:

    def __init__(self, name, step=None, episode=None, mu=-1.0, flip=0.0, archetype=None):
        self.name = name
        self.step = {n: float(w) for n, w in dict(step or {}).items()}
        self.episode = {n: float(w) for n, w in dict(episode or {}).items()}
        unknown = (set(self.step) - set(STEP)) | (set(self.episode) - set(EPISODE))
        if unknown:
            raise KeyError(f"Unknown user reward features: {sorted(unknown)}")
        self.mu = float(mu)
        self.flip = float(flip)
        if not 0.0 <= self.flip <= 1.0:
            raise ValueError("flip must be between 0 and 1")
        self.archetype = archetype or name

    @classmethod
    def from_theta(cls, name, theta, flip=0.0):
        """Create a user directly from Gaussian reward parameter theta."""
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (len(PARAMETER_NAMES),):
            raise ValueError(f"theta must have shape ({len(PARAMETER_NAMES)},)")
        weights = -theta[1:]
        step = {name: float(weights[i]) for i, name in enumerate(PREFERENCE_FEATURES)}
        return cls(name, step=step, mu=-theta[0], flip=flip)

    def feature_vector(self, tau):
        return np.asarray([float(np.mean(STEP[name](tau))) for name in PREFERENCE_FEATURES])

    def weight_vector(self):
        return np.asarray([self.step.get(name, 0.0) for name in PREFERENCE_FEATURES])

    def parameter_vector(self):
        return np.concatenate(([-self.mu], -self.weight_vector()))

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

    def z(self, taus, mu=None):
        mu = self.mu if mu is None else float(mu)
        return self.R(taus) - mu

    def p_good(self, taus, mu=None):
        return sigmoid(self.z(taus, mu=mu))

    def feedback(self, taus, rng):
        p = np.atleast_1d(self.p_good(taus))
        y = (rng.random(len(p)) < p).astype(np.int8)
        return np.where(rng.random(len(p)) < self.flip, 1 - y, y)

    def __repr__(self):
        return f"User({self.name}, mu={self.mu:+.2f}, ||w||={np.linalg.norm(self.weight_vector()):.2f})"


def _corr(a, b):
    return float(np.mean((a - a.mean()) * (b - b.mean())) / max(a.std() * b.std(), 1e-12))


class Users:

    def __init__(self, users, population_mean=None, population_cov=None, sampled_theta=None):
        users = list(users)
        self.users = {u.name: u for u in users}
        if len(self.users) != len(users):
            raise ValueError("User names must be unique")
        self.population_mean = None if population_mean is None else np.asarray(population_mean, dtype=float).copy()
        self.population_cov = None if population_cov is None else np.asarray(population_cov, dtype=float).copy()
        self.sampled_theta = None if sampled_theta is None else np.asarray(sampled_theta, dtype=float).copy()
        if self.sampled_theta is not None and self.sampled_theta.shape != (len(users), len(PARAMETER_NAMES)):
            raise ValueError("sampled_theta shape must match users and parameter dimensions")

    @classmethod
    def designed(cls, specs=None, flip=0.0):
        specs = ARCHETYPES if specs is None else specs
        return cls([User(name, flip=flip, **spec) for name, spec in specs.items()])

    @classmethod
    def gaussian(cls, n, seed, theta_mean=None, theta_cov=None, flip=0.0):
        """Draw fixed users from one multivariate-normal theta population."""
        if n <= 0:
            raise ValueError("n must be positive")
        mean = DEFAULT_THETA_MEAN if theta_mean is None else np.asarray(theta_mean, dtype=float)
        cov = DEFAULT_THETA_COV if theta_cov is None else np.asarray(theta_cov, dtype=float)
        d = len(PARAMETER_NAMES)
        if mean.shape != (d,) or cov.shape != (d, d):
            raise ValueError(f"theta_mean/theta_cov must have shapes ({d},) and ({d}, {d})")
        if not np.allclose(cov, cov.T) or np.linalg.eigvalsh(cov).min() <= 0.0:
            raise ValueError("theta_cov must be symmetric positive definite")

        rng = np.random.default_rng(seed)
        thetas = rng.multivariate_normal(mean, cov, size=n)
        users = [
            User.from_theta(f"user_{i:03d}", theta, flip=flip) for i, theta in enumerate(thetas)
        ]
        return cls(users, population_mean=mean, population_cov=cov, sampled_theta=thetas)

    @classmethod
    def sample(cls, n, seed, **kwargs):
        """Backward-compatible name for Gaussian population sampling."""
        return cls.gaussian(n, seed, **kwargs)

    def parameter_matrix(self):
        return np.stack([u.parameter_vector() for u in self])

    def roundtrip_error(self):
        if self.sampled_theta is None:
            raise ValueError("This Users object has no source theta samples")
        return float(np.max(np.abs(self.parameter_matrix() - self.sampled_theta)))

    def draw_population(self, n, seed):
        if self.population_mean is None or self.population_cov is None:
            raise ValueError("This Users object has no Gaussian population metadata")
        return np.random.default_rng(seed).multivariate_normal(
            self.population_mean, self.population_cov, size=n
        )

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
            p = u.p_good(taus)
            m = dict(R_mean=float(R.mean()), R_std=float(R.std()), p_good=float(p.mean()),
                     informative=float(np.mean((0.05 < p) & (p < 0.95))))
            if ref is not None:
                m["corr"] = _corr(R, ref)
            out[u.name] = m
            print(f"  {u.name:12s} mu={u.mu:+.3f} ||w||={np.linalg.norm(u.weight_vector()):.3f}  "
                  f"R={m['R_mean']:+.3f}±{m['R_std']:.3f}"
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
    "long_averse": dict(step={"pitch_rate_sq": 3.00, "long_accel_sq": 1.00}, mu=-3.00),
    "mid_low_gain": dict(step={"pitch_rate_sq": 3.28, "long_accel_sq": 0.72}, mu=-2.80),
    "balanced": dict(step={"pitch_rate_sq": 3.48, "long_accel_sq": 0.52}, mu=-2.60),
    "mid_high_gain": dict(step={"pitch_rate_sq": 3.64, "long_accel_sq": 0.36}, mu=-2.40),
    "pitch_averse": dict(step={"pitch_rate_sq": 3.74, "long_accel_sq": 0.26}, mu=-2.20),
}

MANUAL_STATS = {
    "dtheta": ["rms", "p2p", "abs_peak", "rms_deriv", "vdv"],
    "ddx_com": ["rms", "p2p", "abs_peak", "rms_deriv", "vdv"],
}
