"""Kalman-filter candidates for 6D bounce/pitch reconstruction.

The classes in this module deliberately share one fitting and filtering API so
that ``lab/reconstruction/baselines.py`` can compare dynamics without changing
the data split, loss, or output convention.  Raw channel order is the order in
``baselines.X_CH``.
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.signal import butter, filtfilt


GRAVITY = 9.81
DEFAULT_FS = 100.0
DEFAULT_BOUNCE_UNIT_MPS = 0.1
DEFAULT_FRONT_CG_M = 1.35
DEFAULT_REAR_CG_M = 1.55

ROLL_I = 0
VERT_ACCEL_I = 1
LONG_ACCEL_I = 4
WHEEL_SLICE = slice(5, 9)
REAR_TORQUE_I = 9   # MG1 follows the rear command in this dataset.
FRONT_TORQUE_I = 10  # MG2 follows the front command in this dataset.


def _lowpass(x, fs, cutoff=5.0):
    b, a = butter(2, cutoff / (fs / 2), "low")
    return filtfilt(b, a, x).astype(np.float32)


def _common_signals(xr, fs):
    wheel_speed = xr[:, :, WHEEL_SLICE] / 3.6
    speed = wheel_speed.mean(2)
    speed_dot = _lowpass(np.gradient(speed, axis=1) * fs, fs)
    vertical_accel = (xr[:, :, VERT_ACCEL_I] - 1.0) * GRAVITY
    longitudinal_specific_force = xr[:, :, LONG_ACCEL_I] * GRAVITY - speed_dot
    return {
        "roll": xr[:, :, ROLL_I],
        "wheel_speed": wheel_speed,
        "speed": speed,
        "speed_dot": speed_dot,
        "vertical_accel": vertical_accel,
        "longitudinal_specific_force": longitudinal_specific_force,
        "torque_front": xr[:, :, FRONT_TORQUE_I],
        "torque_rear": xr[:, :, REAR_TORQUE_I],
    }


def _rear_indices(wheel_speed, fs, wheelbase):
    """Index of the front-axle sample at the rear axle's current position."""
    distance = np.cumsum(np.maximum(wheel_speed.mean(2), 0.0) / fs, axis=1)
    distance -= distance[:, :1]
    rear_index = np.empty(distance.shape, dtype=np.int32)
    for i, dist in enumerate(distance):
        target = dist - wheelbase
        rear_index[i] = np.searchsorted(dist, target, side="left")
        rear_index[i, target < 0] = -1
    return rear_index


def _c2d(ac, bc, dt):
    """Zero-order-hold discretization for a continuous linear system."""
    n, m = ac.shape[0], bc.shape[1]
    if m == 0:
        return expm(ac * dt), np.zeros((n, 0))
    block = np.zeros((n + m, n + m))
    block[:n, :n] = ac
    block[:n, n:] = bc
    disc = expm(block * dt)
    return disc[:n, :n], disc[:n, n:]


class KalmanFilter(ABC):
    """Common linear Kalman-filter and offline-identification framework.

    Subclasses provide matrices, input preparation, parameter initialization,
    and bounds.  ``predict`` always returns [bounce, roll, pitch] with shape
    (episode, 3, time).  Bounce is converted to the raw 6D channel scale and
    pitch is converted from rad/s to deg/s.
    """

    fit_output_indices = (0, 2)
    bounce_state_index = None
    pitch_state_index = None
    use_smoother = False
    default_fit_size = 400
    default_maxfev = 2000

    def __init__(self, p, fs=DEFAULT_FS, warmup=0, control=True,
                 bounce_unit_mps=DEFAULT_BOUNCE_UNIT_MPS, **model_kwargs):
        self.p = np.asarray(p, dtype=float)
        self.fs = float(fs)
        self.dt = 1.0 / self.fs
        self.warmup = int(warmup)
        self.control = bool(control)
        self.bounce_unit_mps = float(bounce_unit_mps)
        self.model_kwargs = dict(model_kwargs)
        self._build_matrices()
        self._validate_matrices()
        self._build_stationary_gains()

    @classmethod
    @abstractmethod
    def initial_parameters(cls):
        raise NotImplementedError

    @abstractmethod
    def parameter_bounds(self):
        raise NotImplementedError

    @abstractmethod
    def _build_matrices(self):
        raise NotImplementedError

    @abstractmethod
    def prepare_inputs(self, xr):
        raise NotImplementedError

    def _validate_matrices(self):
        n = self.A.shape[0]
        if self.A.shape != (n, n):
            raise ValueError("A must be square")
        if self.B.shape[0] != n or self.H.shape[1] != n:
            raise ValueError("A, B, and H state dimensions do not match")
        if self.D.shape != (self.H.shape[0], self.B.shape[1]):
            raise ValueError("D must have shape (measurement, input)")
        if self.Q.shape != (n, n) or self.R.shape != (self.H.shape[0], self.H.shape[0]):
            raise ValueError("Q or R has an invalid shape")

    def _build_stationary_gains(self, steps=300):
        n = self.A.shape[0]
        p = np.eye(n) * (100.0 if self.warmup else 1.0)
        self.Ks = np.empty((steps, n, self.H.shape[0]))
        eye = np.eye(n)
        for i in range(steps):
            pp = self.A @ p @ self.A.T + self.Q
            ph = pp @ self.H.T
            s = self.H @ ph + self.R
            k = np.linalg.solve(s, ph.T).T
            ikh = eye - k @ self.H
            p = ikh @ pp @ ikh.T + k @ self.R @ k.T
            self.Ks[i] = k
        self.K = self.Ks[-1]

    def _extra_terms(self, prepared, t, state_history):
        n = prepared["observations"].shape[0]
        return np.zeros((n, self.A.shape[0])), np.zeros((n, self.H.shape[0]))

    def _filter_states(self, prepared):
        observations = prepared["observations"]
        inputs = prepared["inputs"]
        n_episode, n_time = observations.shape[:2]
        n_state = self.A.shape[0]
        states = np.zeros((n_episode, n_time, n_state))
        predicted = np.zeros_like(states) if self.use_smoother else None
        state = np.zeros((n_episode, n_state))

        if self.use_smoother:
            p = np.eye(n_state) * (100.0 if self.warmup else 1.0)
            p_pred = np.empty((n_time, n_state, n_state))
            p_filt = np.empty_like(p_pred)
            eye = np.eye(n_state)

        for t in range(n_time):
            extra_state, extra_measurement = self._extra_terms(prepared, t, states)
            state = state @ self.A.T + inputs[:, t] @ self.B.T + extra_state
            if self.use_smoother:
                predicted[:, t] = state
                pp = self.A @ p @ self.A.T + self.Q
                ph = pp @ self.H.T
                innovation_cov = self.H @ ph + self.R
                gain = np.linalg.solve(innovation_cov, ph.T).T
                ikh = eye - gain @ self.H
                p = ikh @ pp @ ikh.T + gain @ self.R @ gain.T
                p_pred[t], p_filt[t] = pp, p
            else:
                gain = self.Ks[t] if t < self.warmup and t < len(self.Ks) else self.K
            expected = state @ self.H.T + inputs[:, t] @ self.D.T + extra_measurement
            state += (observations[:, t] - expected) @ gain.T
            states[:, t] = state

        if not self.use_smoother:
            return states

        smoothed = states.copy()
        for t in range(n_time - 2, -1, -1):
            smoother_gain = np.linalg.solve(
                p_pred[t + 1].T, (p_filt[t] @ self.A.T).T
            ).T
            smoothed[:, t] += (smoothed[:, t + 1] - predicted[:, t + 1]) @ smoother_gain.T
        return smoothed

    def _project_outputs(self, states, prepared):
        n, t = states.shape[:2]
        if self.bounce_state_index is None:
            bounce = np.full((n, t), np.nan)
        else:
            bounce = states[:, :, self.bounce_state_index] / self.bounce_unit_mps
        if self.pitch_state_index is None:
            pitch = np.full((n, t), np.nan)
        else:
            pitch = -np.rad2deg(states[:, :, self.pitch_state_index])
        return np.stack([bounce, prepared["roll"], pitch], axis=1)

    def predict_prepared(self, prepared):
        return self._project_outputs(self._filter_states(prepared), prepared)

    def predict(self, xr):
        return self.predict_prepared(self.prepare_inputs(xr))

    def fit_summary(self, result):
        return f"loss {result.fun:.3f}  nfev {result.nfev}"

    @classmethod
    def fit(cls, xr, yr, fs=DEFAULT_FS, warmup=0, control=True,
            fit_size=None, maxfev=None, **model_kwargs):
        fit_size = cls.default_fit_size if fit_size is None else int(fit_size)
        maxfev = cls.default_maxfev if maxfev is None else int(maxfev)
        sub = np.linspace(0, len(xr) - 1, min(fit_size, len(xr))).astype(int)
        xs, ys = xr[sub], yr[sub]
        p0 = cls.initial_parameters()
        template = cls(p0, fs=fs, warmup=warmup, control=control, **model_kwargs)
        prepared = template.prepare_inputs(xs)
        components = tuple(template.fit_output_indices)
        scales = {j: ys[:, :, j].std() + 1e-8 for j in components}

        def loss(p):
            try:
                model = cls(p, fs=fs, warmup=warmup, control=control, **model_kwargs)
                prediction = model.predict_prepared(prepared)
                cost = sum(np.mean(((prediction[:, j] - ys[:, :, j]) / scales[j]) ** 2)
                           for j in components)
            except (FloatingPointError, ValueError, np.linalg.LinAlgError):
                return 1e6
            return cost if np.isfinite(cost) else 1e6

        result = minimize(loss, p0, method="Powell", bounds=template.parameter_bounds(),
                          options={"maxfev": maxfev})
        fitted = cls(result.x, fs=fs, warmup=warmup, control=control, **model_kwargs)
        fitted.fit_result = result
        print("  " + fitted.fit_summary(result))
        return fitted


class ReducedModalKalmanFilter(KalmanFilter):
    """Document Model 1: reduced bounce/pitch body modes with colored forcing."""

    bounce_state_index = 1
    pitch_state_index = 3

    @classmethod
    def initial_parameters(cls):
        dynamics = [8.0, 0.3, 7.8, 0.3, 0.5, 0.0, 0.0, 0.3, 0.0, 0.0, 5.0, 5.0]
        noise = np.log([1.0, 1.0, 1.0, 1.0, 1e-4, 1e-4, 0.1, 0.1])
        return np.r_[dynamics, noise]

    def parameter_bounds(self):
        bounds = [
            (0.1, 30.0), (0.01, 2.0), (0.1, 30.0), (0.01, 2.0),
            (0.05, 3.0), (-20.0, 20.0), (-20.0, 20.0),
            (-20.0, 20.0), (-0.1, 0.1), (-20.0, 20.0),
            (0.05, 30.0), (0.05, 30.0),
        ] + [(np.log(1e-10), np.log(1e4))] * 8
        if not self.control:
            for i in (6, 7, 8, 9):
                bounds[i] = (self.p[i], self.p[i])
        return bounds

    def _build_matrices(self):
        (wz, zz, wt, zt, lever, czt, kappa, ga, gt, gda,
         decay_z, decay_pitch) = self.p[:12]
        qz, qt, qez, qet, qb, qg, rz, rx = np.exp(self.p[12:])
        dt = self.dt
        self.A = np.eye(8)
        self.A[0, 1] = dt
        self.A[1, 0] = -wz * wz * dt
        self.A[1, 1] = 1 - 2 * zz * wz * dt
        self.A[1, 4] = dt
        self.A[2, 3] = dt
        self.A[3, 1] = czt * dt
        self.A[3, 2] = -wt * wt * dt
        self.A[3, 3] = 1 - 2 * zt * wt * dt
        self.A[3, 5] = dt
        self.A[4, 4] = np.exp(-decay_z * dt)
        self.A[5, 5] = np.exp(-decay_pitch * dt)
        self.B = np.zeros((8, 3))
        self.B[1, 0] = kappa * dt
        self.B[3] = ga * dt, gt * dt, gda * dt
        self.H = np.array([
            [-wz * wz, -2 * zz * wz + lever * czt, -lever * wt * wt,
             -2 * lever * zt * wt, 1, lever, 1, 0],
            [0, 0, GRAVITY, 0, 0, 0, 0, GRAVITY],
        ])
        self.D = np.array([
            [kappa + lever * ga, lever * gt, lever * gda],
            [0.0, 0.0, 0.0],
        ])
        if not self.control:
            self.B[:] = 0.0
            self.D[:] = 0.0
        self.Q = np.diag([0, qz, 0, qt, qez, qet, qb, qg]) * dt
        self.R = np.diag([rz, rx])

    def prepare_inputs(self, xr):
        s = _common_signals(xr, self.fs)
        front_accel = np.gradient(s["wheel_speed"][:, :, :2].mean(2), axis=1) * self.fs
        rear_accel = np.gradient(s["wheel_speed"][:, :, 2:].mean(2), axis=1) * self.fs
        delta_accel = _lowpass(front_accel - rear_accel, self.fs)
        return {
            "roll": s["roll"],
            "inputs": np.stack([
                s["speed_dot"], s["torque_front"] + s["torque_rear"], delta_accel
            ], axis=2),
            "observations": np.stack([
                s["vertical_accel"], s["longitudinal_specific_force"]
            ], axis=2),
        }

    def fit_summary(self, result):
        return (f"loss {result.fun:.3f}  wz {self.p[0]:.1f}  wt {self.p[2]:.1f} "
                f"decay=({self.p[10]:.1f},{self.p[11]:.1f})  nfev {result.nfev}")


class DelayedRoadKalmanFilter(KalmanFilter):
    """Current baseline moved from baselines.py and given an explicit name.

    A front equivalent-road acceleration is estimated as a colored state.  Its
    filtered history is replayed at the rear axle after the vehicle has traveled
    one wheelbase.  This is still a reduced sprung-body model, not the 4-DOF
    physical half-car below.
    """

    bounce_state_index = 1
    pitch_state_index = 3

    def __init__(self, *args, front_cg_m=DEFAULT_FRONT_CG_M,
                 rear_cg_m=DEFAULT_REAR_CG_M, **kwargs):
        self.front_cg_m = float(front_cg_m)
        self.rear_cg_m = float(rear_cg_m)
        self.wheelbase_m = self.front_cg_m + self.rear_cg_m
        super().__init__(*args, **kwargs)

    @classmethod
    def initial_parameters(cls):
        dynamics = [8.0, 0.3, 7.8, 0.3, 0.5, 0.7, 5.0, 0.3, 0.0, 0.0]
        noise = np.log([1.0, 1.0, 1.0, 1.0, 1e-4, 1e-4, 0.1, 0.1])
        return np.r_[dynamics, noise]

    def parameter_bounds(self):
        bounds = [
            (0.1, 30.0), (0.01, 2.0), (0.1, 30.0), (0.01, 2.0),
            (0.05, 3.0), (0.05, 2.0), (0.05, 30.0),
            (-20.0, 20.0), (-0.1, 0.1), (-0.1, 0.1),
        ] + [(np.log(1e-10), np.log(1e4))] * 8
        if not self.control:
            for i in (7, 8, 9):
                bounds[i] = (self.p[i], self.p[i])
        return bounds

    def _build_matrices(self):
        wz, zz, wt, zt, lever, road_gain, road_decay, gv, gtf, gtr = self.p[:10]
        qz, qt, qrf, qrr, qb, qg, rz, rx = np.exp(self.p[10:])
        dt = self.dt
        self.A = np.eye(7)
        self.A[0, 1] = dt
        self.A[1, 0] = -wz * wz * dt
        self.A[1, 1] = 1 - 2 * zz * wz * dt
        self.A[1, 4] = dt
        self.A[2, 3] = dt
        self.A[3, 2] = -wt * wt * dt
        self.A[3, 3] = 1 - 2 * zt * wt * dt
        self.A[3, 4] = road_gain * self.front_cg_m * dt
        self.A[4, 4] = np.exp(-road_decay * dt)
        self.B = np.zeros((7, 3))
        self.B[3] = gv * dt, gtf * dt, gtr * dt
        self.H = np.array([
            [-wz * wz, -2 * zz * wz, -lever * wt * wt, -2 * lever * zt * wt,
             1 + lever * road_gain * self.front_cg_m, 1, 0],
            [0, 0, GRAVITY, 0, 0, 0, GRAVITY],
        ])
        self.D = np.array([[lever * gv, lever * gtf, lever * gtr], [0.0, 0.0, 0.0]])
        self.E_rear = np.array([0, 1, 0, -road_gain * self.rear_cg_m, 0, 0, 0]) * dt
        self.J_rear = np.array([1 - lever * road_gain * self.rear_cg_m, 0.0])
        if not self.control:
            self.B[:] = 0.0
            self.D[:] = 0.0
        self.Q = np.diag([0, qz, 0, qt, qrf, qb, qg]) * dt
        rear_map = np.array([0, 1, 0, -road_gain * self.rear_cg_m, 0, 0, 0])
        self.Q += qrr * np.outer(rear_map, rear_map) * dt
        self.R = np.diag([rz, rx])

    def prepare_inputs(self, xr):
        s = _common_signals(xr, self.fs)
        return {
            "roll": s["roll"],
            "inputs": np.stack([
                s["speed_dot"], s["torque_front"], s["torque_rear"]
            ], axis=2),
            "observations": np.stack([
                s["vertical_accel"], s["longitudinal_specific_force"]
            ], axis=2),
            "rear_index": _rear_indices(s["wheel_speed"], self.fs, self.wheelbase_m),
        }

    def _extra_terms(self, prepared, t, state_history):
        j = prepared["rear_index"][:, t]
        rows = np.arange(len(j))
        valid = (j >= 0) & (j < t)
        rear_road = np.zeros(len(j))
        rear_road[valid] = state_history[rows[valid], j[valid], 4]
        return rear_road[:, None] * self.E_rear, rear_road[:, None] * self.J_rear

    def fit_summary(self, result):
        return (f"loss {result.fun:.3f}  wz {self.p[0]:.1f}  wt {self.p[2]:.1f} "
                f"road_gain {self.p[5]:.2f}  road_tau {1 / self.p[6]:.2f}s "
                f"nfev {result.nfev}")


class QuarterCarAugmentedKalmanFilter(KalmanFilter):
    """Document Model 2: 2-DOF quarter-car with augmented road height/rate."""

    fit_output_indices = (0,)
    bounce_state_index = 1
    pitch_state_index = None

    @classmethod
    def initial_parameters(cls):
        physical = [400.0, 55.0, 25_000.0, 2_500.0, 200_000.0, 500.0, 8.0, 0.7]
        noise = np.log([1.0, 1.0, 1.0, 1e-4, 0.1])
        return np.r_[physical, noise]

    def parameter_bounds(self):
        return [
            (150.0, 800.0), (20.0, 150.0),
            (5_000.0, 80_000.0), (100.0, 8_000.0),
            (50_000.0, 500_000.0), (0.0, 5_000.0),
            (0.1, 30.0), (0.01, 2.0),
        ] + [(np.log(1e-12), np.log(1e5))] * 5

    def _build_matrices(self):
        ms, mu, ks, cs, kt, ct, wr, zr = self.p[:8]
        qs, qu, qroad, qbias, rz = np.exp(self.p[8:])
        ac = np.zeros((7, 7))
        ac[0, 1] = 1.0
        ac[1, 0], ac[1, 1] = -ks / ms, -cs / ms
        ac[1, 2], ac[1, 3] = ks / ms, cs / ms
        ac[2, 3] = 1.0
        ac[3, 0], ac[3, 1] = ks / mu, cs / mu
        ac[3, 2], ac[3, 3] = -(ks + kt) / mu, -(cs + ct) / mu
        ac[3, 4], ac[3, 5] = kt / mu, ct / mu
        ac[4, 5] = 1.0
        ac[5, 4], ac[5, 5] = -wr * wr, -2 * zr * wr
        self.A, self.B = _c2d(ac, np.zeros((7, 0)), self.dt)
        self.H = np.array([[-ks / ms, -cs / ms, ks / ms, cs / ms, 0, 0, 1]])
        self.D = np.zeros((1, 0))
        self.Q = np.diag([0, qs, 0, qu, 0, qroad, qbias]) * self.dt
        self.R = np.array([[rz]])

    def prepare_inputs(self, xr):
        s = _common_signals(xr, self.fs)
        n, t = s["roll"].shape
        return {
            "roll": s["roll"],
            "inputs": np.empty((n, t, 0)),
            "observations": s["vertical_accel"][:, :, None],
        }

    def fit_summary(self, result):
        return (f"loss {result.fun:.3f}  ms {self.p[0]:.0f}  mu {self.p[1]:.0f} "
                f"ks {self.p[2]:.0f}  kt {self.p[4]:.0f}  nfev {result.nfev}")


class LinearHalfCarKalmanFilter(KalmanFilter):
    """Document Model 3: physical 4-DOF linear half-car with delayed road height.

    The front road height/rate is augmented in the state.  The rear road height
    is the stored front estimate at the same spatial location.  Masses and
    geometry are fixed by default; suspension/tire parameters are identified.
    """

    bounce_state_index = 4
    pitch_state_index = 5

    def __init__(self, *args, mass_s=1600.0, inertia_y=2500.0,
                 mass_uf=70.0, mass_ur=70.0, front_cg_m=DEFAULT_FRONT_CG_M,
                 rear_cg_m=DEFAULT_REAR_CG_M, cg_height_m=0.55, **kwargs):
        self.mass_s = float(mass_s)
        self.inertia_y = float(inertia_y)
        self.mass_uf = float(mass_uf)
        self.mass_ur = float(mass_ur)
        self.front_cg_m = float(front_cg_m)
        self.rear_cg_m = float(rear_cg_m)
        self.wheelbase_m = self.front_cg_m + self.rear_cg_m
        self.cg_height_m = float(cg_height_m)
        super().__init__(*args, **kwargs)

    @classmethod
    def initial_parameters(cls):
        physical = [25_000.0, 30_000.0, 2_500.0, 2_800.0,
                    200_000.0, 220_000.0, 1.0, 8.0, 0.7, 0.5]
        noise = np.log([1.0, 1.0, 1.0, 1.0, 1.0, 1e-4, 1e-4, 0.1, 0.1])
        return np.r_[physical, noise]

    def parameter_bounds(self):
        bounds = [
            (5_000.0, 100_000.0), (5_000.0, 100_000.0),
            (100.0, 10_000.0), (100.0, 10_000.0),
            (50_000.0, 600_000.0), (50_000.0, 600_000.0),
            (-10.0, 10.0), (0.1, 30.0), (0.01, 2.0), (0.0, 3.0),
        ] + [(np.log(1e-12), np.log(1e6))] * 9
        if not self.control:
            bounds[6] = (self.p[6], self.p[6])
        return bounds

    def _continuous_model(self):
        ksf, ksr, csf, csr, ktf, ktr, torque_gain, wr, zr, lever = self.p[:10]
        a, b = self.front_cg_m, self.rear_cg_m
        ms, iy, muf, mur = self.mass_s, self.inertia_y, self.mass_uf, self.mass_ur

        sf = np.zeros(12)
        sf[[0, 1, 2, 4, 5, 6]] = [ksf, ksf * a, -ksf, csf, csf * a, -csf]
        sr = np.zeros(12)
        sr[[0, 1, 3, 4, 5, 7]] = [ksr, -ksr * b, -ksr, csr, -csr * b, -csr]

        ac = np.zeros((12, 12))
        ac[0, 4] = ac[1, 5] = ac[2, 6] = ac[3, 7] = 1.0
        ac[4] = -(sf + sr) / ms
        ac[5] = (-a * sf + b * sr) / iy
        ac[6] = sf / muf
        ac[6, 2] -= ktf / muf
        ac[6, 8] += ktf / muf
        ac[7] = sr / mur
        ac[7, 3] -= ktr / mur
        ac[8, 9] = 1.0
        ac[9, 8], ac[9, 9] = -wr * wr, -2 * zr * wr

        bc = np.zeros((12, 3))
        bc[5, 0] = -ms * self.cg_height_m / iy
        bc[5, 1:] = torque_gain / iy
        if not self.control:
            bc[:] = 0.0
        ec = np.zeros((12, 1))
        ec[7, 0] = ktr / mur
        return ac, bc, ec, lever

    def _build_matrices(self):
        ac, bc, ec, lever = self._continuous_model()
        qb, qp, quf, qur, qroad, qbz, qbx, rz, rx = np.exp(self.p[10:])
        self.A, combined = _c2d(ac, np.c_[bc, ec], self.dt)
        self.B, self.E_rear = combined[:, :3], combined[:, 3]
        vertical_row = ac[4] + lever * ac[5]
        vertical_row[10] += 1.0
        self.H = np.vstack([
            vertical_row,
            np.eye(1, 12, 1).ravel() * GRAVITY + np.eye(1, 12, 11).ravel(),
        ])
        self.D = np.vstack([bc[4] + lever * bc[5], np.zeros(3)])
        self.J_rear = np.zeros(2)
        self.Q = np.diag([0, 0, 0, 0, qb, qp, quf, qur, 0, qroad, qbz, qbx]) * self.dt
        self.R = np.diag([rz, rx])

    def prepare_inputs(self, xr):
        s = _common_signals(xr, self.fs)
        return {
            "roll": s["roll"],
            "inputs": np.stack([
                s["speed_dot"], s["torque_front"], s["torque_rear"]
            ], axis=2),
            "observations": np.stack([
                s["vertical_accel"], s["longitudinal_specific_force"]
            ], axis=2),
            "rear_index": _rear_indices(s["wheel_speed"], self.fs, self.wheelbase_m),
        }

    def _extra_terms(self, prepared, t, state_history):
        j = prepared["rear_index"][:, t]
        rows = np.arange(len(j))
        valid = (j >= 0) & (j < t)
        rear_height = np.zeros(len(j))
        rear_height[valid] = state_history[rows[valid], j[valid], 8]
        return rear_height[:, None] * self.E_rear, rear_height[:, None] * self.J_rear

    def fit_summary(self, result):
        return (f"loss {result.fun:.3f}  ksf {self.p[0]:.0f}  ksr {self.p[1]:.0f} "
                f"ktf {self.p[4]:.0f}  ktr {self.p[5]:.0f}  nfev {result.nfev}")


class ParametricBumpHalfCarKalmanFilter(KalmanFilter):
    """Linear 4-DOF half-car driven by one cosine-shaped spatial bump.

    Bump height, length, and start position are offline parameters.  Distance is
    centered at each episode's largest vertical-acceleration event so that one
    shared start-position parameter is meaningful across aligned bump windows.
    """

    bounce_state_index = 4
    pitch_state_index = 5

    def __init__(self, *args, mass_s=1600.0, inertia_y=2500.0,
                 mass_uf=70.0, mass_ur=70.0, front_cg_m=DEFAULT_FRONT_CG_M,
                 rear_cg_m=DEFAULT_REAR_CG_M, cg_height_m=0.55, **kwargs):
        self.mass_s = float(mass_s)
        self.inertia_y = float(inertia_y)
        self.mass_uf = float(mass_uf)
        self.mass_ur = float(mass_ur)
        self.front_cg_m = float(front_cg_m)
        self.rear_cg_m = float(rear_cg_m)
        self.wheelbase_m = self.front_cg_m + self.rear_cg_m
        self.cg_height_m = float(cg_height_m)
        super().__init__(*args, **kwargs)

    @classmethod
    def initial_parameters(cls):
        physical = [25_000.0, 30_000.0, 2_500.0, 2_800.0,
                    200_000.0, 220_000.0, 1.0, 0.5,
                    np.log(0.05), np.log(3.0), -1.5]
        noise = np.log([1.0, 1.0, 1.0, 1.0, 1e-4, 1e-4, 0.1, 0.1])
        return np.r_[physical, noise]

    def parameter_bounds(self):
        bounds = [
            (5_000.0, 100_000.0), (5_000.0, 100_000.0),
            (100.0, 10_000.0), (100.0, 10_000.0),
            (50_000.0, 600_000.0), (50_000.0, 600_000.0),
            (-10.0, 10.0), (0.0, 3.0),
            (np.log(1e-4), np.log(0.3)), (np.log(0.2), np.log(20.0)),
            (-10.0, 5.0),
        ] + [(np.log(1e-12), np.log(1e6))] * 8
        if not self.control:
            bounds[6] = (self.p[6], self.p[6])
        return bounds

    def _build_matrices(self):
        ksf, ksr, csf, csr, ktf, ktr, torque_gain, lever = self.p[:8]
        qb, qp, quf, qur, qbz, qbx, rz, rx = np.exp(self.p[11:])
        a, b = self.front_cg_m, self.rear_cg_m
        ms, iy, muf, mur = self.mass_s, self.inertia_y, self.mass_uf, self.mass_ur

        sf = np.zeros(10)
        sf[[0, 1, 2, 4, 5, 6]] = [ksf, ksf * a, -ksf, csf, csf * a, -csf]
        sr = np.zeros(10)
        sr[[0, 1, 3, 4, 5, 7]] = [ksr, -ksr * b, -ksr, csr, -csr * b, -csr]
        ac = np.zeros((10, 10))
        ac[0, 4] = ac[1, 5] = ac[2, 6] = ac[3, 7] = 1.0
        ac[4] = -(sf + sr) / ms
        ac[5] = (-a * sf + b * sr) / iy
        ac[6] = sf / muf
        ac[6, 2] -= ktf / muf
        ac[7] = sr / mur
        ac[7, 3] -= ktr / mur

        bc = np.zeros((10, 3))
        bc[5, 0] = -ms * self.cg_height_m / iy
        bc[5, 1:] = torque_gain / iy
        if not self.control:
            bc[:] = 0.0
        ec = np.zeros((10, 2))
        ec[6, 0] = ktf / muf
        ec[7, 1] = ktr / mur
        self.A, combined = _c2d(ac, np.c_[bc, ec], self.dt)
        self.B, self.E_road = combined[:, :3], combined[:, 3:]

        vertical_row = ac[4] + lever * ac[5]
        vertical_row[8] += 1.0
        self.H = np.vstack([
            vertical_row,
            np.eye(1, 10, 1).ravel() * GRAVITY + np.eye(1, 10, 9).ravel(),
        ])
        self.D = np.vstack([bc[4] + lever * bc[5], np.zeros(3)])
        self.Q = np.diag([0, 0, 0, 0, qb, qp, quf, qur, qbz, qbx]) * self.dt
        self.R = np.diag([rz, rx])

    def prepare_inputs(self, xr):
        s = _common_signals(xr, self.fs)
        distance = np.cumsum(np.maximum(s["speed"], 0.0) / self.fs, axis=1)
        distance -= distance[:, :1]
        peak = np.argmax(np.abs(s["vertical_accel"]), axis=1)
        distance -= distance[np.arange(len(distance)), peak][:, None]
        return {
            "roll": s["roll"],
            "inputs": np.stack([
                s["speed_dot"], s["torque_front"], s["torque_rear"]
            ], axis=2),
            "observations": np.stack([
                s["vertical_accel"], s["longitudinal_specific_force"]
            ], axis=2),
            "position": distance,
        }

    def _bump_height(self, position):
        height, length, start = np.exp(self.p[8]), np.exp(self.p[9]), self.p[10]
        phase = position - start
        inside = (phase >= 0.0) & (phase <= length)
        profile = 0.5 * height * (1.0 - np.cos(2.0 * np.pi * phase / length))
        return np.where(inside, profile, 0.0)

    def _extra_terms(self, prepared, t, state_history):
        position = prepared["position"][:, t]
        roads = np.stack([
            self._bump_height(position),
            self._bump_height(position - self.wheelbase_m),
        ], axis=1)
        return roads @ self.E_road.T, np.zeros((len(position), self.H.shape[0]))

    def fit_summary(self, result):
        return (f"loss {result.fun:.3f}  bump_h {np.exp(self.p[8]):.3f}m "
                f"bump_L {np.exp(self.p[9]):.2f}m  bump_s0 {self.p[10]:.2f}m "
                f"nfev {result.nfev}")


class ParametricBumpHalfCarRTSKalmanFilter(ParametricBumpHalfCarKalmanFilter):
    """Parametric-bump linear half-car followed by an RTS backward pass."""

    use_smoother = True


class LinearHalfCarRTSKalmanFilter(LinearHalfCarKalmanFilter):
    """Linear half-car followed by a conditional RTS backward pass."""

    use_smoother = True


class NonlinearHalfCarEKF(LinearHalfCarKalmanFilter):
    """Document Model 4 candidate: nonlinear half-car EKF.

    Cubic suspension stiffness and smooth torque saturation are added to the
    4-DOF model.  The delayed front-road estimate remains the road prior.  This
    implementation intentionally fixes masses and geometry offline instead of
    putting every mechanical parameter into the online EKF state.
    """

    default_fit_size = 40
    default_maxfev = 300

    @classmethod
    def initial_parameters(cls):
        linear = LinearHalfCarKalmanFilter.initial_parameters()
        return np.r_[linear[:10], 0.0, 0.0, linear[10:]]

    def parameter_bounds(self):
        base = [
            (5_000.0, 100_000.0), (5_000.0, 100_000.0),
            (100.0, 10_000.0), (100.0, 10_000.0),
            (50_000.0, 600_000.0), (50_000.0, 600_000.0),
            (-10.0, 10.0), (0.1, 30.0), (0.01, 2.0), (0.0, 3.0),
            (0.0, 1e9), (0.0, 1e9),
        ] + [(np.log(1e-12), np.log(1e6))] * 9
        if not self.control:
            base[6] = (self.p[6], self.p[6])
        return base

    def _build_matrices(self):
        # Dummy linear matrices satisfy the shared interface; the EKF below
        # uses state-dependent Jacobians and per-episode covariance matrices.
        qb, qp, quf, qur, qroad, qbz, qbx, rz, rx = np.exp(self.p[12:])
        self.Q = np.diag([0, 0, 0, 0, qb, qp, quf, qur, 0, qroad, qbz, qbx]) * self.dt
        self.R = np.diag([rz, rx])
        self.A = np.eye(12)
        self.B = np.zeros((12, 3))
        self.H = np.zeros((2, 12))
        self.D = np.zeros((2, 3))

    def _forces_and_jacobian(self, x):
        ksf, ksr, csf, csr = self.p[:4]
        k3f, k3r = self.p[10:12]
        a, b = self.front_cg_m, self.rear_cg_m
        df = x[:, 0] + a * x[:, 1] - x[:, 2]
        dr = x[:, 0] - b * x[:, 1] - x[:, 3]
        ddf = x[:, 4] + a * x[:, 5] - x[:, 6]
        ddr = x[:, 4] - b * x[:, 5] - x[:, 7]
        fsf = ksf * df + k3f * df ** 3 + csf * ddf
        fsr = ksr * dr + k3r * dr ** 3 + csr * ddr

        jsf = np.zeros((len(x), 12))
        jsr = np.zeros_like(jsf)
        kef, ker = ksf + 3 * k3f * df ** 2, ksr + 3 * k3r * dr ** 2
        jsf[:, [0, 1, 2]] = np.stack([kef, a * kef, -kef], axis=1)
        jsf[:, [4, 5, 6]] = np.array([1, a, -1]) * csf
        jsr[:, [0, 1, 3]] = np.stack([ker, -b * ker, -ker], axis=1)
        jsr[:, [4, 5, 7]] = np.array([1, -b, -1]) * csr
        return fsf, fsr, jsf, jsr

    def _dynamics_and_jacobian(self, x, u, rear_height):
        ktf, ktr, torque_gain, wr, zr = self.p[4:9]
        fsf, fsr, jsf, jsr = self._forces_and_jacobian(x)
        ms, iy, muf, mur = self.mass_s, self.inertia_y, self.mass_uf, self.mass_ur
        a, b = self.front_cg_m, self.rear_cg_m
        torque = 2_000.0 * np.tanh((u[:, 1] + u[:, 2]) / 2_000.0)

        derivative = np.zeros_like(x)
        derivative[:, :4] = x[:, 4:8]
        derivative[:, 4] = -(fsf + fsr) / ms
        control_moment = (-ms * self.cg_height_m * u[:, 0] + torque_gain * torque
                          if self.control else 0.0)
        derivative[:, 5] = (-a * fsf + b * fsr + control_moment) / iy
        derivative[:, 6] = (fsf - ktf * (x[:, 2] - x[:, 8])) / muf
        derivative[:, 7] = (fsr - ktr * (x[:, 3] - rear_height)) / mur
        derivative[:, 8] = x[:, 9]
        derivative[:, 9] = -wr * wr * x[:, 8] - 2 * zr * wr * x[:, 9]

        jac = np.zeros((len(x), 12, 12))
        jac[:, 0, 4] = jac[:, 1, 5] = jac[:, 2, 6] = jac[:, 3, 7] = 1.0
        jac[:, 4] = -(jsf + jsr) / ms
        jac[:, 5] = (-a * jsf + b * jsr) / iy
        jac[:, 6] = jsf / muf
        jac[:, 6, 2] -= ktf / muf
        jac[:, 6, 8] += ktf / muf
        jac[:, 7] = jsr / mur
        jac[:, 7, 3] -= ktr / mur
        jac[:, 8, 9] = 1.0
        jac[:, 9, 8], jac[:, 9, 9] = -wr * wr, -2 * zr * wr
        return derivative, jac

    def _observation_and_jacobian(self, x, u, rear_height):
        derivative, jac = self._dynamics_and_jacobian(x, u, rear_height)
        lever = self.p[9]
        observation = np.stack([
            derivative[:, 4] + lever * derivative[:, 5] + x[:, 10],
            GRAVITY * np.sin(x[:, 1]) + x[:, 11],
        ], axis=1)
        h = np.zeros((len(x), 2, 12))
        h[:, 0] = jac[:, 4] + lever * jac[:, 5]
        h[:, 0, 10] += 1.0
        h[:, 1, 1] = GRAVITY * np.cos(x[:, 1])
        h[:, 1, 11] = 1.0
        return observation, h

    def _filter_states(self, prepared):
        observations, inputs = prepared["observations"], prepared["inputs"]
        n, tmax = observations.shape[:2]
        state = np.zeros((n, 12))
        initial_scale = 100.0 if self.warmup else 1.0
        covariance = np.repeat((np.eye(12) * initial_scale)[None], n, axis=0)
        states = np.zeros((n, tmax, 12))
        rows = np.arange(n)
        eye = np.eye(12)[None]
        for t in range(tmax):
            j = prepared["rear_index"][:, t]
            valid = (j >= 0) & (j < t)
            rear_height = np.zeros(n)
            rear_height[valid] = states[rows[valid], j[valid], 8]
            derivative, continuous_jac = self._dynamics_and_jacobian(
                state, inputs[:, t], rear_height
            )
            predicted = state + self.dt * derivative
            transition_jac = eye + self.dt * continuous_jac
            pp = transition_jac @ covariance @ np.swapaxes(transition_jac, 1, 2) + self.Q
            expected, h = self._observation_and_jacobian(predicted, inputs[:, t], rear_height)
            ph = pp @ np.swapaxes(h, 1, 2)
            innovation_cov = h @ ph + self.R
            gain = np.linalg.solve(innovation_cov, np.swapaxes(ph, 1, 2)).swapaxes(1, 2)
            state = predicted + np.einsum("nij,nj->ni", gain, observations[:, t] - expected)
            ikh = eye - gain @ h
            covariance = ikh @ pp @ np.swapaxes(ikh, 1, 2) + gain @ self.R @ np.swapaxes(gain, 1, 2)
            states[:, t] = state
        return states


KALMAN_FILTERS = {
    "reduced_kf": ReducedModalKalmanFilter,
    "delayed_road_kf": DelayedRoadKalmanFilter,
    "quarter_car_kf": QuarterCarAugmentedKalmanFilter,
    "parametric_halfcar_kf": ParametricBumpHalfCarKalmanFilter,
    "parametric_halfcar_rts": ParametricBumpHalfCarRTSKalmanFilter,
    "linear_halfcar_kf": LinearHalfCarKalmanFilter,
    "linear_halfcar_rts": LinearHalfCarRTSKalmanFilter,
    "nonlinear_halfcar_ekf": NonlinearHalfCarEKF,
}


__all__ = [
    "KalmanFilter",
    "ReducedModalKalmanFilter",
    "DelayedRoadKalmanFilter",
    "QuarterCarAugmentedKalmanFilter",
    "ParametricBumpHalfCarKalmanFilter",
    "ParametricBumpHalfCarRTSKalmanFilter",
    "LinearHalfCarKalmanFilter",
    "LinearHalfCarRTSKalmanFilter",
    "NonlinearHalfCarEKF",
    "KALMAN_FILTERS",
]
