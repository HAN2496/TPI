from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov
from scipy.optimize import minimize


GRAVITY = 9.81
OU_START = np.array([1.3, .3, -5., 0., -3., np.log(2.)])


@dataclass
class OUStateSpace:
    A: np.ndarray
    H: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    P: np.ndarray

    def filter(self, measurement):
        measurement = np.asarray(measurement)
        if measurement.ndim == 2:
            measurement = measurement[..., None]
        steps, size = measurement.shape[1], len(self.A)
        covariance, eye = self.P.copy(), np.eye(size)
        gains = np.empty((steps, size, self.H.shape[0]))
        for step in range(steps):
            covariance = self.A @ covariance @ self.A.T + self.Q
            gain = np.linalg.solve(
                self.H @ covariance @ self.H.T + self.R,
                self.H @ covariance,
            ).T
            correction = eye - gain @ self.H
            covariance = correction @ covariance @ correction.T + gain @ self.R @ gain.T
            gains[step] = gain

        state = np.empty((len(measurement), steps, size))
        posterior = np.zeros((len(measurement), size))
        for step, gain in enumerate(gains):
            prior = posterior @ self.A.T
            posterior = prior + (measurement[:, step] - prior @ self.H.T) @ gain.T
            state[:, step] = posterior
        return state


@dataclass
class OUFit:
    parameters: np.ndarray
    gain: float
    offset: float
    loss: float
    success: bool
    iterations: int
    evaluations: int

    def state_space(self, fs):
        return build_ou(self.parameters, fs)

    def states(self, az_g, fs):
        return estimate_ou(az_g, self.parameters, fs)

    def predict(self, az_g, fs):
        return self.gain * self.states(az_g, fs)[..., 1] + self.offset


def discretize_system(F, Qc, dt):
    size = len(F)
    block = np.zeros((2 * size, 2 * size))
    block[:size, :size] = F
    block[:size, size:] = Qc
    block[size:, size:] = -F.T
    value = expm(block * dt)
    A = value[:size, :size]
    Q = value[:size, size:] @ A.T
    return A, (Q + Q.T) / 2


def ou_bounds(f_max=8., lambda_max=100.):
    if f_max <= .5 or lambda_max <= .05:
        raise ValueError("f_max must exceed 0.5 and lambda_max must exceed 0.05")
    return [
        (.5, f_max),
        (.05, 1.5),
        (-12., 3.),
        (-12., 5.),
        (-12., 4.),
        (np.log(.05), np.log(lambda_max)),
    ]


def build_ou(parameters, fs):
    frequency, damping, log_qv, log_qf, log_r, log_lambda = np.asarray(parameters)
    omega, rate = 2 * np.pi * frequency, np.exp(log_lambda)
    latent = np.array([[-rate]])
    latent_noise = 2 * np.exp(log_qf) * rate
    F = np.array([
        [0., 1., 0.],
        [-omega**2, -2 * damping * omega, 1.],
        [0., 0., -rate],
    ])
    Qc = np.diag([1e-10, np.exp(log_qv), latent_noise])
    A, Q = discretize_system(F, Qc, 1 / fs)
    P = np.eye(3)
    P[2:, 2:] = solve_continuous_lyapunov(latent, -latent_noise)
    return OUStateSpace(A, F[1:2], Q, np.array([[np.exp(log_r)]]), P)


def estimate_ou(az_g, parameters, fs):
    measurement = ((np.asarray(az_g) - 1) * GRAVITY)[..., None]
    return build_ou(parameters, fs).filter(measurement)


def affine_calibration(estimate, target):
    estimate, target = np.asarray(estimate), np.asarray(target)
    centered_estimate = estimate - estimate.mean()
    centered_target = target - target.mean()
    gain = np.sum(centered_estimate * centered_target) / (
        np.sum(centered_estimate * centered_estimate) + 1e-12
    )
    return float(gain), float(target.mean() - gain * estimate.mean())


def fit_ou(
    az_g,
    bounce,
    fs,
    max_episodes=300,
    f_max=6.,
    lambda_max=100.,
    maxiter=100,
    start=None,
):
    az_g, bounce = np.asarray(az_g), np.asarray(bounce)
    if az_g.shape != bounce.shape or az_g.ndim != 2:
        raise ValueError("az_g and bounce must have matching (episodes, time) shapes")
    index = np.linspace(0, len(az_g) - 1, min(max_episodes, len(az_g))).astype(int)
    fit_az, fit_bounce = az_g[index], bounce[index]

    def objective(parameters):
        velocity = estimate_ou(fit_az, parameters, fs)[..., 1]
        gain, offset = affine_calibration(velocity, fit_bounce)
        error = gain * velocity + offset - fit_bounce
        return np.sqrt(np.mean(error**2)) / fit_bounce.std()

    result = minimize(
        objective,
        OU_START.copy() if start is None else np.asarray(start, dtype=float),
        method="Powell",
        bounds=ou_bounds(f_max, lambda_max),
        options={"maxiter": maxiter, "ftol": 1e-6, "xtol": 1e-5},
    )
    velocity = estimate_ou(az_g, result.x, fs)[..., 1]
    gain, offset = affine_calibration(velocity, bounce)
    return OUFit(
        result.x,
        gain,
        offset,
        float(result.fun),
        bool(result.success),
        int(result.nit),
        int(result.nfev),
    )
