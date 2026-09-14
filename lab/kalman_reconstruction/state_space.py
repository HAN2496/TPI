from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm
from scipy.signal import butter, correlate, correlation_lags, sosfilt


GRAVITY = 9.81


@dataclass
class StateSpace:
    A: np.ndarray
    H: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    P: np.ndarray
    B: np.ndarray | None = None
    D: np.ndarray | None = None
    G: np.ndarray | None = None
    Qu: np.ndarray | None = None
    E: np.ndarray | None = None

    def filter(self, y, u=None, x0=None, smooth=False, road=False, extra=None, innovations=False):
        y = np.asarray(y)
        if y.ndim == 2:
            y = y[..., None]
        steps, n, m = y.shape[1], len(self.A), self.H.shape[0]
        p, eye = self.P.copy(), np.eye(n)
        gains = np.empty((steps, n, m))
        pred_cov = np.empty((steps, n, n)) if smooth else None
        post_cov = np.empty_like(pred_cov) if smooth else None
        road_gain = np.empty((steps, self.G.shape[1], n)) if road else None
        cov = np.empty((steps, m, m)) if innovations else None
        nu = np.empty((len(y), steps, m)) if innovations else None
        for t in range(steps):
            p = self.A @ p @ self.A.T + self.Q
            if smooth:
                pred_cov[t] = p
            cov_y = self.H @ p @ self.H.T + self.R
            if innovations:
                cov[t] = cov_y
            gains[t] = k = np.linalg.solve(cov_y, self.H @ p).T
            if road:
                road_gain[t] = np.linalg.solve(p, self.G @ self.Qu).T
            c = eye - k @ self.H
            p = c @ p @ c.T + k @ self.R @ k.T
            if smooth:
                post_cov[t] = p
        state = np.empty((len(y), steps, n))
        inputs = np.empty((len(y), steps, self.G.shape[1])) if road else None
        s = np.zeros((len(y), n)) if x0 is None else np.array(x0, float)
        for t, k in enumerate(gains):
            pred = s @ self.A.T if u is None else s @ self.A.T + u[:, t] @ self.B.T
            if extra is not None:
                pred = pred + extra(t, state)
            innovation = y[:, t] - pred @ self.H.T
            if u is not None and self.D is not None:
                innovation -= u[:, t] @ self.D.T
            if innovations:
                nu[:, t] = innovation
            s = pred + innovation @ k.T
            state[:, t] = s
            if road:
                inputs[:, t] = (s - pred) @ road_gain[t].T
        if smooth:
            for t in range(steps - 2, -1, -1):
                j = np.linalg.solve(pred_cov[t + 1], self.A @ post_cov[t]).T
                pred = state[:, t] @ self.A.T if u is None else state[:, t] @ self.A.T + u[:, t + 1] @ self.B.T
                state[:, t] += (state[:, t + 1] - pred) @ j.T
        if innovations:
            return state, nu, cov
        return (state, inputs) if road else state


def discretize(f, qc, dt):
    n = len(f)
    block = np.zeros((2 * n, 2 * n))
    block[:n, :n], block[:n, n:], block[n:, n:] = f, qc, -f.T
    value = expm(block * dt)
    a = value[:n, :n]
    q = value[:n, n:] @ a.T
    return a, (q + q.T) / 2


def discretize_input(f, b, dt):
    n, m = len(f), b.shape[1]
    block = np.zeros((n + m, n + m))
    block[:n, :n], block[:n, n:] = f, b
    value = expm(block * dt)
    return value[:n, :n], value[:n, n:]


def highpass(x, cutoff, fs):
    a = 1 / (1 + 2 * np.pi * cutoff / fs)
    y = np.zeros_like(x)
    for t in range(1, x.shape[1]):
        y[:, t] = a * (y[:, t - 1] + x[:, t] - x[:, t - 1])
    return y


def bandpass(x, low, high, fs):
    return sosfilt(butter(2, (low, high), btype="bandpass", fs=fs, output="sos"), x, axis=1)


def calibrate(x, y, gain=None):
    if gain is None:
        xc, yc = x - x.mean(), y - y.mean()
        gain = np.sum(xc * yc) / (np.sum(xc * xc) + 1e-12)
    return gain, y.mean() - gain * x.mean()


def metrics(true, pred, fs):
    pc, tc = pred - pred.mean(1, keepdims=True), true - true.mean(1, keepdims=True)
    corr = np.sum(pc * tc, 1) / np.sqrt(np.sum(pc * pc, 1) * np.sum(tc * tc, 1) + 1e-12)
    rmse = np.sqrt(np.mean((pred - true) ** 2, 1))
    lags = correlation_lags(true.shape[1], true.shape[1])
    keep = np.abs(lags) <= round(0.5 * fs)
    lag = [abs(lags[keep][np.argmax(correlate(a - a.mean(), b - b.mean(), method="fft")[keep])]) / fs
           for a, b in zip(true, pred)]
    return corr, rmse, np.asarray(lag)


def innovation_metrics(nu, cov, burn=100):
    # Särkkä & Svensson (2023) Thm 16.9: energy = -log p(y | theta) per sample, first `burn` steps dropped
    nu, cov = nu[:, burn:], cov[burn:]
    m = cov.shape[-1]
    nis = np.einsum("nti,tij,ntj->nt", nu, np.linalg.inv(cov), nu)
    energy = .5 * (nis.mean() + np.linalg.slogdet(2 * np.pi * cov)[1].mean())
    white = nu / np.sqrt(np.einsum("tii->ti", cov))
    wc = white - white.mean(1, keepdims=True)
    acf = lambda lag: np.mean(np.sum(wc[:, lag:] * wc[:, :-lag], 1) / np.sum(wc * wc, 1))
    return dict(energy=energy, nis=nis.mean() / m, acf1=acf(1), acf10=acf(10))


def waveform_metrics(true, pred, fs):
    def corr(a, b):
        ac, bc = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
        return np.sum(ac * bc, 1) / np.sqrt(np.sum(ac * ac, 1) * np.sum(bc * bc, 1) + 1e-12)

    return {
        "corr": corr(true, pred),
        "corr 0.3-1 Hz": corr(bandpass(true, .3, 1, fs), bandpass(pred, .3, 1, fs)),
        "corr 1-3 Hz": corr(bandpass(true, 1, 3, fs), bandpass(pred, 1, 3, fs)),
        "amplitude ratio": pred.std(1) / true.std(1),
        "error-amplitude corr": corr(np.abs(true), np.abs(pred - true)),
    }
