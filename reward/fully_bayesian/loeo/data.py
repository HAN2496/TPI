"""Data sources with one interface: {evaluator: (X [N,T,C], y [N])}, plus channel list and fs.

Real data goes through loader.Dataset / loader.View exactly as run_fully_bayesian.py.
Synthetic data is generated from the hierarchical model itself so the ground truth
(which features are common / evaluator-specific / inactive) is known.
"""
from __future__ import annotations

import numpy as np

from ..model import sigmoid
from . import bank as B


def eligible(data, min_labels, min_per_class):
    out = {}
    for name, (X, y) in data.items():
        y = np.asarray(y)
        if len(y) >= min_labels and y.sum() >= min_per_class and (len(y) - y.sum()) >= min_per_class:
            out[name] = (X, y)
    return out


def load_real(cfg):
    from loader import Dataset, View
    view = View(features=tuple(cfg.channels), around=tuple(cfg.around),
                downsample=cfg.downsample, smooth=tuple(cfg.smooth) if cfg.smooth else None)
    ds = Dataset(cfg.dataset_root)
    names = list(cfg.evaluators) if cfg.evaluators else ds.names
    data = {}
    for n in names:
        X, y = view(ds[n])
        if len(y):
            data[n] = (np.asarray(X, np.float32), np.asarray(y, np.int64))
    return data, list(view.cols), float(view.fs)


# ----------------------------------------------------------------------------- synthetic
def _bump_windows(rng, n, T, fs, n_channels):
    """Damped-oscillation windows with episode-specific severity and per-channel gains."""
    t = np.arange(T) / fs
    X = np.zeros((n, T, n_channels), np.float32)
    for i in range(n):
        sev = rng.lognormal(0.0, 0.5)                          # bump severity
        f = rng.uniform(1.0, 3.0)                              # body mode frequency
        zeta = rng.uniform(0.1, 0.4)
        t0 = rng.uniform(0.3, 0.7) * t[-1] * 0.5
        env = np.exp(-zeta * 2 * np.pi * f * np.clip(t - t0, 0, None)) * (t >= t0)
        base = sev * env * np.sin(2 * np.pi * f * (t - t0))
        for c in range(n_channels):
            gain = rng.lognormal(0.0, 0.3)
            phase = rng.uniform(-0.5, 0.5)
            shaped = sev * env * np.sin(2 * np.pi * f * (t - t0) + phase) * gain
            noise = rng.normal(0, 0.05 * (1 + 0.5 * c), T)
            X[i, :, c] = (0.5 * base + shaped + noise).astype(np.float32)
    return X


def load_synthetic(cfg, seed=None):
    """Generate evaluators from the hierarchical model; returns (data, channels, fs, truth)."""
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    channels = list(cfg.channels)
    fs = 100.0 / cfg.downsample
    T = int(round((cfg.around[1] - cfg.around[0]) * fs))
    U = cfg.syn_n_evaluators
    sizes = rng.integers(cfg.syn_min_episodes, cfg.syn_max_episodes + 1, size=U)
    sizes[0] = cfg.syn_max_episodes                           # one long-stream evaluator
    Xs = [_bump_windows(rng, int(n), T, fs, len(channels)) for n in sizes]

    # features on the *generating* bank (pruned once on all synthetic episodes)
    bank = B.full_bank(channels)
    pruned, _ = B.prune_bank(Xs, channels, fs, bank, rho_max=cfg.rho_max)
    phi = B.make_pipeline(pruned, channels, fs).fit(Xs, [None] * U)
    Zs = [phi.transform(X) for X in Xs]
    names = list(phi.feature_names)
    d = len(names)

    # ground truth: k_common common features, k_ind evaluator-specific, rest inactive
    sel = rng.permutation(np.arange(1, d))                     # skip bias
    common = sel[:cfg.syn_k_common]
    individual = sel[cfg.syn_k_common:cfg.syn_k_common + cfg.syn_k_individual]
    mu = np.zeros(d); sd = np.zeros(d)
    mu[0] = 0.2
    mu[common] = rng.choice([-1, 1], size=len(common)) * rng.uniform(0.8, 1.5, size=len(common))
    sd[common] = 0.25
    sd[individual] = 1.2
    thetas = mu[None, :] + sd[None, :] * rng.standard_normal((U, d))
    data = {}
    for u in range(U):
        p = sigmoid(Zs[u] @ thetas[u])
        y = (rng.random(len(p)) < p).astype(np.int64)
        data[f"S{u + 1:02d}"] = (Xs[u], y)
    truth = {
        "feature_names": names,
        "common": [names[j] for j in common],
        "individual": [names[j] for j in individual],
        "inactive": [names[j] for j in range(1, d) if j not in set(common) | set(individual)],
        "mu": mu.tolist(), "sd": sd.tolist(),
    }
    return data, channels, fs, truth
