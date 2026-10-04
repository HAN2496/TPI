"""Data access for the CoPL LOEO driver: same loader and eligibility rules as run_loeo.py."""
from __future__ import annotations

from itertools import combinations

import numpy as np

from reward.fully_bayesian.loeo import data as D


def load_data(cfg):
    """{evaluator: (X [N,T,C], y [N])}, channel names, fs. Population restriction as in run_loeo."""
    if cfg.data == "synthetic":
        data, channels, fs, _ = D.load_synthetic(cfg)
    else:
        data, channels, fs = D.load_real(cfg)
    if cfg.pop_min_labels or cfg.pop_min_per_class:
        data = D.eligible(data, cfg.pop_min_labels, cfg.pop_min_per_class)
    return data, list(channels), float(fs)


def fold_names(cfg, data):
    elig = D.eligible(data, cfg.min_labels, cfg.min_per_class)
    names = list(elig) if not cfg.folds else [n for n in cfg.folds if n in elig]
    return names, elig


def channel_indices(channels, subset):
    """Column indices of `subset` (channel names) inside `channels`; () means all."""
    if not subset:
        return list(range(len(channels)))
    missing = [c for c in subset if c not in channels]
    if missing:
        raise ValueError(f"channels {missing} not loaded (loaded: {channels})")
    return [channels.index(c) for c in subset]


def select_channels(data, idx):
    return {n: (X[:, :, idx], y) for n, (X, y) in data.items()}


def channel_sets(channels, mode):
    """Named channel subsets for the encoder / channel studies."""
    chans = list(channels)
    sets = {"full": tuple(chans)}
    if mode == "full":
        return sets
    for c in chans:
        sets[f"without_{c}"] = tuple(x for x in chans if x != c)
    for c in chans:
        sets[f"only_{c}"] = (c,)
    imu = tuple(c for c in chans if c.startswith("IMU_"))
    if imu and len(imu) < len(chans):
        sets["imu_only"] = imu
    if mode == "all":
        for r in range(2, len(chans)):
            for comb in combinations(chans, r):
                key = "+".join(comb)
                sets.setdefault(key, tuple(comb))
    return sets


def population_split(data, held):
    pop_names = [n for n in data if n != held]
    return pop_names, {n: data[n] for n in pop_names}
