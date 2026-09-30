"""Feature and sensor selection under the held-out ELPD criterion (paper Section V, claude_notes/02).

* projection-predictive forward path on the fold's reference model gives the candidate order
* every candidate is a column subset of the fold's full pruned pipeline; it is refit with a
  reduced Gibbs chain and scored on the held-out evaluator (MLPD only, `light` evaluation)
* the one-standard-error rule picks the smallest size within 1 SE of the full bank
* sensors: exhaustive non-empty channel subsets, same refit-and-score
* stability: Nogueira et al. (2018) index of the per-fold selected sets
"""
from __future__ import annotations

import time
from itertools import combinations

import numpy as np

from .. import projpred
from . import bank as B
from . import protocol as P


def size_grid(d_features, requested):
    ks = sorted(set(int(k) for k in requested if 1 <= int(k) < d_features))
    if d_features not in ks:
        ks.append(d_features)
    return ks


def projpred_path(cfg, phi, pop, pop_data, unit="feature"):
    """Forward order of units on the reference model (bias excluded from counts)."""
    return projpred.select(cfg, phi, pop, pop_data, unit=unit)


def path_subsets(order, ks):
    """Map each requested size k (number of non-bias columns) to the feature names on the path."""
    out = {}
    for row in order[1:]:
        k = row["n_features"] - 1
        if k in ks:
            out[k] = [f for f in row["selected_features"] if f != "bias"]
    return out


def bank_from_features(bank, feature_names):
    """Restrict a {channel: [stats]} bank to the given 'channel__stat' names (keeps bank order)."""
    keep = set(feature_names)
    sub = {}
    for ch, stats in bank.items():
        s = [st for st in stats if f"{ch}__{st}" in keep]
        if s:
            sub[ch] = s
    return sub


def features_of_channels(phi, channels_subset):
    keep = set(channels_subset)
    return [f for f, g in zip(phi.feature_names, phi.groups) if f != "bias" and g in keep]


def refit_and_score(cfg, phi_full, Z_pop, ys, names, Z_held, y_held, feature_names, budgets, seed):
    """Refit the hierarchy on a column subset of the full pipeline and score the held-out evaluator.

    Z_pop / Z_held are the *full* design matrices (already transformed once per fold); the subset
    is taken by column selection, which is exact because standardization is per column.
    Returns (metrics_by_budget, lpd_by_budget, timing, feature_names_with_bias).
    """
    tic = time.time()
    sub = B.ColumnSubset(phi_full, feature_names)
    Zs = [sub.transform_Z(Z) for Z in Z_pop]
    pop, stats = P.fit_population_Z(cfg, Zs, ys, sub.feature_names, names, sub.groups,
                                    spike_slab=cfg.sel_spike_slab, reduced=True, seed=seed)
    tic_ev = time.time()
    res, lpd, info = P.evaluate_proposed(cfg, pop, sub.transform_Z(Z_held), np.asarray(y_held), seed,
                                         budgets=budgets, light=True)
    t_eval = time.time() - tic_ev
    total = time.time() - tic
    timing = dict(features=total - stats["seconds"] - t_eval, gibbs=stats["seconds"], eval=t_eval, total=total)
    return res, lpd, timing, sub.feature_names


def sensor_subsets(channels, min_size=1):
    subs = []
    for r in range(min_size, len(channels) + 1):
        subs.extend(list(c) for c in combinations(channels, r))
    return subs


def one_se_rule(sizes, mean_by_size, se_diff_by_size, full_size):
    """Smallest size whose mean is within one SE (of the paired difference) of the full model."""
    full = mean_by_size[full_size]
    for k in sorted(sizes):
        if k == full_size:
            return k
        if np.isfinite(mean_by_size.get(k, np.nan)) and (full - mean_by_size[k]) <= se_diff_by_size.get(k, np.inf):
            return k
    return full_size


def nogueira_stability(sets, universe):
    """Nogueira, Sechidis, Brown (2018) stability of a collection of selected feature sets."""
    universe = list(universe)
    idx = {f: j for j, f in enumerate(universe)}
    Zm = np.zeros((len(sets), len(universe)))
    for i, s in enumerate(sets):
        for f in s:
            if f in idx:
                Zm[i, idx[f]] = 1.0
    M, d = Zm.shape
    if M < 2:
        return float("nan")
    p = Zm.mean(0)
    kbar = Zm.sum(1).mean()
    if kbar <= 0 or kbar >= d:
        return float("nan")
    var = (M / (M - 1)) * p * (1 - p)
    return float(1.0 - var.mean() / ((kbar / d) * (1 - kbar / d)))


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else float("nan")
