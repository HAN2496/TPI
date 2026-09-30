"""Feature and sensor selection under the held-out ELPD criterion (paper Section V, claude_notes/02).

* projection-predictive forward path on the fold's reference model gives the candidate order
* every candidate size is refit (reduced Gibbs) and scored on the held-out evaluator (MLPD)
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
    order = projpred.select(cfg, phi, pop, pop_data, unit=unit)
    return order        # list of rows with 'added', 'selected_features', 'cols', 'projection_kl', 'captured'


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


def refit_and_score(cfg, sub_bank, channels, fs, pop_data, names, held, budgets, seed):
    """Fit the hierarchical model on `sub_bank` for the population and score the held-out evaluator.

    Returns (metrics_by_budget, lpd_by_budget, timing, feature_names) where timing is a dict of
    wall-clock seconds: 'features' (extraction + transform), 'gibbs' (reduced chain), 'eval'
    (held-out protocol incl. metrics) and 'total'.
    """
    tic = time.time()
    phi = B.make_pipeline(sub_bank, channels, fs).fit([pop_data[n][0] for n in names], None)
    t_feat = time.time() - tic
    pop, _, _, stats = P.fit_population(cfg, phi, pop_data, names,
                                        spike_slab=cfg.sel_spike_slab, reduced=True, seed=seed)
    tic_ev = time.time()
    Z = phi.transform(held[0]).astype(np.float64)
    res, lpd, info = P.evaluate_proposed(cfg, pop, Z, np.asarray(held[1]), seed, budgets=budgets)
    t_eval = time.time() - tic_ev
    total = time.time() - tic
    timing = dict(features=total - stats["seconds"] - t_eval, gibbs=stats["seconds"], eval=t_eval, total=total)
    return res, lpd, timing, phi.feature_names


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
