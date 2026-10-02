"""LOEO folds, budgets, offsets and per-evaluator evaluation (paper Section VI-B).

For a held-out evaluator the episodes are split in recorded order into a context
stream (first half) and a fixed holdout (second half).  Budget t reveals the first
t context episodes starting at chronological offset o; results are averaged over
offsets.  The proposed model is warm-started across increasing budgets.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

from ..model import Population
from . import metrics as M


def split_stream(n, frac=0.5):
    s = int(n * frac)
    return np.arange(s), np.arange(s, n)


def budget_grid(cfg, n_ctx):
    ts = [t for t in cfg.budgets if t <= n_ctx]
    if cfg.include_final and n_ctx not in ts:
        ts.append(n_ctx)
    return sorted(set(ts))


def offsets_for(cfg, n_ctx, t_max):
    """Chronological start offsets so that o + t_max <= n_ctx."""
    room = n_ctx - t_max
    if room <= 0 or cfg.n_offsets <= 1:
        return [0]
    step = max(1, room // (cfg.n_offsets - 1))
    return sorted(set(min(o, room) for o in range(0, room + 1, step)))[: cfg.n_offsets]


def _budgets_and_offsets(cfg, ts, n_ctx):
    """Keep budgets that the context can supply; offsets are sized for the largest *regular*
    budget (a budget equal to n_ctx is the 'final' point and is always evaluated at offset 0)."""
    ts = sorted(set(int(t) for t in ts if t <= n_ctx))
    t_reg = max([t for t in ts if 0 < t < n_ctx], default=0)
    offs = offsets_for(cfg, n_ctx, t_reg) if t_reg > 0 else [0]
    return ts, offs


def _nanmean(vals):
    a = np.asarray(vals, float)
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")


def population_cfg(cfg, spike_slab=None, reduced=False, seed=None):
    return SimpleNamespace(
        n_samples=cfg.sel_n_samples if reduced else cfg.n_samples,
        n_burnin=cfg.sel_n_burnin if reduced else cfg.n_burnin,
        thin=1, niw_kappa0=cfg.niw_kappa0, niw_nu0=cfg.niw_nu0,
        niw_lambda0_scale=cfg.niw_lambda0_scale, newuser_n_iters=cfg.newuser_n_iters,
        eps_var=None, spike_slab=cfg.spike_slab if spike_slab is None else spike_slab,
        spike_slab_unit=cfg.spike_slab_unit, spike_slab_a=cfg.spike_slab_a,
        spike_slab_b=cfg.spike_slab_b, seed=cfg.seed if seed is None else seed,
    )


def fit_population_Z(cfg, Zs, ys, feature_names, names, groups, spike_slab=None, reduced=False, seed=None):
    """Fit the hierarchy on already-transformed design matrices."""
    pop = Population(population_cfg(cfg, spike_slab, reduced, seed))
    tic = time.time()
    stats = pop.fit([np.asarray(Z, np.float64) for Z in Zs], [np.asarray(y, float) for y in ys],
                    list(feature_names), list(names), list(groups))
    stats["seconds"] = time.time() - tic
    return pop, stats


def fit_population(cfg, phi, pop_data, names, spike_slab=None, reduced=False, seed=None):
    Zs = [phi.transform(pop_data[n][0]).astype(np.float64) for n in names]
    ys = [np.asarray(pop_data[n][1], float) for n in names]
    pop, stats = fit_population_Z(cfg, Zs, ys, phi.feature_names, names, phi.groups, spike_slab, reduced, seed)
    return pop, Zs, ys, stats


def evaluate_proposed(cfg, pop, Z, y, seed, budgets=None, light=False):
    """Proposed model on one held-out evaluator. Returns {budget: metrics} and lpd vectors.

    light=True skips the bootstrap reliability interval (selection only reads MLPD).
    """
    ctx_idx, hold_idx = split_stream(len(y), cfg.ctx_frac)
    Z_ctx, y_ctx, Z_hold, y_hold = Z[ctx_idx], y[ctx_idx], Z[hold_idx], y[hold_idx]
    n_ctx = len(y_ctx)
    ts, offs = _budgets_and_offsets(cfg, budgets if budgets is not None else budget_grid(cfg, n_ctx), n_ctx)
    per_budget = {t: [] for t in ts}
    lpds = {t: [] for t in ts}
    rng = np.random.default_rng(seed)
    for o in offs:
        star = pop.new_user(seed=seed + o)
        for t in ts:
            if t == 0:
                P = star.predict(Z_hold)[2]
            else:
                lo, hi = o, min(o + t, n_ctx)
                if hi - lo < t and o > 0:                        # this offset cannot supply t labels
                    continue
                star.fit(Z_ctx[lo:hi], y_ctx[lo:hi], rng=rng)    # warm start from previous budget
                P = star.predict(Z_hold)[2]
            m, lpd = M.summarize(y_hold, P, seed=seed, light=light, width_max=getattr(cfg, "reliable_w_max", 0.15))
            m["t"], m["offset"] = int(t), int(o)
            per_budget[t].append(m)
            lpds[t].append(lpd)
    out = {}
    for t in ts:
        if not per_budget[t]:
            continue
        keys = [k for k in per_budget[t][0] if isinstance(per_budget[t][0][k], (int, float, bool))]
        agg = {k: _nanmean([mm[k] for mm in per_budget[t]]) for k in keys}
        agg["n_offsets"] = len(per_budget[t])
        agg["reliable_frac"] = float(np.mean([mm["reliable"] for mm in per_budget[t]]))
        out[t] = agg
    lpd_mean = {t: np.mean(np.stack(v), axis=0) for t, v in lpds.items() if v}
    return out, lpd_mean, dict(n_ctx=n_ctx, n_hold=len(y_hold), budgets=ts, offsets=offs)


def evaluate_baseline(cfg, model, Z, y, budgets, particles=False):
    ctx_idx, hold_idx = split_stream(len(y), cfg.ctx_frac)
    Z_ctx, y_ctx, Z_hold, y_hold = Z[ctx_idx], y[ctx_idx], Z[hold_idx], y[hold_idx]
    n_ctx = len(y_ctx)
    ts, offs = _budgets_and_offsets(cfg, budgets, n_ctx)
    out, lpds = {}, {}
    for t in ts:
        rows, lv = [], []
        for o in offs:
            lo, hi = (o, min(o + t, n_ctx)) if t > 0 else (0, 0)
            if t > 0 and hi - lo < t and o > 0:
                continue
            if particles:
                P = model.predict_particles(Z_ctx[lo:hi], y_ctx[lo:hi], Z_hold)
                m, lpd = M.summarize(y_hold, P, seed=cfg.seed, width_max=getattr(cfg, "reliable_w_max", 0.15))
            else:
                p = model.predict(Z_ctx[lo:hi], y_ctx[lo:hi], Z_hold)
                if np.any(np.isnan(p)):
                    continue
                m, lpd = M.summarize_point(y_hold, p)
            rows.append(m); lv.append(lpd)
        if rows:
            keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float, bool))]
            out[t] = {k: _nanmean([r[k] for r in rows]) for k in keys}
            out[t]["n_offsets"] = len(rows)
            if particles:                                    # same key as the proposed model
                out[t]["reliable_frac"] = float(np.mean([r["reliable"] for r in rows]))
            lpds[t] = np.mean(np.stack(lv), axis=0)
    return out, lpds


def prequential(cfg, pop, Z, y, seed, max_steps=None):
    """One-step-ahead log score along the whole personal stream (Dawid's prequential score)."""
    n = len(y) if max_steps is None else min(len(y), max_steps)
    star = pop.new_user(seed=seed)
    rng = np.random.default_rng(seed)
    scores = np.empty(n)
    for i in range(n):
        p = star.predict(Z[i:i + 1])[0][0]
        scores[i] = M.pointwise_lpd(y[i:i + 1], [p])[0]
        star.fit(Z[:i + 1], y[:i + 1], rng=rng)                 # warm start, prefix 1..i
    return scores
