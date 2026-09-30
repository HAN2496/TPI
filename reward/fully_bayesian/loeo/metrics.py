"""Metrics for the LOEO protocol (paper Section VI-D, claude_notes/01).

All functions take y in {0,1} and either a point probability vector p (N,) or a
particle matrix P (M, N) of posterior draws of the judgment probability.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

_EPS = 1e-12


def pointwise_lpd(y, p):
    p = np.clip(np.asarray(p, float), _EPS, 1 - _EPS)
    return np.where(np.asarray(y) == 1, np.log(p), np.log1p(-p))


def sum_se(v):
    """(sum, se) with se = sqrt(n * var) (Vehtari et al. 2017, eq. 23)."""
    v = np.asarray(v, float)
    if v.size < 2:
        return float(v.sum()), float("nan")
    return float(v.sum()), float(np.sqrt(v.size * v.var(ddof=1)))


def has_both(y):
    return len(np.unique(np.asarray(y))) == 2


def auroc(y, p):
    return float(roc_auc_score(y, p)) if has_both(y) else float("nan")


def auprc(y, p):
    return float(average_precision_score(y, p)) if has_both(y) else float("nan")


def brier(y, p):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def ece(y, p, n_bins=10):
    """Expected calibration error with equal-width bins on p."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    out = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            out += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(out)


def calibration_slope_intercept(y, p):
    """Weak calibration (Van Calster): logistic regression of y on logit(p)."""
    from sklearn.linear_model import LogisticRegression
    if not has_both(y):
        return float("nan"), float("nan")
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    x = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(x, np.asarray(y, int))
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def decompose(P):
    """Epistemic Var_m(p) and aleatoric E_m[p(1-p)] per episode from particles (M, N)."""
    P = np.asarray(P, float)
    epi = P.var(axis=0, ddof=1) if P.shape[0] > 1 else np.zeros(P.shape[1])
    ale = (P * (1 - P)).mean(axis=0)
    return epi, ale


def correctness_auroc(y, p_mean, score):
    """AUROC of `score` (e.g. epistemic variance) for predicting a *wrong* decision."""
    wrong = ((np.asarray(p_mean) > 0.5).astype(int) != np.asarray(y).astype(int)).astype(int)
    return float(roc_auc_score(wrong, score)) if has_both(wrong) else float("nan")


def risk_coverage(y, p_mean, score):
    """Selective prediction: rank by ascending `score` (most confident first).

    Returns AURC, excess AURC (vs. oracle ordering), and the risk-coverage curve.
    """
    y = np.asarray(y).astype(int); p_mean = np.asarray(p_mean, float); score = np.asarray(score, float)
    err = ((p_mean > 0.5).astype(int) != y).astype(float)
    n = len(y)
    order = np.argsort(score, kind="stable")
    risk = np.cumsum(err[order]) / np.arange(1, n + 1)
    cov = np.arange(1, n + 1) / n
    aurc = float(risk.mean())
    oracle = np.cumsum(np.sort(err)) / np.arange(1, n + 1)
    return aurc, float(aurc - oracle.mean()), cov, risk


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.size < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    return float(spearmanr(a, b).correlation)


def summarize(y, P, seed=0, trust_K=600):
    """All holdout metrics for particle predictions P (M, N)."""
    from core.metrics import auroc_trust_interval
    P = np.asarray(P, float)
    p = P.mean(axis=0)
    epi, ale = decompose(P)
    lpd = pointwise_lpd(y, p)
    aurc, eaurc, _, _ = risk_coverage(y, p, epi)
    slope, intercept = calibration_slope_intercept(y, p)
    trust = auroc_trust_interval(np.asarray(y), P, seed, K=trust_K)
    return {
        "n": int(len(y)), "n_pos": int(np.sum(y)),
        "mlpd": float(lpd.mean()), "elpd": float(lpd.sum()),
        "brier": brier(y, p), "auroc": auroc(y, p), "auprc": auprc(y, p),
        "ece": ece(y, p), "cal_slope": slope, "cal_intercept": intercept,
        "epi_mean": float(epi.mean()), "ale_mean": float(ale.mean()),
        "epi_ale_spearman": spearman(epi, ale),
        "correctness_auroc_epi": correctness_auroc(y, p, epi),
        "correctness_auroc_total": correctness_auroc(y, p, -np.abs(p - 0.5)),
        "aurc_epi": aurc, "eaurc_epi": eaurc,
        "auroc_ci_lo": trust["ci_lo"], "auroc_ci_hi": trust["ci_hi"],
        "reliable": bool(trust["trustworthy"]),
    }, lpd


def summarize_point(y, p):
    """Metrics for a point-probability baseline (no particles)."""
    p = np.asarray(p, float)
    lpd = pointwise_lpd(y, p)
    slope, intercept = calibration_slope_intercept(y, p)
    aurc, eaurc, _, _ = risk_coverage(y, p, -np.abs(p - 0.5))
    return {
        "n": int(len(y)), "n_pos": int(np.sum(y)),
        "mlpd": float(lpd.mean()), "elpd": float(lpd.sum()),
        "brier": brier(y, p), "auroc": auroc(y, p), "auprc": auprc(y, p),
        "ece": ece(y, p), "cal_slope": slope, "cal_intercept": intercept,
        "correctness_auroc_total": correctness_auroc(y, p, -np.abs(p - 0.5)),
        "aurc_total": aurc, "eaurc_total": eaurc,
    }, lpd


def macro(values):
    """Mean and evaluator-level SE of a list (nan-aware)."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if v.size == 0:
        return float("nan"), float("nan"), 0
    se = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else float("nan")
    return float(v.mean()), se, int(v.size)
