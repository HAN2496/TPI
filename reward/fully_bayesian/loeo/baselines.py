"""Baselines for the LOEO protocol (paper Section VI-C).

Every baseline exposes  fit_population(Zs, ys)  and  predict(Z_ctx, y_ctx, Z_hold) -> probs (N,)
so the driver can evaluate them at each budget exactly like the proposed model.
EB-MAP additionally returns particles so the same uncertainty metrics apply.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

from ..model import sigmoid


def _stack(Zs, ys):
    return np.concatenate(Zs, 0), np.concatenate(ys, 0)


class Pooled:
    """One L2 logistic model on all population labels; context appended at each budget."""
    name = "pooled"

    def __init__(self, C=1.0):
        self.C = C

    def fit_population(self, Zs, ys):
        self.Zp, self.yp = _stack(Zs, ys)
        return self

    def predict(self, Z_ctx, y_ctx, Z_hold):
        Z = np.concatenate([self.Zp, Z_ctx], 0) if len(y_ctx) else self.Zp
        y = np.concatenate([self.yp, y_ctx], 0) if len(y_ctx) else self.yp
        lr = LogisticRegression(C=self.C, fit_intercept=False, max_iter=2000).fit(Z, y)
        return lr.predict_proba(Z_hold)[:, 1]


class Indep:
    """L2 logistic model on the held-out evaluator's context only (no transfer)."""
    name = "indep"

    def __init__(self, C=1.0):
        self.C = C

    def fit_population(self, Zs, ys):
        self.prior = float(np.mean(np.concatenate(ys)))
        return self

    def predict(self, Z_ctx, y_ctx, Z_hold):
        n = len(y_ctx)
        if n == 0:
            return np.full(len(Z_hold), np.nan)                 # undefined at cold start
        if len(np.unique(y_ctx)) < 2:                             # one class: Laplace-smoothed rate
            return np.full(len(Z_hold), (y_ctx.sum() + 0.5) / (n + 1.0))
        lr = LogisticRegression(C=self.C, fit_intercept=False, max_iter=2000).fit(Z_ctx, y_ctx)
        return lr.predict_proba(Z_hold)[:, 1]


class EBMAP:
    """Empirical-Bayes partial pooling with a point-estimated population and Laplace posterior.

    Population: per-evaluator L2 logistic coefficients -> mu_hat (mean), Sigma_hat (covariance + ridge).
    New evaluator: MAP of the logistic likelihood with prior N(mu_hat, Sigma_hat); predictive
    by M draws from the Laplace approximation N(theta_map, H^-1).  At t=0 the draws come
    from the prior itself.
    """
    name = "ebmap"

    def __init__(self, C=1.0, ridge=1e-2, M=400, seed=0):
        self.C, self.ridge, self.M, self.seed = C, ridge, M, seed

    def fit_population(self, Zs, ys):
        coefs = []
        for Z, y in zip(Zs, ys):
            if len(np.unique(y)) < 2:
                continue
            coefs.append(LogisticRegression(C=self.C, fit_intercept=False, max_iter=2000)
                         .fit(Z, y).coef_[0])
        coefs = np.asarray(coefs)
        d = coefs.shape[1]
        self.mu = coefs.mean(0)
        cov = np.cov(coefs.T) if len(coefs) > 1 else np.eye(d)
        self.Sigma = cov + self.ridge * np.eye(d)
        self.Sinv = np.linalg.inv(self.Sigma)
        return self

    def _map(self, Z, y):
        kappa = y - 0.5
        def f(th):
            eta = Z @ th
            nll = -np.sum(y * eta - np.logaddexp(0.0, eta))
            r = th - self.mu
            return nll + 0.5 * r @ self.Sinv @ r
        def g(th):
            p = sigmoid(Z @ th)
            return -Z.T @ (y - p) + self.Sinv @ (th - self.mu)
        res = minimize(f, self.mu.copy(), jac=g, method="L-BFGS-B")
        th = res.x
        p = sigmoid(Z @ th)
        H = (Z.T * (p * (1 - p))) @ Z + self.Sinv
        return th, H

    def particles(self, Z_ctx, y_ctx):
        rng = np.random.default_rng(self.seed)
        if len(y_ctx) == 0:
            L = np.linalg.cholesky(self.Sigma)
            return self.mu[None, :] + rng.standard_normal((self.M, len(self.mu))) @ L.T
        th, H = self._map(Z_ctx, y_ctx)
        L = np.linalg.cholesky(np.linalg.inv(H) + 1e-9 * np.eye(len(th)))
        return th[None, :] + rng.standard_normal((self.M, len(th))) @ L.T

    def predict_particles(self, Z_ctx, y_ctx, Z_hold):
        return sigmoid(self.particles(Z_ctx, y_ctx) @ Z_hold.T)          # (M, N)

    def predict(self, Z_ctx, y_ctx, Z_hold):
        return self.predict_particles(Z_ctx, y_ctx, Z_hold).mean(0)


class GBT:
    """Gradient-boosted trees on pooled episodes with the evaluator id as a categorical input."""
    name = "gbt"

    def __init__(self, seed=0):
        self.seed = seed

    def fit_population(self, Zs, ys):
        self.Zs, self.ys = list(Zs), list(ys)
        return self

    def predict(self, Z_ctx, y_ctx, Z_hold):
        from sklearn.ensemble import HistGradientBoostingClassifier
        Zs = self.Zs + ([Z_ctx] if len(y_ctx) else [])
        ys = self.ys + ([y_ctx] if len(y_ctx) else [])
        ids = np.concatenate([np.full(len(y), i) for i, y in enumerate(ys)])
        new_id = len(self.Zs) if len(y_ctx) else -1               # -1 = unseen category -> treated as missing
        Z = np.column_stack([np.concatenate(Zs, 0)[:, 1:], ids])   # drop the constant bias column
        y = np.concatenate(ys, 0)
        cat = np.zeros(Z.shape[1], bool); cat[-1] = True
        model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3,
                                               categorical_features=cat, random_state=self.seed)
        model.fit(Z, y)
        Zh = np.column_stack([Z_hold[:, 1:], np.full(len(Z_hold), new_id if new_id >= 0 else np.nan)])
        return model.predict_proba(Zh)[:, 1]


def make_baselines(cfg):
    out = [Pooled(cfg.baseline_C), Indep(cfg.baseline_C),
           EBMAP(cfg.baseline_C, M=cfg.ebmap_M, seed=cfg.seed)]
    if cfg.gbt:
        out.append(GBT(seed=cfg.seed))
    return out
