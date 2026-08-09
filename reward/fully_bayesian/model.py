from collections import OrderedDict

import numpy as np
from polyagamma import random_polyagamma
from scipy.linalg import solve_triangular
from scipy.stats import invwishart


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logistic_loglik(theta, Z, y):
    eta = Z @ theta
    return float(np.sum(y * eta - np.logaddexp(0.0, eta)))


def _bernoulli_probability(log_odds):
    if log_odds >= 0:
        return 1.0 / (1.0 + np.exp(-min(log_odds, 700.0)))
    e = np.exp(max(log_odds, -700.0))
    return e / (1.0 + e)


class UserReward:
    """Posterior reward samples for a train or new user.

    theta stores the effective coefficient gamma * slab_theta used for
    prediction. slab_theta is updated during personalization while the learned
    population-level gamma draw remains fixed.
    """

    def __init__(self, theta, pop, name=None, slab_theta=None, gamma=None):
        self.pop = pop
        self.name = name
        self.slab_theta = theta.copy() if slab_theta is None else slab_theta
        self.gamma = np.ones_like(theta) if gamma is None else gamma
        self.theta = self.slab_theta * self.gamma

    def reward(self, Z):
        return self.theta @ np.asarray(Z).T

    def predict(self, Z):
        probs = sigmoid(self.reward(Z))
        return probs.mean(axis=0), probs.std(axis=0), probs

    def clone(self):
        return UserReward(
            self.theta.copy(), self.pop, self.name,
            slab_theta=self.slab_theta.copy(), gamma=self.gamma.copy(),
        )
 
    def fit(self, Z, y, n_iters=None):
        """PG mini-Gibbs adaptation with population gamma fixed per draw."""
        pop = self.pop
        n_iters = n_iters or pop.newuser_n_iters
        kappa = y - 0.5
        N = len(y)
        Sinv, Sinv_mu = pop._cache()
        M, d = self.slab_theta.shape
        Ij = pop.jitter * np.eye(d)
        rng = np.random.default_rng()

        eps = np.zeros((M, N))
        if not pop.spike_slab:
            Z_T = Z.T
            for _ in range(n_iters):
                mean_eta = self.slab_theta @ Z_T
                omega = random_polyagamma(
                    h=1.0, z=mean_eta + eps, random_state=rng
                )
                if pop.use_eps:
                    prec = omega + pop.inv_eps
                    eps = (
                        (kappa[None] - omega * mean_eta) / prec
                        + rng.standard_normal((M, N)) / np.sqrt(prec)
                    )
                A = (Z_T[None] * omega[:, None, :]) @ Z + Sinv + Ij[None]
                b = ((kappa[None] - omega * eps) @ Z + Sinv_mu)[..., None]
                L = np.linalg.cholesky(A)
                yb = np.linalg.solve(L, b)
                noise = rng.standard_normal((M, d, 1))
                res = np.linalg.solve(
                    np.transpose(L, (0, 2, 1)),
                    np.concatenate([yb, noise], axis=-1),
                )
                self.slab_theta = res[..., 0] + res[..., 1]
            self.theta = self.slab_theta
            return self

        Z_gamma = Z[None, :, :] * self.gamma[:, None, :]
        for _ in range(n_iters):
            mean_eta = np.einsum("mnd,md->mn", Z_gamma, self.slab_theta)
            omega = random_polyagamma(
                h=1.0, z=mean_eta + eps, random_state=rng
            )
            if pop.use_eps:
                prec = omega + pop.inv_eps
                eps = (
                    (kappa[None] - omega * mean_eta) / prec
                    + rng.standard_normal((M, N)) / np.sqrt(prec)
                )
            A = (
                np.einsum("mni,mn,mnj->mij", Z_gamma, omega, Z_gamma)
                + Sinv + Ij[None]
            )
            b = (
                np.einsum("mni,mn->mi", Z_gamma, kappa[None] - omega * eps)
                + Sinv_mu
            )[..., None]
            L = np.linalg.cholesky(A)
            yb = np.linalg.solve(L, b)
            noise = rng.standard_normal((M, d, 1))
            res = np.linalg.solve(
                np.transpose(L, (0, 2, 1)),
                np.concatenate([yb, noise], axis=-1),
            )
            self.slab_theta = res[..., 0] + res[..., 1]
        self.theta = self.slab_theta * self.gamma
        return self

    def __repr__(self):
        return f"UserReward({self.name}, M={self.theta.shape[0]}, d={self.theta.shape[1]})"


class Population:
    """Hierarchical PG-Gibbs model with optional group spike-and-slab."""

    def __init__(self, cfg):
        self.n_samples = cfg.n_samples
        self.n_burnin = cfg.n_burnin
        self.thin = cfg.thin
        self.niw_kappa0 = cfg.niw_kappa0
        self.niw_nu0 = cfg.niw_nu0
        self.niw_lambda0_scale = cfg.niw_lambda0_scale
        self.newuser_n_iters = cfg.newuser_n_iters
        self.eps_var = cfg.eps_var
        self.use_eps = cfg.eps_var is not None
        self.inv_eps = 1.0 / cfg.eps_var if self.use_eps else 0.0
        self.spike_slab = cfg.spike_slab
        self.spike_slab_unit = cfg.spike_slab_unit
        self.spike_slab_a = cfg.spike_slab_a
        self.spike_slab_b = cfg.spike_slab_b
        self.jitter = 1e-6
        self.random_state = cfg.seed
        self._user_cache = None

    def _setup_gamma(self, feature_names, feature_groups):
        if self.spike_slab_unit not in ("sensor", "feature"):
            raise ValueError("spike_slab_unit must be 'sensor' or 'feature'")
        if self.spike_slab_a <= 0 or self.spike_slab_b <= 0:
            raise ValueError("spike_slab_a and spike_slab_b must be positive")

        bias = {
            j for j, (name, group) in enumerate(zip(feature_names, feature_groups))
            if name == "bias" or group == "bias"
        }
        units = OrderedDict()
        for j, (name, group) in enumerate(zip(feature_names, feature_groups)):
            if j in bias:
                continue
            label = group if self.spike_slab_unit == "sensor" else name
            units.setdefault(str(label), []).append(j)

        self.gamma_unit_names = list(units)
        self.gamma_unit_columns = [np.asarray(cols, dtype=int) for cols in units.values()]
        self.gamma_fixed_columns = np.asarray(sorted(bias), dtype=int)

    def _expand_gamma(self, unit_gamma):
        gamma = np.zeros(self.d, dtype=float)
        gamma[self.gamma_fixed_columns] = 1.0
        for value, cols in zip(unit_gamma, self.gamma_unit_columns):
            gamma[cols] = value
        return gamma

    def _sample_gamma(
        self, Z_list, kappa_list, omegas, eps_list, slab_theta,
        unit_gamma, feature_gamma, pi, rng,
    ):
        eta_list = [
            Z @ (feature_gamma * slab_theta[u])
            for u, Z in enumerate(Z_list)
        ]
        pi_safe = np.clip(pi, 1e-12, 1.0 - 1e-12)
        prior_log_odds = np.log(pi_safe) - np.log1p(-pi_safe)

        for k in rng.permutation(len(unit_gamma)):
            cols = self.gamma_unit_columns[k]
            current = unit_gamma[k]
            contributions = [
                Z[:, cols] @ slab_theta[u, cols]
                for u, Z in enumerate(Z_list)
            ]
            log_bayes_factor = 0.0
            bases = []
            for u, contribution in enumerate(contributions):
                base = eta_list[u] - current * contribution
                bases.append(base)
                h0 = base + eps_list[u]
                log_bayes_factor += np.sum(
                    kappa_list[u] * contribution
                    - 0.5 * omegas[u]
                    * ((h0 + contribution) ** 2 - h0 ** 2)
                )

            probability = _bernoulli_probability(
                prior_log_odds + log_bayes_factor
            )
            new_value = float(rng.random() < probability)
            unit_gamma[k] = new_value
            for u, contribution in enumerate(contributions):
                eta_list[u] = bases[u] + new_value * contribution

        feature_gamma = self._expand_gamma(unit_gamma)
        return unit_gamma, feature_gamma

    def fit(self, Zs, ys, feature_names, names, feature_groups=None):
        rng = np.random.default_rng(self.random_state)
        np.random.seed(self.random_state)

        valid = [
            (n, Z, y) for n, Z, y in zip(names, Zs, ys)
            if len(np.unique(y)) >= 2
        ]
        assert valid, "No train users with both positive and negative labels"
        self.user_names = [v[0] for v in valid]
        Z_list = [np.asarray(v[1], dtype=float) for v in valid]
        y_list = [np.asarray(v[2], dtype=float) for v in valid]
        kappa_list = [y - 0.5 for y in y_list]
        U = len(valid)
        d = Z_list[0].shape[1]

        self.feature_names = list(feature_names)
        self.feature_groups = (
            list(feature_names) if feature_groups is None else list(feature_groups)
        )
        if len(self.feature_names) != d or len(self.feature_groups) != d:
            raise ValueError("feature names, groups, and design columns must agree")
        self.d, self.U = d, U
        self.m0 = np.zeros(d)
        self.niw_nu0 = (
            float(d + 2) if self.niw_nu0 is None else float(self.niw_nu0)
        )
        self.Lambda0 = self.niw_lambda0_scale * np.eye(d)
        Ij = self.jitter * np.eye(d)
        self._setup_gamma(self.feature_names, self.feature_groups)

        slab_theta = rng.standard_normal((U, d)) * 0.1
        slab_Sigma = np.eye(d)
        slab_mu = np.zeros(d)
        eps_list = [np.zeros(len(y)) for y in y_list]
        unit_gamma = np.ones(len(self.gamma_unit_names), dtype=float)
        feature_gamma = self._expand_gamma(unit_gamma)
        pi = self.spike_slab_a / (self.spike_slab_a + self.spike_slab_b)

        n_total = self.n_burnin + self.n_samples * self.thin
        slab_mu_samples = np.zeros((self.n_samples, d))
        slab_Sigma_samples = np.zeros((self.n_samples, d, d))
        slab_theta_samples = np.zeros((self.n_samples, U, d))
        gamma_samples = np.ones((self.n_samples, d))
        gamma_unit_samples = np.ones((self.n_samples, len(unit_gamma)))
        pi_samples = np.ones(self.n_samples)
        tr_loglik = np.zeros(n_total)
        tr_mu = np.zeros(n_total)
        tr_sigma = np.zeros(n_total)
        tr_theta = np.zeros((n_total, U))
        tr_gamma = np.full(n_total, len(unit_gamma), dtype=float)
        tr_pi = np.ones(n_total)

        save_idx = 0
        for it in range(n_total):
            effective_theta = slab_theta * feature_gamma[None, :]
            mean_eta_list = [
                Z @ effective_theta[u] for u, Z in enumerate(Z_list)
            ]
            omegas = [
                random_polyagamma(
                    h=1.0, z=mean_eta_list[u] + eps_list[u], random_state=rng
                )
                for u in range(U)
            ]

            if self.use_eps:
                for u in range(U):
                    prec = omegas[u] + self.inv_eps
                    mean = (
                        kappa_list[u] - omegas[u] * mean_eta_list[u]
                    ) / prec
                    eps_list[u] = (
                        mean + rng.standard_normal(len(prec)) / np.sqrt(prec)
                    )

            if self.spike_slab:
                unit_gamma, feature_gamma = self._sample_gamma(
                    Z_list, kappa_list, omegas, eps_list, slab_theta,
                    unit_gamma, feature_gamma, pi, rng,
                )
                included = int(unit_gamma.sum())
                pi = rng.beta(
                    self.spike_slab_a + included,
                    self.spike_slab_b + len(unit_gamma) - included,
                )

            slab_Sigma_inv = np.linalg.inv(slab_Sigma + Ij)
            slab_Sigma_inv_mu = slab_Sigma_inv @ slab_mu
            for u in range(U):
                Z_gamma = Z_list[u] * feature_gamma[None, :]
                omega = omegas[u]
                A = (Z_gamma.T * omega) @ Z_gamma + slab_Sigma_inv
                L = np.linalg.cholesky(A + Ij)
                rhs = (
                    Z_gamma.T @ (kappa_list[u] - omega * eps_list[u])
                    + slab_Sigma_inv_mu
                )
                yb = solve_triangular(L, rhs, lower=True)
                m_post = solve_triangular(L.T, yb, lower=False)
                slab_theta[u] = m_post + solve_triangular(
                    L.T, rng.standard_normal(d), lower=False
                )

            theta_bar = slab_theta.mean(axis=0)
            diff = slab_theta - theta_bar
            kappa_n = self.niw_kappa0 + U
            nu_n = self.niw_nu0 + U
            d0 = theta_bar - self.m0
            Lambda_n = (
                self.Lambda0 + diff.T @ diff
                + (self.niw_kappa0 * U / kappa_n) * np.outer(d0, d0)
            )
            m_n = (
                self.niw_kappa0 * self.m0 + U * theta_bar
            ) / kappa_n
            slab_Sigma = np.atleast_2d(
                invwishart.rvs(df=nu_n, scale=Lambda_n, random_state=rng)
            )
            slab_mu = (
                m_n
                + np.linalg.cholesky(slab_Sigma / kappa_n + Ij)
                @ rng.standard_normal(d)
            )

            effective_theta = slab_theta * feature_gamma[None, :]
            tr_loglik[it] = sum(
                logistic_loglik(effective_theta[u], Z_list[u], y_list[u])
                for u in range(U)
            )
            tr_mu[it] = np.linalg.norm(slab_mu * feature_gamma)
            tr_sigma[it] = np.trace(
                slab_Sigma * np.outer(feature_gamma, feature_gamma)
            )
            tr_theta[it] = np.linalg.norm(effective_theta, axis=1)
            tr_gamma[it] = unit_gamma.sum()
            tr_pi[it] = pi

            if (
                it >= self.n_burnin
                and (it - self.n_burnin) % self.thin == 0
                and save_idx < self.n_samples
            ):
                slab_mu_samples[save_idx] = slab_mu
                slab_Sigma_samples[save_idx] = slab_Sigma
                slab_theta_samples[save_idx] = slab_theta
                gamma_samples[save_idx] = feature_gamma
                gamma_unit_samples[save_idx] = unit_gamma
                pi_samples[save_idx] = pi
                save_idx += 1

        self.slab_mu_samples = slab_mu_samples
        self.slab_Sigma_samples = slab_Sigma_samples
        self.slab_theta_samples = slab_theta_samples
        self.gamma_samples = gamma_samples
        self.gamma_unit_samples = gamma_unit_samples
        self.pi_samples = pi_samples
        self.trace = {
            "loglik": tr_loglik,
            "mu_norm": tr_mu,
            "sigma_trace": tr_sigma,
            "theta_norm": tr_theta,
            "gamma_count": tr_gamma,
            "pi": tr_pi,
        }
        self._finalize()
        return {
            "gibbs/loglik_final": float(tr_loglik[-1]),
            "gibbs/n_samples": float(self.n_samples),
            "gibbs/n_burnin": float(self.n_burnin),
            "gibbs/n_train_users": float(U),
            "gibbs/spike_slab": float(self.spike_slab),
            "gibbs/included_units_mean": float(
                self.gamma_unit_samples.sum(axis=1).mean()
            ),
            "gibbs/pi_mean": float(self.pi_samples.mean()),
        }

    def _finalize(self):
        gamma_outer = (
            self.gamma_samples[:, :, None] * self.gamma_samples[:, None, :]
        )
        self.mu_samples = self.slab_mu_samples * self.gamma_samples
        self.Sigma_samples = self.slab_Sigma_samples * gamma_outer
        self.theta_samples = (
            self.slab_theta_samples * self.gamma_samples[:, None, :]
        )
        self.gamma_pip = self.gamma_samples.mean(axis=0)
        self.gamma_unit_pip = self.gamma_unit_samples.mean(axis=0)
        self.pi_bar = float(self.pi_samples.mean())
        self.slab_mu_bar = self.slab_mu_samples.mean(axis=0)
        self.slab_Sigma_bar = self.slab_Sigma_samples.mean(axis=0)
        self.mu_bar = self.mu_samples.mean(axis=0)
        self.Sigma_bar = self.Sigma_samples.mean(axis=0)
        self.theta_means = self.theta_samples.mean(axis=0)
        self.theta_stds = self.theta_samples.std(axis=0)

    def user(self, name):
        index = self.user_names.index(name)
        return UserReward(
            self.theta_samples[:, index], self, name,
            slab_theta=self.slab_theta_samples[:, index],
            gamma=self.gamma_samples,
        )

    def new_user(self, name="*"):
        rng = np.random.default_rng(self.random_state + 1)
        M, d = self.n_samples, self.d
        Ij = self.jitter * np.eye(d)
        slab_theta = np.empty((M, d))
        for m in range(M):
            L = np.linalg.cholesky(self.slab_Sigma_samples[m] + Ij)
            slab_theta[m] = (
                self.slab_mu_samples[m] + L @ rng.standard_normal(d)
            )
        return UserReward(
            slab_theta * self.gamma_samples,
            self,
            name,
            slab_theta=slab_theta,
            gamma=self.gamma_samples.copy(),
        )

    def _cache(self):
        if self._user_cache is None:
            Ij = self.jitter * np.eye(self.d)
            Sinv = np.linalg.inv(self.slab_Sigma_samples + Ij[None])
            Sinv_mu = np.einsum(
                "mij,mj->mi", Sinv, self.slab_mu_samples
            )
            self._user_cache = (Sinv, Sinv_mu)
        return self._user_cache

    def state_dict(self):
        keys = (
            "n_samples", "n_burnin", "thin", "niw_kappa0", "niw_nu0",
            "niw_lambda0_scale", "newuser_n_iters", "eps_var", "jitter",
            "random_state", "spike_slab", "spike_slab_unit", "spike_slab_a",
            "spike_slab_b", "feature_names", "feature_groups", "user_names",
            "gamma_unit_names", "gamma_unit_columns", "gamma_fixed_columns",
            "slab_mu_samples", "slab_Sigma_samples", "slab_theta_samples",
            "gamma_samples", "gamma_unit_samples", "pi_samples", "trace",
            "mu_samples", "Sigma_samples", "theta_samples",
        )
        return {key: getattr(self, key) for key in keys}

    @classmethod
    def from_state_dict(cls, state):
        obj = cls.__new__(cls)
        obj.__dict__.update(state)
        if "spike_slab" not in state:
            obj.spike_slab = False
            obj.spike_slab_unit = "feature"
            obj.spike_slab_a = 1.0
            obj.spike_slab_b = 1.0
        if "feature_groups" not in state:
            obj.feature_groups = list(obj.feature_names)
        if "slab_mu_samples" not in state:
            obj.slab_mu_samples = obj.mu_samples.copy()
            obj.slab_Sigma_samples = obj.Sigma_samples.copy()
            obj.slab_theta_samples = obj.theta_samples.copy()
        obj.d = obj.slab_mu_samples.shape[1]
        obj.U = obj.slab_theta_samples.shape[1]
        if "gamma_samples" not in state:
            obj.gamma_samples = np.ones((obj.n_samples, obj.d))
            obj.gamma_unit_samples = np.empty((obj.n_samples, 0))
            obj.pi_samples = np.ones(obj.n_samples)
            obj.gamma_unit_names = []
            obj.gamma_unit_columns = []
            obj.gamma_fixed_columns = np.empty(0, dtype=int)
        obj.use_eps = obj.eps_var is not None
        obj.inv_eps = 1.0 / obj.eps_var if obj.use_eps else 0.0
        obj._user_cache = None
        obj.m0 = np.zeros(obj.d)
        obj.Lambda0 = obj.niw_lambda0_scale * np.eye(obj.d)
        obj._finalize()
        return obj
