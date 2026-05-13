import numpy as np

from ..bayes_additive.model import OnlineBayesianLogisticReward, sigmoid


class UserPosteriorMixtureBayesianLogisticReward(OnlineBayesianLogisticReward):
    """Test prior built as a mixture of per-train-user Laplace posteriors q_u=N(m_u, S_u).

    Each train user u contributes a Gaussian component centered at its MAP estimate m_u
    with diagonal Laplace covariance S_u. The test user prior is optionally augmented
    with a `population` component weighted by `global_component_weight`. Component
    covariances are inflated by `transfer_var_scale * diag(Sigma_0)` to reflect that
    test and train users are not identical.

    Because each mixture component has its own prior covariance, component weights are
    updated using a diagonal Laplace evidence approximation, not only the MAP objective.
    The correction term -0.5 log|Sigma_k| + 0.5 log|S_k| makes components with different
    uncertainty comparable.
    """

    def __init__(self, C=0.5, penalty="l1", prior_var_floor=0.05, user_var_scale=1.0,
                 update_temperature=1.0, map_max_iter=8, map_tol=1e-5,
                 map_use_full_cov=True, prototype_shrinkage=1.0,
                 component_temperature=1.0, global_component_weight=0.0,
                 transfer_var_scale=0.25, posterior_var_floor=1e-8,
                 component_var_floor=1e-4, random_state=42):
        super().__init__(
            C=C, penalty=penalty, prior_var_floor=prior_var_floor,
            user_var_scale=user_var_scale, update_temperature=update_temperature,
            map_max_iter=map_max_iter, map_tol=map_tol,
            map_use_full_cov=map_use_full_cov, random_state=random_state,
        )
        self.prototype_shrinkage = prototype_shrinkage
        self.component_temperature = component_temperature
        self.global_component_weight = global_component_weight
        self.transfer_var_scale = transfer_var_scale
        self.posterior_var_floor = posterior_var_floor
        self.component_var_floor = component_var_floor

    def fit_population(self, Phi_by_user, y_by_user, feature_names, user_names=None):
        stats = super().fit_population(Phi_by_user, y_by_user, feature_names, user_names=user_names)
        user_names = user_names or [f"user_{i}" for i in range(len(Phi_by_user))]
        base_transfer_diag = float(self.transfer_var_scale) * np.diag(self.prior_cov)
        rho = float(self.prototype_shrinkage)

        names, means, cov_diags, fit_scores = [], [], [], []
        user_post_names, user_post_means, user_post_cov_diags = [], [], []
        for name, Phi, y in zip(user_names, Phi_by_user, y_by_user):
            y = np.asarray(y, dtype=np.float64).ravel()
            if len(np.unique(y)) < 2:
                continue
            theta, cov_diag, obj = self._fit_map_given_prior(
                np.asarray(Phi, dtype=np.float64),
                y,
                self.prior_mean,
                self.prior_cov,
                self.prior_mean.copy(),
                prior_precision=self._prior_precision,
            )
            if rho != 1.0:
                theta = self.prior_mean + rho * (theta - self.prior_mean)
                cov_diag = (rho ** 2) * cov_diag
            cov_diag = np.maximum(cov_diag, self.posterior_var_floor)

            user_post_names.append(name)
            user_post_means.append(theta.copy())
            user_post_cov_diags.append(cov_diag.copy())

            transferred = np.maximum(cov_diag + base_transfer_diag, self.component_var_floor)
            names.append(name)
            means.append(theta)
            cov_diags.append(transferred)
            fit_scores.append(float(obj))

        if self.global_component_weight > 0.0:
            names.insert(0, "population")
            means.insert(0, self.prior_mean.copy())
            cov_diags.insert(0, np.maximum(np.diag(self.prior_cov), self.component_var_floor))
            fit_scores.insert(0, 0.0)

        if not means:
            names = ["population"]
            means = [self.prior_mean.copy()]
            cov_diags = [np.maximum(np.diag(self.prior_cov), self.component_var_floor)]
            fit_scores = [0.0]

        self.prototype_names = list(names)
        self.component_prior_means = np.stack(means, axis=0).astype(np.float64)
        self.component_prior_cov_diags = np.stack(cov_diags, axis=0).astype(np.float64)
        self.prototype_fit_scores = np.asarray(fit_scores, dtype=np.float64)

        D = self.prior_mean.shape[0]
        self.user_posterior_names = list(user_post_names)
        self.user_posterior_means = (
            np.stack(user_post_means, axis=0).astype(np.float64)
            if user_post_means else np.zeros((0, D), dtype=np.float64)
        )
        self.user_posterior_cov_diags = (
            np.stack(user_post_cov_diags, axis=0).astype(np.float64)
            if user_post_cov_diags else np.zeros((0, D), dtype=np.float64)
        )

        n = len(self.prototype_names)
        if self.global_component_weight > 0.0 and self.prototype_names[0] == "population" and n > 1:
            rest = (1.0 - self.global_component_weight) / (n - 1)
            weights = np.asarray([self.global_component_weight] + [rest] * (n - 1), dtype=np.float64)
        else:
            weights = np.full(n, 1.0 / n, dtype=np.float64)
        weights = weights / weights.sum()
        self.component_log_prior = np.log(np.maximum(weights, 1e-12))

        assert self.component_prior_means.shape == self.component_prior_cov_diags.shape, (
            f"prior means {self.component_prior_means.shape} vs cov_diags {self.component_prior_cov_diags.shape}"
        )
        assert len(self.prototype_names) == self.component_prior_means.shape[0]
        if self.user_posterior_cov_diags.shape[0] >= 2:
            spread = float(self.user_posterior_cov_diags.std(axis=0).max())
            if spread <= 1e-12:
                print("[Warning] Train user posterior covariance diagonals appear nearly identical.")

        self.reset_user()
        return stats

    def reset_user(self):
        if not hasattr(self, "component_prior_means") or self.component_prior_means is None:
            return OnlineBayesianLogisticReward.reset_user(self)
        self.component_means = self.component_prior_means.copy()
        self.component_cov_diags = self.component_prior_cov_diags.copy()
        self.component_log_weights = self.component_log_prior.copy()
        assert np.allclose(self.component_cov_diags, self.component_prior_cov_diags)
        self._refresh_mixture_moments()
        return self

    def predict(self, Phi, mean=None):
        if mean is not None or not hasattr(self, "component_means"):
            return OnlineBayesianLogisticReward.predict(self, Phi, mean=mean)
        Phi = np.asarray(Phi, dtype=np.float64)
        probs = sigmoid(Phi @ self.component_means.T)
        weights = self.component_weights()
        return probs @ weights

    def fit_user_map(self, Phi, y, warm_start=True):
        Phi = np.asarray(Phi, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        if len(y) == 0:
            return self.reset_user()

        means, cov_diags, log_evidences = [], [], []
        for k, prior_mean in enumerate(self.component_prior_means):
            start = self.component_means[k] if warm_start else prior_mean
            prior_cov_diag = np.maximum(self.component_prior_cov_diags[k], self.component_var_floor)
            prior_cov = np.diag(prior_cov_diag)
            prior_precision = np.diag(1.0 / prior_cov_diag)

            theta, cov_diag, obj = self._fit_map_given_prior(
                Phi, y, prior_mean, prior_cov, start,
                prior_precision=prior_precision,
            )
            cov_diag = np.maximum(cov_diag, self.posterior_var_floor)

            log_prior_det = np.sum(np.log(np.maximum(prior_cov_diag, self.posterior_var_floor)))
            log_post_det = np.sum(np.log(np.maximum(cov_diag, self.posterior_var_floor)))
            log_evidence = -obj - 0.5 * log_prior_det + 0.5 * log_post_det

            means.append(theta)
            cov_diags.append(cov_diag)
            log_evidences.append(log_evidence)

        self.component_means = np.stack(means, axis=0)
        self.component_cov_diags = np.stack(cov_diags, axis=0)
        log_evidences = np.asarray(log_evidences, dtype=np.float64)
        assert np.all(np.isfinite(log_evidences)), "Non-finite log evidence in component weight update"

        temp = max(float(self.component_temperature), 1e-6)
        logw = self.component_log_prior + log_evidences / temp
        logw = logw - np.max(logw)
        self.component_log_weights = logw - np.log(np.sum(np.exp(logw)))

        assert np.all(np.isfinite(self.component_means))
        assert np.all(np.isfinite(self.component_cov_diags))
        assert np.all(np.isfinite(self.component_log_weights))
        assert abs(self.component_weights().sum() - 1.0) < 1e-6

        self._refresh_mixture_moments()
        return self

    def component_weights(self):
        if not hasattr(self, "component_log_weights"):
            return np.ones(1, dtype=np.float64)
        w = np.exp(self.component_log_weights - np.max(self.component_log_weights))
        return w / np.sum(w)

    def component_summary(self):
        if not hasattr(self, "prototype_names"):
            return []
        weights = self.component_weights()
        return [
            {
                "name": name,
                "weight": float(weights[i]),
                "fit_score": float(self.prototype_fit_scores[i]),
            }
            for i, name in enumerate(self.prototype_names)
        ]

    def _refresh_mixture_moments(self):
        weights = self.component_weights()
        self.mean = weights @ self.component_means
        within = weights @ self.component_cov_diags
        between = weights @ ((self.component_means - self.mean[None, :]) ** 2)
        self.cov = np.diag(np.maximum(within + between, 1e-8))

    def posterior_distance_matrix(self):
        """Symmetric KL distance between train-user diagonal Gaussian posteriors only.
        Population component is excluded since it is not a train-user posterior.
        """
        if self.user_posterior_means is None or self.user_posterior_means.shape[0] < 2:
            return None, []
        m = self.user_posterior_means
        s = np.maximum(self.user_posterior_cov_diags, self.posterior_var_floor)
        n = m.shape[0]
        dist = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                kl_ij = 0.5 * np.sum(s[i] / s[j] + (m[j] - m[i]) ** 2 / s[j] - 1.0 + np.log(s[j] / s[i]))
                kl_ji = 0.5 * np.sum(s[j] / s[i] + (m[i] - m[j]) ** 2 / s[i] - 1.0 + np.log(s[i] / s[j]))
                dist[i, j] = 0.5 * (kl_ij + kl_ji)
        assert np.all(np.isfinite(dist))
        return dist, list(self.user_posterior_names)

    def state_dict(self):
        state = OnlineBayesianLogisticReward.state_dict(self)
        state.update({
            "model_class": "UserPosteriorMixtureBayesianLogisticReward",
            "prototype_shrinkage": self.prototype_shrinkage,
            "component_temperature": self.component_temperature,
            "global_component_weight": self.global_component_weight,
            "transfer_var_scale": self.transfer_var_scale,
            "posterior_var_floor": self.posterior_var_floor,
            "component_var_floor": self.component_var_floor,
            "prototype_names": list(getattr(self, "prototype_names", [])),
            "component_prior_means": getattr(self, "component_prior_means", None),
            "component_prior_cov_diags": getattr(self, "component_prior_cov_diags", None),
            "component_log_prior": getattr(self, "component_log_prior", None),
            "prototype_fit_scores": getattr(self, "prototype_fit_scores", None),
            "user_posterior_names": list(getattr(self, "user_posterior_names", [])),
            "user_posterior_means": getattr(self, "user_posterior_means", None),
            "user_posterior_cov_diags": getattr(self, "user_posterior_cov_diags", None),
        })
        return state

    @classmethod
    def from_state_dict(cls, state):
        obj = cls(
            C=state["C"], penalty=state["penalty"],
            prior_var_floor=state["prior_var_floor"],
            user_var_scale=state["user_var_scale"],
            update_temperature=state["update_temperature"],
            map_max_iter=state.get("map_max_iter", 8),
            map_tol=state.get("map_tol", 1e-5),
            map_use_full_cov=state.get("map_use_full_cov", True),
            prototype_shrinkage=state.get("prototype_shrinkage", 1.0),
            component_temperature=state.get("component_temperature", 1.0),
            global_component_weight=state.get("global_component_weight", 0.0),
            transfer_var_scale=state.get("transfer_var_scale", 0.25),
            posterior_var_floor=state.get("posterior_var_floor", 1e-8),
            component_var_floor=state.get("component_var_floor", 1e-4),
            random_state=state["random_state"],
        )
        obj.feature_names = state["feature_names"]
        obj.prior_mean = state["prior_mean"]
        obj.prior_cov = state["prior_cov"]
        obj._prepare_prior_precision()
        D = obj.prior_mean.shape[0]

        obj.user_posterior_names = list(state.get("user_posterior_names", []))
        obj.user_posterior_means = state.get("user_posterior_means")
        if obj.user_posterior_means is None:
            obj.user_posterior_means = np.zeros((0, D), dtype=np.float64)
        obj.user_posterior_cov_diags = state.get("user_posterior_cov_diags")
        if obj.user_posterior_cov_diags is None:
            obj.user_posterior_cov_diags = np.zeros((0, D), dtype=np.float64)

        obj.prototype_names = list(state.get("prototype_names", []))
        obj.component_prior_means = state.get("component_prior_means")
        obj.component_prior_cov_diags = state.get("component_prior_cov_diags")
        obj.component_log_prior = state.get("component_log_prior")
        obj.prototype_fit_scores = state.get("prototype_fit_scores")

        if (
            obj.component_prior_means is None
            or obj.component_prior_cov_diags is None
            or obj.component_log_prior is None
        ):
            obj.prototype_names = ["population"]
            obj.component_prior_means = obj.prior_mean[None, :]
            obj.component_prior_cov_diags = np.maximum(
                np.diag(obj.prior_cov), obj.component_var_floor
            )[None, :]
            obj.component_log_prior = np.log(np.ones(1, dtype=np.float64))
            obj.prototype_fit_scores = np.zeros(1, dtype=np.float64)

        obj.reset_user()
        return obj
