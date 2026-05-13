import numpy as np
import inspect
from sklearn.linear_model import LogisticRegression


def sigmoid(x):
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


class OnlineBayesianLogisticReward:
    """Hierarchical prior + online MAP posterior updates."""

    def __init__(self, C=0.5, penalty="l1", prior_var_floor=0.05, user_var_scale=1.0,
                 update_temperature=1.0, map_max_iter=8, map_tol=1e-5,
                 map_use_full_cov=True, random_state=42):
        self.C = C
        self.penalty = penalty
        self.prior_var_floor = prior_var_floor
        self.user_var_scale = user_var_scale
        self.update_temperature = update_temperature
        self.map_max_iter = map_max_iter
        self.map_tol = map_tol
        self.map_use_full_cov = map_use_full_cov
        self.random_state = random_state

    def fit_population(self, Phi_by_user, y_by_user, feature_names, user_names=None):
        self.feature_names = list(feature_names)
        Phi_all = np.concatenate(Phi_by_user, axis=0)
        y_all = np.concatenate(y_by_user, axis=0).astype(int)

        pooled = self._fit_logistic(Phi_all, y_all)
        theta = self._theta_from_clf(pooled, Phi_all.shape[1])
        pooled_cov = self._laplace_cov(Phi_all, theta)

        user_thetas = []
        for Phi, y in zip(Phi_by_user, y_by_user):
            y = y.astype(int)
            if len(np.unique(y)) < 2:
                continue
            clf = self._fit_logistic(Phi, y)
            user_thetas.append(self._theta_from_clf(clf, Phi.shape[1]))

        if len(user_thetas) >= 2:
            user_thetas = np.stack(user_thetas, axis=0)
            user_var = np.var(user_thetas, axis=0, ddof=1)
        else:
            user_var = np.zeros_like(theta)

        cov = pooled_cov + np.diag(self.user_var_scale * user_var)
        diag = np.maximum(np.diag(cov), self.prior_var_floor)
        cov[np.diag_indices_from(cov)] = diag

        self.prior_mean = theta.astype(np.float64)
        self.prior_cov = cov.astype(np.float64)
        self._prepare_prior_precision()
        self.reset_user()

        probs = self.predict(Phi_all, mean=self.prior_mean)
        return {"population_brier": float(np.mean((probs - y_all) ** 2))}

    def reset_user(self):
        self.mean = self.prior_mean.copy()
        self.cov = self.prior_cov.copy()
        return self

    def predict(self, Phi, mean=None):
        mean = self.mean if mean is None else mean
        return sigmoid(np.asarray(Phi) @ mean)

    def update_one(self, phi, y):
        phi = np.asarray(phi, dtype=np.float64).ravel()
        y = float(y)
        p = float(sigmoid(phi @ self.mean))
        r = max(p * (1.0 - p), 1e-4) / max(self.update_temperature, 1e-6)

        s_phi = self.cov @ phi
        denom = 1.0 + r * float(phi @ s_phi)
        self.cov = self.cov - (r / denom) * np.outer(s_phi, s_phi)
        self.cov = 0.5 * (self.cov + self.cov.T)
        self.mean = self.mean + self.cov @ phi * (y - p)
        return p

    def update_many(self, Phi, y):
        preds = []
        for phi, yi in zip(Phi, y):
            preds.append(self.update_one(phi, yi))
        return np.asarray(preds, dtype=np.float64)

    def fit_user_map(self, Phi, y, warm_start=True):
        Phi = np.asarray(Phi, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        if len(y) == 0:
            return self.reset_user()

        theta = self.mean.copy() if warm_start else self.prior_mean.copy()
        theta, cov_diag, _ = self._fit_map_given_prior(
            Phi, y, self.prior_mean, self.prior_cov, theta,
            prior_precision=self._prior_precision)

        self.mean = theta
        self.cov = np.diag(cov_diag)
        return self

    def _fit_map_given_prior(self, Phi, y, prior_mean, prior_cov, theta,
                             prior_precision=None):
        prior_mean = np.asarray(prior_mean, dtype=np.float64)
        prior_cov = np.asarray(prior_cov, dtype=np.float64)
        theta = np.asarray(theta, dtype=np.float64).copy()
        likelihood_scale = 1.0 / max(self.update_temperature, 1e-6)
        if self.map_use_full_cov:
            precision = prior_precision
            if precision is None:
                precision = self._precision_from_cov(prior_cov)
            prior_grad = lambda v: precision @ (v - prior_mean)
            solve_step = lambda w_, grad_: self._solve_cov_plus_lowrank(
                prior_cov, Phi, w_, grad_)
            cov_diag_fn = lambda w_: self._cov_diag_cov_plus_lowrank(
                prior_cov, Phi, w_)
        else:
            precision_diag = 1.0 / np.maximum(np.diag(prior_cov), self.prior_var_floor)
            precision = precision_diag
            prior_grad = lambda v: precision_diag * (v - prior_mean)
            solve_step = lambda w_, grad_: self._solve_diag_plus_lowrank(
                Phi, w_, grad_, precision_diag)
            cov_diag_fn = lambda w_: self._cov_diag_diag_plus_lowrank(
                Phi, w_, precision_diag)

        for _ in range(self.map_max_iter):
            eta = Phi @ theta
            p = sigmoid(eta)
            w = likelihood_scale * np.maximum(p * (1.0 - p), 1e-5)
            grad = likelihood_scale * (Phi.T @ (p - y)) + prior_grad(theta)
            step = solve_step(w, grad)
            current_obj = self._map_objective(
                Phi, y, theta, precision, likelihood_scale, prior_mean)
            accepted = False
            for scale in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125,
                          0.015625, 0.0078125):
                candidate = theta - scale * step
                if self._map_objective(
                        Phi, y, candidate, precision, likelihood_scale, prior_mean) <= current_obj:
                    theta_next = candidate
                    accepted_step = scale * step
                    accepted = True
                    break
            if not accepted:
                break
            if np.linalg.norm(accepted_step) <= self.map_tol * (1.0 + np.linalg.norm(theta)):
                theta = theta_next
                break
            theta = theta_next

        p = sigmoid(Phi @ theta)
        w = likelihood_scale * np.maximum(p * (1.0 - p), 1e-5)
        cov_diag = cov_diag_fn(w)
        obj = self._map_objective(Phi, y, theta, precision, likelihood_scale, prior_mean)

        return theta, cov_diag, obj

    def state_dict(self):
        return {
            "C": self.C,
            "penalty": self.penalty,
            "prior_var_floor": self.prior_var_floor,
            "user_var_scale": self.user_var_scale,
            "update_temperature": self.update_temperature,
            "map_max_iter": self.map_max_iter,
            "map_tol": self.map_tol,
            "map_use_full_cov": self.map_use_full_cov,
            "random_state": self.random_state,
            "feature_names": self.feature_names,
            "prior_mean": self.prior_mean,
            "prior_cov": self.prior_cov,
        }

    @classmethod
    def from_state_dict(cls, state):
        obj = cls(C=state["C"], penalty=state["penalty"],
                  prior_var_floor=state["prior_var_floor"],
                  user_var_scale=state["user_var_scale"],
                  update_temperature=state["update_temperature"],
                  map_max_iter=state.get("map_max_iter", 8),
                  map_tol=state.get("map_tol", 1e-5),
                  map_use_full_cov=state.get("map_use_full_cov", True),
                  random_state=state["random_state"])
        obj.feature_names = state["feature_names"]
        obj.prior_mean = state["prior_mean"]
        obj.prior_cov = state["prior_cov"]
        obj._prepare_prior_precision()
        obj.reset_user()
        return obj

    def _prepare_prior_precision(self):
        self._prior_precision = self._precision_from_cov(self.prior_cov)

    def _precision_from_cov(self, cov):
        jitter = 1e-6 * np.eye(cov.shape[0])
        try:
            return np.linalg.inv(cov + jitter)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(cov + jitter)

    def _fit_logistic(self, Phi, y):
        return LogisticRegression(**self._logistic_kwargs()).fit(Phi[:, 1:], y)

    def _logistic_kwargs(self):
        common = {
            "C": self.C,
            "class_weight": "balanced",
            "max_iter": 2000,
            "random_state": self.random_state,
        }
        penalty_default = inspect.signature(LogisticRegression).parameters["penalty"].default

        if penalty_default == "deprecated":
            common["l1_ratio"] = 1.0 if self.penalty == "l1" else 0.0
            common["solver"] = "liblinear" if self.penalty == "l1" else "lbfgs"
            return common

        common["penalty"] = "l1" if self.penalty == "l1" else "l2"
        common["solver"] = "liblinear" if self.penalty == "l1" else "lbfgs"
        return common

    def _solve_diag_plus_lowrank(self, Phi, w, grad, precision_diag):
        d_inv = 1.0 / precision_diag
        sqrt_w = np.sqrt(w)
        A = Phi * sqrt_w[:, None]
        d_inv_grad = d_inv * grad
        rhs = A @ d_inv_grad
        AD = A * d_inv[None, :]
        middle = np.eye(A.shape[0]) + AD @ A.T
        try:
            z = np.linalg.solve(middle, rhs)
        except np.linalg.LinAlgError:
            z = np.linalg.pinv(middle) @ rhs
        return d_inv_grad - d_inv * (A.T @ z)

    def _map_objective(self, Phi, y, theta, precision, likelihood_scale, prior_mean):
        eta = Phi @ theta
        nll = likelihood_scale * np.sum(np.logaddexp(0.0, eta) - y * eta)
        diff = theta - prior_mean
        with np.errstate(over="ignore", invalid="ignore"):
            if np.ndim(precision) == 1:
                prior = 0.5 * np.sum(precision * diff * diff)
            else:
                prior = 0.5 * float(diff @ precision @ diff)
            obj = float(nll + prior)
        return obj if np.isfinite(obj) else float("inf")

    def _solve_cov_plus_lowrank(self, cov, Phi, w, grad):
        sqrt_w = np.sqrt(w)
        A = Phi * sqrt_w[:, None]
        cov_grad = cov @ grad
        B = A @ cov
        rhs = A @ cov_grad
        middle = np.eye(A.shape[0]) + B @ A.T
        try:
            z = np.linalg.solve(middle, rhs)
        except np.linalg.LinAlgError:
            z = np.linalg.pinv(middle) @ rhs
        return cov_grad - B.T @ z

    def _cov_diag_diag_plus_lowrank(self, Phi, w, precision_diag):
        d_inv = 1.0 / precision_diag
        sqrt_w = np.sqrt(w)
        A = Phi * sqrt_w[:, None]
        AD = A * d_inv[None, :]
        middle = np.eye(A.shape[0]) + AD @ A.T
        try:
            middle_inv_A = np.linalg.solve(middle, A)
        except np.linalg.LinAlgError:
            middle_inv_A = np.linalg.pinv(middle) @ A
        correction = (d_inv ** 2) * np.sum(A * middle_inv_A, axis=0)
        return np.maximum(d_inv - correction, 1e-8)

    def _cov_diag_cov_plus_lowrank(self, cov, Phi, w):
        sqrt_w = np.sqrt(w)
        A = Phi * sqrt_w[:, None]
        B = A @ cov
        middle = np.eye(A.shape[0]) + B @ A.T
        try:
            middle_inv_B = np.linalg.solve(middle, B)
        except np.linalg.LinAlgError:
            middle_inv_B = np.linalg.pinv(middle) @ B
        correction = np.sum(B * middle_inv_B, axis=0)
        return np.maximum(np.diag(cov) - correction, 1e-8)

    def _theta_from_clf(self, clf, n_features_with_bias):
        theta = np.zeros(n_features_with_bias, dtype=np.float64)
        theta[0] = float(clf.intercept_[0])
        theta[1:] = clf.coef_.reshape(-1)
        return theta

    def _laplace_cov(self, Phi, theta):
        p = sigmoid(Phi @ theta)
        w = p * (1.0 - p)
        precision = (Phi.T * w) @ Phi
        precision += (1.0 / max(self.C, 1e-6)) * np.eye(Phi.shape[1])
        precision[0, 0] *= 0.1
        precision += 1e-6 * np.eye(Phi.shape[1])
        try:
            return np.linalg.inv(precision)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(precision)


class PrototypeBayesianLogisticReward(OnlineBayesianLogisticReward):
    """Soft latent preference types built from historical user profiles."""

    def __init__(self, C=0.5, penalty="l1", prior_var_floor=0.05, user_var_scale=1.0,
                 update_temperature=1.0, map_max_iter=8, map_tol=1e-5,
                 map_use_full_cov=True, prototype_shrinkage=1.0,
                 component_var_scale=1.0, component_temperature=1.0,
                 global_component_weight=0.0, random_state=42):
        super().__init__(
            C=C,
            penalty=penalty,
            prior_var_floor=prior_var_floor,
            user_var_scale=user_var_scale,
            update_temperature=update_temperature,
            map_max_iter=map_max_iter,
            map_tol=map_tol,
            map_use_full_cov=map_use_full_cov,
            random_state=random_state,
        )
        self.prototype_shrinkage = prototype_shrinkage
        self.component_var_scale = component_var_scale
        self.component_temperature = component_temperature
        self.global_component_weight = global_component_weight

    def fit_population(self, Phi_by_user, y_by_user, feature_names, user_names=None):
        stats = super().fit_population(Phi_by_user, y_by_user, feature_names, user_names=user_names)
        user_names = user_names or [f"user_{i}" for i in range(len(Phi_by_user))]

        prototype_names = []
        prototype_means = []
        prototype_fit_scores = []
        for name, Phi, y in zip(user_names, Phi_by_user, y_by_user):
            y = np.asarray(y, dtype=np.float64).ravel()
            if len(np.unique(y)) < 2:
                continue
            theta, _, obj = self._fit_map_given_prior(
                np.asarray(Phi, dtype=np.float64),
                y,
                self.prior_mean,
                self.prior_cov,
                self.prior_mean.copy(),
                prior_precision=self._prior_precision,
            )
            theta = self.prior_mean + self.prototype_shrinkage * (theta - self.prior_mean)
            prototype_names.append(name)
            prototype_means.append(theta)
            prototype_fit_scores.append(float(obj))

        if self.global_component_weight > 0.0:
            prototype_names.insert(0, "population")
            prototype_means.insert(0, self.prior_mean.copy())
            prototype_fit_scores.insert(0, 0.0)

        if not prototype_means:
            prototype_names = ["population"]
            prototype_means = [self.prior_mean.copy()]
            prototype_fit_scores = [0.0]

        component_cov = self.prior_cov * float(self.component_var_scale)
        diag = np.maximum(np.diag(component_cov), self.prior_var_floor)
        component_cov[np.diag_indices_from(component_cov)] = diag

        self.prototype_names = list(prototype_names)
        self.component_prior_means = np.stack(prototype_means, axis=0).astype(np.float64)
        self.component_prior_cov = component_cov.astype(np.float64)
        self.component_prior_precision = self._precision_from_cov(self.component_prior_cov)
        self.prototype_fit_scores = np.asarray(prototype_fit_scores, dtype=np.float64)

        n = len(self.prototype_names)
        if self.global_component_weight > 0.0 and self.prototype_names[0] == "population" and n > 1:
            rest = (1.0 - self.global_component_weight) / (n - 1)
            weights = np.asarray([self.global_component_weight] + [rest] * (n - 1), dtype=np.float64)
        else:
            weights = np.full(n, 1.0 / n, dtype=np.float64)
        weights = weights / weights.sum()
        self.component_log_prior = np.log(np.maximum(weights, 1e-12))
        self.reset_user()
        return stats

    def reset_user(self):
        if not hasattr(self, "component_prior_means"):
            return super().reset_user()
        self.component_means = self.component_prior_means.copy()
        self.component_cov_diags = np.tile(
            np.diag(self.component_prior_cov)[None, :],
            (len(self.prototype_names), 1),
        )
        self.component_log_weights = self.component_log_prior.copy()
        self._refresh_mixture_moments()
        return self

    def predict(self, Phi, mean=None):
        if mean is not None or not hasattr(self, "component_means"):
            return super().predict(Phi, mean=mean)
        Phi = np.asarray(Phi, dtype=np.float64)
        probs = sigmoid(Phi @ self.component_means.T)
        weights = self.component_weights()
        return probs @ weights

    def fit_user_map(self, Phi, y, warm_start=True):
        Phi = np.asarray(Phi, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        if len(y) == 0:
            return self.reset_user()

        means = []
        cov_diags = []
        objectives = []
        for k, prior_mean in enumerate(self.component_prior_means):
            start = self.component_means[k] if warm_start else prior_mean
            theta, cov_diag, obj = self._fit_map_given_prior(
                Phi,
                y,
                prior_mean,
                self.component_prior_cov,
                start,
                prior_precision=self.component_prior_precision,
            )
            means.append(theta)
            cov_diags.append(cov_diag)
            objectives.append(obj)

        self.component_means = np.stack(means, axis=0)
        self.component_cov_diags = np.stack(cov_diags, axis=0)
        objectives = np.asarray(objectives, dtype=np.float64)
        temp = max(float(self.component_temperature), 1e-6)
        logw = self.component_log_prior - objectives / temp
        logw = logw - np.max(logw)
        self.component_log_weights = logw - np.log(np.sum(np.exp(logw)))
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

    def state_dict(self):
        state = super().state_dict()
        state.update({
            "model_class": "PrototypeBayesianLogisticReward",
            "prototype_shrinkage": self.prototype_shrinkage,
            "component_var_scale": self.component_var_scale,
            "component_temperature": self.component_temperature,
            "global_component_weight": self.global_component_weight,
            "prototype_names": getattr(self, "prototype_names", []),
            "component_prior_means": getattr(self, "component_prior_means", None),
            "component_prior_cov": getattr(self, "component_prior_cov", None),
            "component_log_prior": getattr(self, "component_log_prior", None),
            "prototype_fit_scores": getattr(self, "prototype_fit_scores", None),
        })
        return state

    @classmethod
    def from_state_dict(cls, state):
        obj = cls(
            C=state["C"],
            penalty=state["penalty"],
            prior_var_floor=state["prior_var_floor"],
            user_var_scale=state["user_var_scale"],
            update_temperature=state["update_temperature"],
            map_max_iter=state.get("map_max_iter", 8),
            map_tol=state.get("map_tol", 1e-5),
            map_use_full_cov=state.get("map_use_full_cov", True),
            prototype_shrinkage=state.get("prototype_shrinkage", 1.0),
            component_var_scale=state.get("component_var_scale", 1.0),
            component_temperature=state.get("component_temperature", 1.0),
            global_component_weight=state.get("global_component_weight", 0.0),
            random_state=state["random_state"],
        )
        obj.feature_names = state["feature_names"]
        obj.prior_mean = state["prior_mean"]
        obj.prior_cov = state["prior_cov"]
        obj._prepare_prior_precision()
        obj.prototype_names = state.get("prototype_names", [])
        obj.component_prior_means = state.get("component_prior_means")
        obj.component_prior_cov = state.get("component_prior_cov")
        obj.component_log_prior = state.get("component_log_prior")
        obj.prototype_fit_scores = state.get("prototype_fit_scores")
        if obj.component_prior_cov is not None:
            obj.component_prior_precision = obj._precision_from_cov(obj.component_prior_cov)
        obj.reset_user()
        return obj
