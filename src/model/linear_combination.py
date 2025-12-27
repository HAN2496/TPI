import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from .base import NeuralModel, RegressionModel, feature_map_torch, feature_map_np, feature_map_dim

class NNOnlineCombination(NeuralModel):
    def __init__(self, input_dim, form="quad_exp", reduce="mean"):
        super().__init__(reduce=reduce)
        self.input_dim = input_dim
        self.form = form

        if form == "quad_exp":
            self.w3 = nn.Parameter(torch.tensor(1.0))
            self.w4 = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_parameter("w3", None)
            self.register_parameter("w4", None)

        feat_dim = feature_map_dim(input_dim, form)
        self.linear = nn.Linear(feat_dim, 1, bias=True)

    def step_rewards(self, x):
        phi = feature_map_torch(x, self.form, self.w3, self.w4)  # (B,T,F)
        return self.linear(phi).squeeze(-1)                      # (B,T)

    def forward(self, x):
        r = self.step_rewards(x)
        if self.reduce == "sum":
            return r.sum(dim=1)
        if self.reduce == "mean":
            return r.mean(dim=1)
        raise ValueError(f"Unknown reduce: {self.reduce}")

def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

class OnlineCombination(RegressionModel):
    def __init__(self, input_dim, form, w3=None, w4=None, reduce="mean",
                 max_iter=100, C=1.0, solver="lbfgs", random_state=None, clip=20.0):
        super().__init__()
        self.input_dim = input_dim
        self.form = form
        self.w3 = w3
        self.w4 = w4
        self.reduce = reduce
        self.clip = clip

        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            max_iter=max_iter,
            C=C,
            solver=solver,
            random_state=random_state
        )

    def step_features(self, X):
        return feature_map_np(X, self.form, self.w3, self.w4, clip=self.clip)

    def transform_steps(self, X, fit=False):
        phi = self.step_features(X)
        b, t, f = phi.shape
        phi2 = phi.reshape(b * t, f)
        phi2s = self.scaler.fit_transform(phi2) if fit else self.scaler.transform(phi2)
        return phi2s.reshape(b, t, f)

    def episode_features(self, phi_scaled):
        if self.reduce == "sum":
            return phi_scaled.sum(axis=1)
        if self.reduce == "mean":
            return phi_scaled.mean(axis=1)
        raise ValueError(f"Unknown reduce: {self.reduce}")

    def fit(self, X, y):
        phi_s = self.transform_steps(X, fit=True)
        Z = self.episode_features(phi_s)
        self.model.fit(Z, y)
        return self

    def decision_function(self, X):
        phi_s = self.transform_steps(X, fit=False)
        Z = self.episode_features(phi_s)
        return self.model.decision_function(Z)

    def predict_probability(self, X):
        X_np = _to_numpy(X)
        phi_s = self.transform_steps(X_np, fit=False)
        Z = self.episode_features(phi_s)
        return self.model.predict_proba(Z)[:, 1]

    def predict_label(self, X, threshold=None):
        if threshold is None:
            threshold = self.best_threshold
        return (self.predict_probability(X) >= threshold).astype(int)

    def step_contributions(self, X):
        phi_s = self.transform_steps(X, fit=False)
        w = self.model.coef_.reshape(-1)
        return (phi_s * w[None, None, :]).sum(axis=-1)

    def step_rewards(self, X, include_intercept=True, detach=False):
        X_np = _to_numpy(X)
        contrib = self.step_contributions(X_np)
        if not include_intercept:
            return contrib

        b = float(self.model.intercept_[0])
        T = contrib.shape[1]

        if self.reduce == "sum":
            return contrib + (b / T)
        if self.reduce == "mean":
            return contrib + b
        raise ValueError(f"Unknown reduce: {self.reduce}")

    def forward(self, X):
        X_np = _to_numpy(X)
        r = self.step_rewards(X_np, include_intercept=True)
        return r.sum(axis=1) if self.reduce == "sum" else r.mean(axis=1)

    def raw_reward_params(self):
        beta = self.model.coef_.reshape(-1)
        mu = self.scaler.mean_
        sigma = self.scaler.scale_
        w_raw = beta / sigma
        b_raw = float(self.model.intercept_[0]) - float((w_raw * mu).sum())
        return w_raw, b_raw

    def state_dict(self):
        return {
            "model": self.model,
            "scaler": self.scaler,
            "input_dim": self.input_dim,
            "form": self.form,
            "w3": self.w3,
            "w4": self.w4,
            "reduce": self.reduce,
            "clip": self.clip,
            "best_threshold": self.best_threshold,
        }

    def load_state_dict(self, sd):
        self.model = sd["model"]
        self.scaler = sd["scaler"]
        self.input_dim = sd["input_dim"]
        self.form = sd["form"]
        self.w3 = sd["w3"]
        self.w4 = sd["w4"]
        self.reduce = sd["reduce"]
        self.clip = sd["clip"]
        self.best_threshold = sd["best_threshold"]
        return self