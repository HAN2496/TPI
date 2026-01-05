import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .base import NeuralModel, RegressionModel, build_mlp, feature_map_torch, feature_map_np, feature_map_dim, _to_numpy

class OnlineMLP(NeuralModel):
    def __init__(self, input_dim, hidden_dims, dropout_rates, act_name, use_batchnorm=False, reduce='mean'):
        super().__init__(reduce=reduce)

        act = self._get_activation(act_name)
        self.net = build_mlp(input_dim, hidden_dims, act, dropout_rates, use_batchnorm)

    def step_rewards(self, x, detach=False):
        b, t, d = x.shape # (Batch size, Seq len, Dim)
        r = self.net(x.reshape(-1, d)).squeeze(-1).reshape(b, t)  # (B,T)
        if detach:
            r = r.detach().cpu().numpy()
        return r.reshape(b, t)

    def forward(self, x):
        r = self.step_rewards(x)  # (B,T)
        if self.reduce == 'sum':
            return r.sum(dim=1)
        if self.reduce == 'mean':
            return r.mean(dim=1)
        raise ValueError(f"Unknown reduce: {self.reduce}")

class OnlineLSTM(NeuralModel):
    def __init__(self, input_dim,
                 lstm_hidden_dim, lstm_layers, lstm_dropout,
                 hidden_dims, dropout_rates, act_name, use_batchnorm=False, reduce='mean'):
        super().__init__(reduce=reduce)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
            batch_first=True
        )

        act = self._get_activation(act_name)
        self.head = build_mlp(lstm_hidden_dim, hidden_dims, act, dropout_rates, use_batchnorm)

    def step_rewards(self, x, detach=False):
        h, _ = self.lstm(x) # (Batch size, Seq len, Dim)
        b, t, hdim = h.shape
        r = self.head(h.reshape(-1, hdim)).squeeze(-1).reshape(b, t)  # (B,T)
        if detach:
            r = r.detach().cpu().numpy()
        return r

    def forward(self, x):
        r = self.step_rewards(x)
        if self.reduce == 'sum':
            return r.sum(dim=1)
        if self.reduce == 'mean':
            return r.mean(dim=1)
        raise ValueError(f"Unknown reduce: {self.reduce}")


class OnlineAttention(NeuralModel):
    def __init__(self, input_dim, attn_dim, attn_heads, attn_dropout,
                 ffn_dim, ffn_dropout, hidden_dims, dropout_rates, act_name,
                 use_batchnorm=False, reduce='mean', use_ffn=True):
        super().__init__(reduce=reduce)

        if attn_dim % attn_heads != 0:
            raise ValueError(f"attn_dim({attn_dim}) must be divisible by attn_heads({attn_heads})")

        self.use_ffn = use_ffn

        self.in_proj = nn.Linear(input_dim, attn_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=attn_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        self.attn_ln = nn.LayerNorm(attn_dim)
        self.attn_drop = nn.Dropout(attn_dropout)

        if use_ffn:
            self.ffn = nn.Sequential(
                nn.Linear(attn_dim, ffn_dim),
                self._get_activation(act_name),
                nn.Dropout(ffn_dropout),
                nn.Linear(ffn_dim, attn_dim),
                nn.Dropout(ffn_dropout),
            )
            self.ffn_ln = nn.LayerNorm(attn_dim)

        act = self._get_activation(act_name)
        self.head = build_mlp(attn_dim, hidden_dims, act, dropout_rates, use_batchnorm)

    def _causal_attn_mask(self, T, device):
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def step_rewards(self, x, detach=False):
        b, t, _ = x.shape
        h = self.in_proj(x)

        attn_mask = self._causal_attn_mask(t, h.device)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)

        h = self.attn_ln(h + self.attn_drop(attn_out))

        if self.use_ffn:
            h2 = self.ffn(h)
            h = self.ffn_ln(h + h2)

        r = self.head(h.reshape(b * t, -1)).squeeze(-1).reshape(b, t)

        if detach:
            return r.detach().cpu().numpy()
        return r

    def forward(self, x):
        r = self.step_rewards(x)
        if self.reduce == 'sum':
            return r.sum(dim=1)
        if self.reduce == 'mean':
            return r.mean(dim=1)
        raise ValueError(f"Unknown reduce: {self.reduce}")


class OnlineNNRegression(NeuralModel):
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
        phi = feature_map_torch(x, self.form, self.w3, self.w4)
        return self.linear(phi).squeeze(-1)

    def forward(self, x):
        r = self.step_rewards(x)
        if self.reduce == "sum":
            return r.sum(dim=1)
        if self.reduce == "mean":
            return r.mean(dim=1)
        raise ValueError(f"Unknown reduce: {self.reduce}")


class OnlineRegression(RegressionModel):
    def __init__(self, basis, w3=None, w4=None, reduce="mean",
                 max_iter=100, C=1.0, solver="lbfgs", random_state=None, clip=20.0):
        super().__init__()

        self.basis = basis

        self.w3 = w3
        self.w4 = w4
        self.reduce = reduce
        self.clip = clip

        self.ops = {
            "abs": lambda x: np.abs(x),
            "quad": lambda x: x ** 2,
            "cube": lambda x: x ** 3,
            "shifted_exp": lambda x, w3, w4: np.exp(np.clip(w3 * np.maximum(np.abs(x) - w4, 0.0), None, self.clip))
        }

        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            max_iter=max_iter,
            C=C,
            solver=solver,
            random_state=random_state
        )

    def step_features(self, X):
        X_np = _to_numpy(X)
        features = []

        for dim, (feat_name, ops_list) in enumerate(self.basis.items()):
            dim_data = X_np[:, :, dim]

            for op_name in ops_list:
                if op_name == "shifted_exp":
                    w3_val = self.w3[feat_name]
                    w4_val = self.w4[feat_name]
                    val = self.ops[op_name](dim_data, w3_val, w4_val)
                else:
                    val = self.ops[op_name](dim_data)
                features.append(val)

        if not features:
            return np.zeros((X_np.shape[0], X_np.shape[1], 0))

        return np.stack(features, axis=-1)

    def transform_steps(self, X, fit=False):
        phi = self.step_features(X)
        b, t, f = phi.shape
        phi2 = phi.reshape(b * t, f)
        
        if fit:
            phi2s = self.scaler.fit_transform(phi2)
        else:
            phi2s = self.scaler.transform(phi2)
            
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
        phi_s = self.transform_steps(X, fit=False)
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

    def state_dict(self):
        return {
            "model": self.model,
            "scaler": self.scaler,
            "basis": self.basis,
            "w3": self.w3,
            "w4": self.w4,
            "reduce": self.reduce,
            "clip": self.clip,
            "best_threshold": self.best_threshold,
        }

    def load_state_dict(self, sd):
        self.model = sd["model"]
        self.scaler = sd["scaler"]
        self.basis = sd["basis"]
        self.w3 = sd.get("w3")
        self.w4 = sd.get("w4")
        self.reduce = sd["reduce"]
        self.clip = sd["clip"]
        self.best_threshold = sd["best_threshold"]
        return self