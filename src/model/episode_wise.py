import numpy as np
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .base import NeuralModel, RegressionModel, build_mlp, _to_numpy

class OfflineLSTM(NeuralModel):
    def __init__(self, input_dim,
                 lstm_hidden_dim, lstm_layers, lstm_dropout, bidirectional,
                 hidden_dims, dropout_rates, act_name, use_batchnorm=False, reduce='mean'):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
            batch_first=True,
            bidirectional=bidirectional
        )

        act = self._get_activation(act_name)
        self.head = build_mlp(lstm_hidden_dim, hidden_dims, act, dropout_rates, use_batchnorm)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h = h[-1]
        out = self.head(h)
        return out.squeeze(-1)



class OfflineRegression(RegressionModel):
    def __init__(self, basis, C=1.0, solver="lbfgs", max_iter=100, random_state=None):
        super().__init__()

        self.basis = basis

        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            max_iter=max_iter,
            C=C,
            solver=solver,
            random_state=random_state
        )

        self.ops = {
            "mean": lambda x: x.mean(axis=1),
            "std": lambda x: x.std(axis=1),
            "max": lambda x: x.max(axis=1),
            "min": lambda x: x.min(axis=1),
            "abs_mean": lambda x: np.abs(x).mean(axis=1),
            "sqrt_mean": lambda x: (x ** 2).mean(axis=1),
            "rmse": lambda x: np.sqrt((x ** 2).mean(axis=1)),
        }

    def extract_features(self, X):
        X_np = _to_numpy(X)
        features = []

        for dim, (feat_name, stats) in enumerate(self.basis.items()):
            dim_data = X_np[:, :, dim]

            for stat_name in stats:
                val = self.ops[stat_name](dim_data)
                features.append(val)

        return np.stack(features, axis=1)

    def fit(self, X, y):
        Z = self.extract_features(X)
        Z_scaled = self.scaler.fit_transform(Z)
        self.model.fit(Z_scaled, y)
        return self

    def decision_function(self, X):
        Z = self.extract_features(X)
        Z_scaled = self.scaler.transform(Z)
        return self.model.decision_function(Z_scaled)

    def predict_probability(self, X):
        Z = self.extract_features(X)
        Z_scaled = self.scaler.transform(Z)
        return self.model.predict_proba(Z_scaled)[:, 1]

    def predict_label(self, X, threshold=None):
        if threshold is None:
            threshold = self.best_threshold
        return (self.predict_probability(X) >= threshold).astype(int)

    def forward(self, X):
        return self.decision_function(X)

    def state_dict(self):
        return {
            "model": self.model,
            "scaler": self.scaler,
            "basis": self.basis,
            "best_threshold": self.best_threshold,
        }

    def load_state_dict(self, sd):
        self.model = sd["model"]
        self.scaler = sd["scaler"]
        self.basis = sd["basis"]
        self.best_threshold = sd["best_threshold"]
        return self