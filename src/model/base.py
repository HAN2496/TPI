from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import numpy as np

class BaseModel(nn.Module, ABC):
    def __init__(self, reduce='mean'):
        super().__init__()
        self.reduce = reduce
        self.best_threshold = 0.5

    @abstractmethod
    def forward(self, x): # R(tau)
        raise NotImplementedError

    def _get_activation(self, activation):
        activations = {
            'relu': nn.ReLU(),
            'silu': nn.SiLU(), 
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU()
        }
        return activations.get(activation, nn.SiLU())

    def predict_label(self, x, threshold=None):
        if threshold is None:
            threshold = self.best_threshold
        probs = self.predict_probability(x)
        return (probs >= threshold).long()

    def predict_probability(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

def build_mlp(in_dim, hidden_dims, act, dropout_rate=0.0, use_batchnorm=False):
    layers = []
    for h in hidden_dims:
        layers.append(nn.Linear(in_dim, h))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(h))
        layers.append(act)
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        in_dim = h
    layers.append(nn.Linear(in_dim, 1))
    return nn.Sequential(*layers)


def feature_map_dim(input_dim, form):
    if form == "quad_only":
        return input_dim
    if form in ("linear_quad", "quad_exp"):
        return input_dim * 2
    raise ValueError(f"Unknown form: {form}")

def feature_map_torch(x, form, w3=None, w4=None, clip=20.0):
    if form == "linear_quad":
        return torch.cat([torch.abs(x), x * x], dim=-1)

    if form == "quad_only":
        return x * x

    if form == "quad_exp":
        if w3 is None or w4 is None:
            raise ValueError("quad_exp requires w3, w4")
        quad = x * x

        w3_t = torch.as_tensor(w3, dtype=x.dtype, device=x.device)
        w4_t = torch.as_tensor(w4, dtype=x.dtype, device=x.device)

        shifted = torch.relu(torch.abs(x) - w4_t)
        expv = torch.exp(torch.clamp(w3_t * shifted, max=clip))
        return torch.cat([quad, expv], dim=-1)

    raise ValueError(f"Unknown form: {form}")

def feature_map_np(X, form, w3=None, w4=None, clip=20.0):
    if form == "linear_quad":
        return np.concatenate([np.abs(X), X * X], axis=-1)

    if form == "quad_only":
        return X * X

    if form == "quad_exp":
        if w3 is None or w4 is None:
            raise ValueError("quad_exp requires w3, w4")
        quad = X * X

        w3_np = np.asarray(w3, dtype=X.dtype)
        w4_np = np.asarray(w4, dtype=X.dtype)

        shifted = np.maximum(np.abs(X) - w4_np, 0.0)
        expv = np.exp(np.clip(w3_np * shifted, None, clip))
        return np.concatenate([quad, expv], axis=-1)

    raise ValueError(f"Unknown form: {form}")
