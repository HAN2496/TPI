from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import numpy as np

class BaseModel(ABC):
    is_neural: bool
    is_online: bool
    def __init__(self):
        self.best_threshold = 0.5

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, x):
        raise NotImplementedError

    @abstractmethod
    def predict_label(self, x, threshold=None):
        raise NotImplementedError

class NeuralModel(BaseModel, nn.Module):
    is_neural = True
    def __init__(self, reduce='mean'):
        BaseModel.__init__(self)
        nn.Module.__init__(self)
        self.reduce = reduce

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError

    def _get_activation(self, activation):
        activations = {
            'relu': nn.ReLU(),
            'silu': nn.SiLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU()
        }
        return activations[activation]

    def predict_label(self, x, threshold=None):
        if threshold is None:
            threshold = self.best_threshold
        probs = self.predict_proba(x)
        return (probs >= threshold).long()

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

class RegressionModel(BaseModel):
    is_neural = False
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, x):
        raise NotImplementedError

    @abstractmethod
    def predict_label(self, x, threshold=None):
        raise NotImplementedError

    def eval(self):
        return self
    
    def load_state_dict(self, sd):
        return self

    def to(self, device):
        return self

def build_mlp(in_dim, hidden_dims, act, dropout_rates, use_batchnorm=False):
    layers = []
    for hidden_dim, dropout_rate in zip(hidden_dims, dropout_rates):
        layers.append(nn.Linear(in_dim, hidden_dim))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(act)
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, 1))
    return nn.Sequential(*layers)


def feature_map_dim(input_dim, form):
    if form == "quad_only":
        return input_dim
    if form in ("abs_quad", "quad_exp"):
        return input_dim * 2
    raise ValueError(f"Unknown form: {form}")

def feature_map_torch(x, form, w3=None, w4=None, clip=20.0):
    if form == "abs_quad":
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
    if form == "abs_quad":
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

def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    import torch
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)