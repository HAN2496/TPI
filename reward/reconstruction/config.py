"""Configuration shared by reconstruction entry points."""

from dataclasses import dataclass


@dataclass
class LSTMConfig:
    """Architecture and training settings for the reconstruction LSTM."""

    epochs: int = 45
    hidden_dim: int = 96
    layers: int = 1
    dropout: float = 0.0
    batch_size: int = 32
    lr: float = 5e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    def model_kwargs(self):
        return {
            "hidden": self.hidden_dim,
            "layers": self.layers,
            "dropout": self.dropout,
        }

    def fit_kwargs(self):
        return {
            "bs": self.batch_size,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "grad_clip": self.grad_clip,
        }


@dataclass
class KalmanConfig:
    """Identification and inference settings for the reconstruction Kalman filter."""

    warmup: int = 0
    eta_t: bool = False
