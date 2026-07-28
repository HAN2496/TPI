import numpy as np
from dataclasses import dataclass, asdict

from .channels import resolve_features
from .dataset import Episode


@dataclass(frozen=True)
class View:
    features: tuple = None       # None=all | group name | tuple of names/groups
    around: tuple = (-1.0, 2.0)  # window relative to event_time [s]
    downsample: int = 1
    smooth: tuple = None         # (cutoff_hz, order) | None
    pad: bool = True

    def __post_init__(self):
        for k in ("features", "around", "smooth"):
            v = getattr(self, k)
            if isinstance(v, list):
                object.__setattr__(self, k, tuple(v))

    @property
    def cols(self):
        return resolve_features(self.features)

    @property
    def fs(self):
        return 100.0 / self.downsample   # raw 100Hz logging

    @property
    def duration(self):
        return float(self.around[1] - self.around[0])

    def matrix(self, ep):
        sig = ep.signals.smoothed(*self.smooth, self.cols) if self.smooth else ep.signals
        t0, t1 = ep.event_time + self.around[0], ep.event_time + self.around[1]
        return sig.matrix(self.cols, (t0, t1), self.downsample, self.pad)

    def tensor(self, eps):
        eps = [ep for ep in eps if ep.label is not None]
        X = [self.matrix(ep) for ep in eps]
        y = np.array([int(ep.label) for ep in eps], dtype=np.int64)
        return (np.stack(X) if self.pad else X), y

    def to_json(self):
        return asdict(self)

    def __call__(self, x):
        return self.matrix(x) if isinstance(x, Episode) else self.tensor(x)
