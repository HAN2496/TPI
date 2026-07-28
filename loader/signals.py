import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import butter, filtfilt

from .channels import CHANNELS, GROUPS


def read_frame(path):
    if path.suffix == ".mat":
        m = sio.loadmat(path)
        cols = [k for k in m if not k.startswith("__")]
        return pd.DataFrame(np.column_stack([m[k].ravel() for k in cols]), columns=cols)
    return pd.read_csv(path)


class Signal:
    def __init__(self, name, time, values):
        self.name = name
        self.time = time
        self.values = values
        self.group, self.display, self.latex, self.unit = CHANNELS[name]

    @property
    def axis_label(self):
        body = f"${self.latex}$" if self.latex else self.display
        return body + (f" [{self.unit}]" if self.unit else "")

    def window(self, t0, t1):
        m = (self.time >= t0) & (self.time < t1)
        return Signal(self.name, self.time[m], self.values[m])

    def __len__(self):
        return len(self.values)

    def __repr__(self):
        return f"Signal({self.name}={self.display}, n={len(self)})"


class Signals:
    def __init__(self, df, names=None):
        self.df = df
        self.names = names if names is not None else [c for c in df.columns if c != "Time"]
        self.time = df["Time"].to_numpy(np.float32)

    def group(self, name):
        return Signals(self.df, [c for c in GROUPS[name] if c in self.df.columns])

    @property
    def displays(self):
        return [CHANNELS[n][1] for n in self.names]

    def array(self, names=None):
        return self.df[names if names is not None else self.names].to_numpy(np.float32)

    def window(self, t0, t1):
        seg = self.df[(self.df["Time"] >= t0) & (self.df["Time"] < t1)].reset_index(drop=True)
        return Signals(seg, self.names)

    def smoothed(self, cutoff=12.0, order=2, cols=None):
        cols = cols if cols is not None else self.names
        dt = np.median(np.diff(self.time))
        b, a = butter(order, cutoff / (0.5 / dt), btype="low")
        df = self.df.copy()
        for c in cols:
            df[c] = filtfilt(b, a, df[c].to_numpy(np.float64))
        return Signals(df, self.names)

    def matrix(self, cols, time_range, downsample=1, pad=True, fill=0.0):
        df = self.df.iloc[::downsample] if downsample > 1 else self.df
        t = df["Time"].to_numpy(np.float32)
        m = (t >= time_range[0]) & (t < time_range[1])
        v = np.nan_to_num(df[cols].to_numpy(np.float32)[m], nan=fill)
        if not pad:
            return v
        dt = np.median(np.diff(t))
        expected = int(round((time_range[1] - time_range[0]) / dt))
        if len(v) < expected:
            v = np.vstack([v, np.full((expected - len(v), v.shape[1]), fill, np.float32)])
        return v[:expected]

    def __getitem__(self, name):
        return Signal(name, self.time, self.df[name].to_numpy(np.float32))

    def __getattr__(self, key):
        if key in GROUPS:
            return self.group(key)
        raise AttributeError(key)

    def __iter__(self):
        return (self[n] for n in self.names)

    def __len__(self):
        return len(self.names)

    def __repr__(self):
        return f"Signals({len(self.names)}: {self.names})" if len(self.names) <= 6 \
            else f"Signals({len(self.names)} signals)"
