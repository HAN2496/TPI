"""Candidate feature bank and redundancy pruning (paper Section V-A).

The bank lists, per channel kind, the statistics that are physically defensible
for that kind (docs/fully_bayesian/features/main.tex).  The order inside each
list is the *priority order* used by pruning: when two statistics of the same
channel have |Pearson r| >= rho_max on the population episodes, the one that
appears later in the list is dropped.
"""
from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace

import numpy as np

from ..features import FNS, Features

# Channel kind by name.  Acceleration channels are used as-is, rate channels are
# additionally differentiated (rate -> acceleration) and get the W_e / W_k
# weightings on the derivative.
KIND = {
    "IMU_VerAccelVal": "accel_z",
    "IMU_LongAccelVal": "accel_xy",
    "IMU_LatAccelVal": "accel_xy",
    "Pitch_rate_6D": "rate_rot",
    "Roll_rate_6D": "rate_rot",
    "IMU_RollRtVal": "rate_rot",
    "IMU_YawRtVal": "rate_rot",
    "Bounce_rate_6D": "rate_z",
}

_COMMON = ["p2p", "abs_peak", "rms", "std", "p95_abs", "impulse_abs",
           "crest", "vdv", "mtvv", "sigma_sd", "band_low", "band_mid"]
_DERIV = ["p2p_deriv", "abs_peak_deriv", "rms_deriv", "vdv_deriv"]

FULL_BANK = {
    "accel_z":  _COMMON[:9] + ["wrms_z"] + _COMMON[9:],
    "accel_xy": _COMMON[:9] + ["wrms_xy"] + _COMMON[9:],
    "rate_rot": _DERIV + ["wrms_rot"] + _COMMON,
    "rate_z":   _DERIV + ["wrms_z_deriv"] + _COMMON,
}


def full_bank(channels):
    """{channel: [statistic, ...]} for the given channel names, in priority order."""
    bank = OrderedDict()
    for ch in channels:
        kind = KIND.get(ch)
        if kind is None:
            raise KeyError(f"no bank kind for channel {ch!r}; add it to loeo.bank.KIND")
        stats = FULL_BANK[kind]
        for s in stats:
            if s not in FNS:
                raise KeyError(f"unknown statistic {s!r} in bank for {ch}")
        bank[ch] = list(stats)
    return bank


def _raw_features(X, channels, fs, bank):
    idx = {ch: j for j, ch in enumerate(channels)}
    cols, names = [], []
    for ch, stats in bank.items():
        for s in stats:
            cols.append(FNS[s](X[:, :, idx[ch]], fs))
            names.append((ch, s))
    return np.stack(cols, axis=1), names


def prune_bank(X_list, channels, fs, bank, rho_max=0.95, min_std=1e-9):
    """Greedy within-channel pruning by absolute Pearson correlation.

    X_list: raw windows of the *population* evaluators only.
    Returns (pruned_bank, report) where report lists removed (channel, stat, kept_by, r).
    """
    X = np.concatenate(X_list, axis=0)
    F, names = _raw_features(X, channels, fs, bank)
    sd = F.std(0)
    kept = OrderedDict((ch, []) for ch in bank)
    report = []
    for ch in bank:
        js = [j for j, (c, _) in enumerate(names) if c == ch]
        chosen = []
        for j in js:                                   # priority order = list order
            if sd[j] < min_std:
                report.append({"channel": ch, "stat": names[j][1], "kept_by": None,
                               "r": None, "reason": "constant"})
                continue
            worst, worst_k = 0.0, None
            for k in chosen:
                r = abs(np.corrcoef(F[:, j], F[:, k])[0, 1])
                if r > worst:
                    worst, worst_k = r, k
            if worst_k is not None and worst >= rho_max:
                report.append({"channel": ch, "stat": names[j][1], "kept_by": names[worst_k][1],
                               "r": float(worst), "reason": "correlated"})
                continue
            chosen.append(j)
            kept[ch].append(names[j][1])
    return kept, report


def make_pipeline(manual_stats, channels, fs, standardize=True, include_bias=True):
    """Features pipeline object for a (pruned) bank without a full run Config."""
    cfg = SimpleNamespace(
        view=SimpleNamespace(cols=list(channels), fs=fs),
        standardize=standardize, include_bias=include_bias,
        manual_stats=OrderedDict(manual_stats),
    )
    return Features(cfg)


def restrict_bank(bank, channels_subset):
    return OrderedDict((ch, list(stats)) for ch, stats in bank.items() if ch in channels_subset)


class ColumnSubset:
    """A column subset of a fitted full pipeline, exposing the same feature_names / groups interface.

    Selecting columns after the full transform is exact because standardization is per column
    and the bias column is kept.  Used so that selection candidates do not re-extract features
    from raw windows.
    """

    def __init__(self, phi_full, feature_names):
        keep = set(feature_names)
        self.idx = [j for j, f in enumerate(phi_full.feature_names) if f == "bias" or f in keep]
        missing = keep - set(phi_full.feature_names)
        if missing:
            raise KeyError(f"features not in the full pipeline: {sorted(missing)}")
        self.feature_names = [phi_full.feature_names[j] for j in self.idx]
        self.groups = [phi_full.groups[j] for j in self.idx]
        self.pairs = [(g, f.split("__", 1)[1]) for f, g in zip(self.feature_names, self.groups) if f != "bias"]
        self._phi = phi_full

    def transform_Z(self, Z_full):
        return np.asarray(Z_full, np.float64)[:, self.idx]

    def transform(self, X):
        return self.transform_Z(self._phi.transform(X))
