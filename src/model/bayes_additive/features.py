import numpy as np


def _safe_stat(fn, x, default=0.0):
    try:
        v = fn(x)
        if np.isfinite(v):
            return float(v)
    except Exception:
        pass
    return float(default)


class EpisodeFeatureExtractor:
    """Streaming-friendly episode summaries from selected vehicle channels."""

    def __init__(self, signal_names, duration=2.0, n_subwindows=1):
        self.signal_names = list(signal_names)
        self.duration = float(duration)
        self.n_subwindows = int(n_subwindows)

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        dt = self.duration / max(X.shape[1] - 1, 1)
        rows = []
        for ep in X:
            vals, names = self._one(ep, dt)
            rows.append(vals)
        return np.asarray(rows, dtype=np.float32), names

    def _one(self, ep, dt):
        out = {}
        self._add_signal_stats(out, ep, dt, name_prefix="", compact=False)
        self._add_named_interactions(out)

        if self.n_subwindows > 1:
            indices = np.array_split(np.arange(ep.shape[0]), self.n_subwindows)
            for widx, idx in enumerate(indices):
                if len(idx) < 2:
                    continue
                self._add_signal_stats(out, ep[idx], dt, name_prefix=f"w{widx}_", compact=True)

        names = list(out.keys())
        return [out[n] for n in names], names

    def _add_signal_stats(self, out, ep, dt, name_prefix="", compact=False):
        for j, name in enumerate(self.signal_names):
            x = np.asarray(ep[:, j], dtype=np.float64)
            dx = np.diff(x) / max(dt, 1e-6)
            prefix = f"{name_prefix}{self._short_name(name)}"

            out[f"{prefix}_mean"] = _safe_stat(np.mean, x)
            out[f"{prefix}_std"] = _safe_stat(np.std, x)
            out[f"{prefix}_abs_peak"] = _safe_stat(lambda z: np.max(np.abs(z)), x)
            out[f"{prefix}_energy"] = _safe_stat(lambda z: np.mean(z * z), x)
            out[f"{prefix}_slope_rms"] = _safe_stat(lambda z: np.sqrt(np.mean(z * z)), dx)

            if compact:
                continue

            out[f"{prefix}_rms"] = _safe_stat(lambda z: np.sqrt(np.mean(z * z)), x)
            out[f"{prefix}_p95_abs"] = _safe_stat(lambda z: np.percentile(np.abs(z), 95), x)
            out[f"{prefix}_range"] = _safe_stat(lambda z: np.max(z) - np.min(z), x)
            out[f"{prefix}_impulse_abs"] = _safe_stat(lambda z: np.sum(np.abs(z)) * dt, x)
            out[f"{prefix}_slope_abs_peak"] = _safe_stat(lambda z: np.max(np.abs(z)), dx)

    def _short_name(self, name):
        mapping = {
            "IMU_LongAccelVal": "long_acc",
            "IMU_LatAccelVal": "lat_acc",
            "IMU_VerAccelVal": "vertical_acc",
            "IMU_YawRtVal": "yaw_rate",
            "IMU_RollRtVal": "imu_roll_rate",
            "Pitch_rate_6D": "pitch_rate",
            "Bounce_rate_6D": "bounce_rate",
            "Roll_rate_6D": "roll_rate",
            "SAS_AnglVal": "steer_angle",
            "SAS_SpdVal": "steer_speed",
            "VCU_AccPedDepVal": "accel_pedal",
            "IEB_EstTtlBrkFrcNmV": "brake_force",
        }
        return mapping.get(name, name)

    def _add_named_interactions(self, out):
        def get(name):
            return out.get(name, 0.0)

        out["long_slope_x_pitch_peak"] = get("long_acc_slope_rms") * get("pitch_rate_abs_peak")
        out["long_slope_x_bounce_energy"] = get("long_acc_slope_rms") * get("bounce_rate_energy")
        out["vertical_peak_x_pitch_peak"] = get("vertical_acc_abs_peak") * get("pitch_rate_abs_peak")
        out["vertical_peak_x_bounce_energy"] = get("vertical_acc_abs_peak") * get("bounce_rate_energy")
        out["pitch_x_bounce_energy"] = get("pitch_rate_energy") * get("bounce_rate_energy")


class SoftBasisTransformer:
    """Standardized linear + soft hinge bases for sparse additive preference models."""

    def __init__(self, knots=(-1.0, 0.0, 1.0), include_below=True, eps=1e-6):
        self.knots = tuple(float(k) for k in knots)
        self.include_below = include_below
        self.eps = eps

    def fit(self, raw_features, raw_names):
        raw_features = np.asarray(raw_features, dtype=np.float64)
        self.raw_names = list(raw_names)
        self.mean = raw_features.mean(axis=0)
        self.std = raw_features.std(axis=0) + self.eps
        self.feature_names = ["bias"]
        self.groups = ["bias"]

        for name, mu, sd in zip(self.raw_names, self.mean, self.std):
            self.feature_names.append(f"{name}:linear")
            self.groups.append(name)
            for k in self.knots:
                raw_thr = mu + k * sd
                self.feature_names.append(f"{name}:above_{raw_thr:.4g}")
                self.groups.append(name)
                if self.include_below:
                    self.feature_names.append(f"{name}:below_{raw_thr:.4g}")
                    self.groups.append(name)
        return self

    def transform(self, raw_features):
        raw_features = np.asarray(raw_features, dtype=np.float64)
        z = (raw_features - self.mean) / self.std
        cols = [np.ones((z.shape[0], 1), dtype=np.float64)]
        for j in range(z.shape[1]):
            zj = z[:, j:j + 1]
            cols.append(zj)
            for k in self.knots:
                cols.append(np.maximum(0.0, zj - k))
                if self.include_below:
                    cols.append(np.maximum(0.0, k - zj))
        return np.concatenate(cols, axis=1).astype(np.float32)

    def fit_transform(self, raw_features, raw_names):
        self.fit(raw_features, raw_names)
        return self.transform(raw_features)

    def group_contributions(self, theta, phi, cov=None, top_k=12):
        theta = np.asarray(theta)
        phi = np.asarray(phi)
        if phi.ndim == 2:
            contrib = np.mean(phi * theta[None, :], axis=0)
        else:
            contrib = phi * theta

        by_group = {}
        for idx, group in enumerate(self.groups):
            if group == "bias":
                continue
            rec = by_group.setdefault(group, {"signed": 0.0, "abs": 0.0, "var": 0.0})
            rec["signed"] += float(contrib[idx])
            rec["abs"] += float(abs(contrib[idx]))
            if cov is not None:
                rec["var"] += float(cov[idx, idx])

        rows = []
        for group, rec in by_group.items():
            uncertainty = float(np.sqrt(max(rec["var"], 0.0))) if cov is not None else None
            rows.append((group, rec["signed"], rec["abs"], uncertainty))
        rows.sort(key=lambda r: r[2], reverse=True)
        return rows[:top_k]
