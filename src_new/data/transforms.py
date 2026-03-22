import numpy as np
from scipy.signal import butter, filtfilt


def design_lpf(fs, cutoff, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def apply_smoothing(states, cutoff=2.0, order=2):
    t = states['Time'].values
    dt = np.median(np.diff(t))
    if dt <= 0:
        return states.copy()

    fs = 1.0 / dt
    b, a = design_lpf(fs, cutoff=cutoff, order=order)

    smoothed = states.copy()
    for col in states.columns:
        if col == 'Time':
            continue
        x = states[col].values
        smoothed[col] = filtfilt(b, a, x)

    return smoothed


def apply_normalization(states, method='minmax'):
    normalized = states.copy()

    for col in states.columns:
        if col == 'Time':
            continue

        values = states[col].values

        if method == 'minmax':
            min_val = values.min()
            max_val = values.max()
            if max_val - min_val > 1e-8:
                normalized[col] = (values - min_val) / (max_val - min_val)
            else:
                normalized[col] = 0.0

        elif method == 'zscore':
            mean_val = values.mean()
            std_val = values.std()
            if std_val > 1e-8:
                normalized[col] = (values - mean_val) / std_val
            else:
                normalized[col] = 0.0

        else:
            raise ValueError(f"Unknown normalization method: {method}")

    return normalized
