import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import xml.etree.ElementTree as ET
from pathlib import Path
from scipy.signal import butter, filtfilt


def design_lpf(fs, cutoff, order=2):
    """
    Design Butterworth Low-Pass Filter

    Args:
        fs: Sampling frequency [Hz]
        cutoff: Cutoff frequency [Hz]
        order: Filter order

    Returns:
        b, a: Filter coefficients
    """
    nyq = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyq  # Normalized cutoff frequency (0~1)
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def apply_smoothing(states, cutoff=2.0, order=2):
    """
    Apply Low-Pass Filter to states DataFrame

    Args:
        states: DataFrame with Time column
        cutoff: Cutoff frequency [Hz]
        order: Filter order

    Returns:
        Smoothed DataFrame
    """
    # Estimate sampling frequency
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
        # Bidirectional filtering to remove phase delay
        smoothed[col] = filtfilt(b, a, x)

    return smoothed


def apply_normalization(states, method='minmax'):
    """
    Apply normalization to states DataFrame

    Args:
        states: DataFrame with Time column
        method: Normalization method ('minmax' or 'zscore')

    Returns:
        Normalized DataFrame
    """
    normalized = states.copy()

    for col in states.columns:
        if col == 'Time':
            continue

        values = states[col].values

        if method == 'minmax':
            # Min-Max normalization: scale to [0, 1]
            min_val = values.min()
            max_val = values.max()
            if max_val - min_val > 1e-8:  # Avoid division by zero
                normalized[col] = (values - min_val) / (max_val - min_val)
            else:
                normalized[col] = 0.0

        elif method == 'zscore':
            # Z-score normalization: mean=0, std=1
            mean_val = values.mean()
            std_val = values.std()
            if std_val > 1e-8:  # Avoid division by zero
                normalized[col] = (values - mean_val) / std_val
            else:
                normalized[col] = 0.0

        else:
            raise ValueError(f"Unknown normalization method: {method}")

    return normalized


class Dataset:
    def __init__(self, name, base_folder='datasets', lazy=False, downsample=1, smooth=False, normalize=False, **kwargs):
        self.name = name
        self.base_folder = Path(base_folder)
        self.lazy = lazy
        self.downsample = downsample
        self.smooth = smooth
        self.normalize = normalize

        # Smoothing parameters
        self.smooth_cutoff = kwargs.get('smooth_cutoff', 12.0)
        self.smooth_order = kwargs.get('smooth_order', 2)

        # Normalization parameters
        self.norm_method = kwargs.get('norm_method', 'minmax')

        self.data = {}
        self._loaded = False

        if not lazy:
            self._load()

    def _load(self):
        if self._loaded:
            return

        idx = 0
        for dirpath, dirnames, filenames in os.walk(self.base_folder):
            # Only load non-smooth CSV files (raw data)
            csv_files = [f for f in filenames if f.lower().endswith(".csv") and "_state_" in f and not f.lower().endswith("_smooth.csv")]
            txt_files = [f for f in filenames if f.lower().endswith(".txt") and "_info_" in f]
            csv_by_ts = {f.split("_state_")[0]: f for f in csv_files}
            txt_by_ts = {f.split("_info_")[0]: f for f in txt_files}

            common_ts = set(csv_by_ts.keys()) & set(txt_by_ts.keys())

            for ts in common_ts:
                info_path = os.path.join(dirpath, txt_by_ts[ts])
                with open(info_path, "r", encoding="utf-8") as f:
                    info = json.load(f)

                if info.get("Driver") == self.name:
                    csv_path = os.path.join(dirpath, csv_by_ts[ts])
                    states = pd.read_csv(csv_path)

                    # Apply downsampling
                    if self.downsample > 1:
                        states = states.iloc[::self.downsample].reset_index(drop=True)

                    # Apply smoothing if requested
                    if self.smooth:
                        states = apply_smoothing(states, cutoff=self.smooth_cutoff, order=self.smooth_order)

                    # Apply normalization if requested
                    if self.normalize:
                        states = apply_normalization(states, method=self.norm_method)

                    time_diffs = states["Time"].diff().dropna()
                    dt = time_diffs.median()

                    # Extract label from filename
                    base = os.path.splitext(csv_by_ts[ts])[0]
                    label_raw = base.split("_")[-1]
                    if label_raw == "True":
                        label = True
                    elif label_raw == "False":
                        label = False
                    else:
                        label = None

                    self.data[idx] = {
                        'timestamp': ts,
                        'label': label,
                        'states': states,
                        'info': info,
                        'dt': dt,
                    }
                    idx += 1

        self._loaded = True

    def _ensure_loaded(self):
        if not self._loaded:
            self._load()

    def __len__(self):
        self._ensure_loaded()
        return len(self.data)

    def __getitem__(self, idx):
        self._ensure_loaded()
        return self.data[idx]

    def __iter__(self):
        self._ensure_loaded()
        return iter(self.data.values())

    def to_sequences(self, feature_cols, time_range=(5, 8), fill_value=0.0, pad=False):
        """
        Convert dataset to sequences for ML training

        Args:
            feature_cols: List of column names to use as features
            time_range: Tuple of (start_time, end_time) to extract
            fill_value: Value to use for fillna and padding
            pad: If True, pad sequences to expected length based on dt

        Returns:
            t: List of numpy arrays of time values or stacked array if pad=True
            X: List of numpy arrays of shape (seq_len, n_features) or stacked array if pad=True
            y: numpy array of shape (n_samples,) with binary labels (1=True, 0=False)
        """
        self._ensure_loaded()

        t, X, y = [], [], []
        start_time, end_time = time_range
        duration = end_time - start_time

        for item in self.data.values():
            # Skip items without label
            if item["label"] is None:
                continue

            states = item["states"]

            # Extract time segment
            seg = states[(states["Time"] >= start_time) & (states["Time"] < end_time)]
            if len(seg) == 0:
                continue

            # Extract time values
            time_vals = seg["Time"].values.astype(np.float32)

            # Select features and fill NaN
            seg_features = seg[feature_cols].fillna(fill_value)
            v = seg_features.values.astype(np.float32)

            # Apply padding if requested
            if pad and item["dt"] is not None:
                expected_len = int(np.round(duration / item["dt"]))
                current_len = len(v)

                if current_len < expected_len:
                    pad_len = expected_len - current_len
                    pad_array = np.full((pad_len, v.shape[1]), fill_value, dtype=np.float32)
                    v = np.vstack([v, pad_array])

                    # Pad time values
                    last_time = time_vals[-1] if len(time_vals) > 0 else start_time
                    time_pad = np.arange(1, pad_len + 1) * item["dt"] + last_time
                    time_vals = np.concatenate([time_vals, time_pad])
                elif current_len > expected_len:
                    v = v[:expected_len]
                    time_vals = time_vals[:expected_len]

            t.append(time_vals)
            X.append(v)
            y.append(1 if item["label"] else 0)

        # Stack if pad=True (all same length), otherwise return list
        if pad and t:
            t = np.stack(t)
            X = np.stack(X)

        y = np.array(y, dtype=np.int64)

        return t, X, y


class DatasetManager:
    def __init__(self, base_folder='datasets', lazy=True, downsample=1, smooth=False, normalize=False, **kwargs):
        self.base_folder = Path(base_folder)
        self.lazy = lazy
        self.downsample = downsample
        self.smooth = smooth
        self.normalize = normalize
        self.kwargs = kwargs  # Store additional kwargs to pass to Dataset
        self.datasets = {}
        self._folder_index = {}  # folder_path -> {timestamp -> data_info}
        self._scan()

    def _scan(self):
        """Scan all data and group by driver name + build folder index"""
        drivers = set()

        for dirpath, dirnames, filenames in os.walk(self.base_folder):
            # Only scan non-smooth CSV files (raw data)
            csv_files = [f for f in filenames if f.lower().endswith(".csv") and "_state_" in f and not f.lower().endswith("_smooth.csv")]
            txt_files = [f for f in filenames if f.lower().endswith(".txt") and "_info_" in f]
            csv_by_ts = {f.split("_state_")[0]: f for f in csv_files}
            txt_by_ts = {f.split("_info_")[0]: f for f in txt_files}
            common_ts = set(csv_by_ts.keys()) & set(txt_by_ts.keys())

            if not common_ts:
                continue

            # Build folder index
            folder_key = str(Path(dirpath))
            if folder_key not in self._folder_index:
                self._folder_index[folder_key] = {}

            for ts in common_ts:
                info_path = os.path.join(dirpath, txt_by_ts[ts])
                csv_path = os.path.join(dirpath, csv_by_ts[ts])

                # Find related files
                video_path = None
                mat_path = None
                gps_path = None

                video_files = [f for f in filenames if f.startswith(ts) and f.endswith(".mp4")]
                mat_files = [f for f in filenames if f.startswith(ts) and f.endswith(".mat")]
                gps_files = [f for f in filenames if f.startswith(ts) and f.endswith(".kml")]

                if video_files:
                    video_path = os.path.join(dirpath, video_files[0])
                if mat_files:
                    mat_path = os.path.join(dirpath, mat_files[0])
                if gps_files:
                    gps_path = os.path.join(dirpath, gps_files[0])

                self._folder_index[folder_key][ts] = {
                    'csv_path': csv_path,
                    'info_path': info_path,
                    'video_path': video_path,
                    'mat_path': mat_path,
                    'gps_path': gps_path,
                }

                # Collect drivers
                with open(info_path, "r", encoding="utf-8") as f:
                    info = json.load(f)
                driver = info.get("Driver")
                if driver:
                    drivers.add(driver)

        # Create Dataset for each driver
        for driver in drivers:
            self.datasets[driver] = Dataset(driver, self.base_folder, lazy=self.lazy,
                                           downsample=self.downsample, smooth=self.smooth,
                                           normalize=self.normalize, **self.kwargs)

    # --- Driver-based access ---
    def get(self, name):
        return self.datasets.get(name)

    def keys(self):
        return list(self.datasets.keys())

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, name):
        return self.datasets[name]

    def __iter__(self):
        return iter(self.datasets.items())

    # --- Folder-based access (for backward compatibility) ---
    def get_file_list(self, folder_path=None):
        """Get list of timestamps in a folder"""
        if folder_path is None:
            folder_path = str(self.base_folder)
        folder_key = str(Path(folder_path))
        return sorted(self._folder_index.get(folder_key, {}).keys())

    def load_all(self, timestamp, folder_path=None):
        """Load all data for a specific timestamp in a folder"""
        if folder_path is None:
            folder_path = str(self.base_folder)
        folder_key = str(Path(folder_path))

        if folder_key not in self._folder_index:
            raise KeyError(f"Folder {folder_path} not found")
        if timestamp not in self._folder_index[folder_key]:
            raise KeyError(f"Timestamp {timestamp} not found in {folder_path}")

        file_info = self._folder_index[folder_key][timestamp]

        # Load CSV
        csv_data = pd.read_csv(file_info['csv_path'])

        # Load info
        with open(file_info['info_path'], 'r', encoding='utf-8') as f:
            info = json.load(f)

        # Load MAT if exists
        sensors = {}
        if file_info['mat_path'] and os.path.exists(file_info['mat_path']):
            mat_data = scipy.io.loadmat(file_info['mat_path'])
            sensors = {k: v for k, v in mat_data.items() if not k.startswith('__')}

        # Load GPS if exists
        gps_coords = []
        if file_info['gps_path'] and os.path.exists(file_info['gps_path']):
            tree = ET.parse(file_info['gps_path'])
            root = tree.getroot()
            ns = {'kml': 'http://www.opengis.net/kml/2.2'}
            coordinates_elem = root.find('.//kml:coordinates', ns)
            if coordinates_elem is not None:
                coords_text = coordinates_elem.text.strip()
                for coord in coords_text.split():
                    if coord:
                        lon, lat, alt = map(float, coord.split(','))
                        gps_coords.append((lon, lat, alt))

        return {
            'info': info,
            'sensors': sensors,
            'gps': gps_coords
        }

    def get_video_path(self, timestamp, folder_path=None):
        """Get video path for a specific timestamp"""
        if folder_path is None:
            folder_path = str(self.base_folder)
        folder_key = str(Path(folder_path))

        if folder_key not in self._folder_index:
            return None
        if timestamp not in self._folder_index[folder_key]:
            return None

        return self._folder_index[folder_key][timestamp].get('video_path')
