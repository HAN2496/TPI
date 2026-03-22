import os
import json
import pandas as pd
import numpy as np
import scipy.io
import xml.etree.ElementTree as ET
from pathlib import Path

from .transforms import apply_smoothing, apply_normalization


class Dataset:
    def __init__(self, name, base_folder='datasets', lazy=False, downsample=1, smooth=False, normalize=False,
                 episodes=None, **kwargs):
        self.name = name
        self.base_folder = Path(base_folder)
        self.lazy = lazy
        self.downsample = downsample
        self.smooth = smooth
        self.normalize = normalize
        self._episodes = episodes

        self.smooth_cutoff = kwargs.get('smooth_cutoff', 12.0)
        self.smooth_order = kwargs.get('smooth_order', 2)
        self.norm_method = kwargs.get('norm_method', 'minmax')

        self.data = {}
        self._loaded = False

        if not lazy:
            self._load()

    def _load(self):
        if self._loaded:
            return
        if self._episodes is not None:
            self._load_from_catalog()
        else:
            self._load_from_scan()

    def _load_from_catalog(self):
        for idx, ep in enumerate(self._episodes):
            csv_path = self.base_folder / ep["dirpath"] / ep["csv"]
            states = pd.read_csv(csv_path)

            if self.smooth:
                states = apply_smoothing(states, cutoff=self.smooth_cutoff, order=self.smooth_order)
            if self.downsample > 1:
                states = states.iloc[::self.downsample].reset_index(drop=True)
            if self.normalize:
                states = apply_normalization(states, method=self.norm_method)

            dt = states["Time"].diff().dropna().median()
            self.data[idx] = {
                "timestamp": ep["id"],
                "label": ep["label"],
                "states": states,
                "info": {},
                "dt": dt,
            }
        self._loaded = True

    def _load_from_scan(self):
        idx = 0
        for dirpath, dirnames, filenames in os.walk(self.base_folder):
            dirnames.sort()
            filenames.sort()
            csv_files = [f for f in filenames if f.lower().endswith(".csv") and "_state_" in f and not f.lower().endswith("_smooth.csv")]
            txt_files = [f for f in filenames if f.lower().endswith(".txt") and "_info_" in f]
            csv_by_ts = {f.split("_state_")[0]: f for f in csv_files}
            txt_by_ts = {f.split("_info_")[0]: f for f in txt_files}

            common_ts = set(csv_by_ts.keys()) & set(txt_by_ts.keys())

            for ts in sorted(common_ts):
                info_path = os.path.join(dirpath, txt_by_ts[ts])
                with open(info_path, "r", encoding="utf-8") as f:
                    info = json.load(f)

                if info.get("Driver") == self.name:
                    csv_path = os.path.join(dirpath, csv_by_ts[ts])
                    states = pd.read_csv(csv_path)

                    if self.smooth:
                        states = apply_smoothing(states, cutoff=self.smooth_cutoff, order=self.smooth_order)
                    if self.downsample > 1:
                        states = states.iloc[::self.downsample].reset_index(drop=True)

                    if self.normalize:
                        states = apply_normalization(states, method=self.norm_method)

                    time_diffs = states["Time"].diff().dropna()
                    dt = time_diffs.median()

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

    def to_sequences(self, feature_cols=None, time_range=(5, 8), fill_value=0.0, pad=False):
        self._ensure_loaded()

        t, X, y = [], [], []
        start_time, end_time = time_range
        duration = end_time - start_time

        for item in self.data.values():
            if item["label"] is None:
                continue

            states = item["states"]
            seg = states[(states["Time"] >= start_time) & (states["Time"] < end_time)]
            if len(seg) == 0:
                continue

            time_vals = seg["Time"].values.astype(np.float32)
            cols = feature_cols if feature_cols is not None else [col for col in seg.columns if col != "Time"]
            seg_features = seg[cols].fillna(fill_value)
            v = seg_features.values.astype(np.float32)

            if pad and item["dt"] is not None:
                expected_len = int(np.round(duration / item["dt"]))
                current_len = len(v)

                if current_len < expected_len:
                    pad_len = expected_len - current_len
                    pad_array = np.full((pad_len, v.shape[1]), fill_value, dtype=np.float32)
                    v = np.vstack([v, pad_array])
                    last_time = time_vals[-1] if len(time_vals) > 0 else start_time
                    time_pad = np.arange(1, pad_len + 1) * item["dt"] + last_time
                    time_vals = np.concatenate([time_vals, time_pad])
                elif current_len > expected_len:
                    v = v[:expected_len]
                    time_vals = time_vals[:expected_len]

            t.append(time_vals)
            X.append(v)
            y.append(1 if item["label"] else 0)

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
        self.kwargs = kwargs
        self.datasets = {}
        self._folder_index = {}
        self._scan()

    def _scan(self):
        drivers = set()

        for dirpath, dirnames, filenames in os.walk(self.base_folder):
            dirnames.sort()
            filenames.sort()
            csv_files = [f for f in filenames if f.lower().endswith(".csv") and "_state_" in f and not f.lower().endswith("_smooth.csv")]
            txt_files = [f for f in filenames if f.lower().endswith(".txt") and "_info_" in f]
            csv_by_ts = {f.split("_state_")[0]: f for f in csv_files}
            txt_by_ts = {f.split("_info_")[0]: f for f in txt_files}
            common_ts = set(csv_by_ts.keys()) & set(txt_by_ts.keys())

            if not common_ts:
                continue

            folder_key = str(Path(dirpath))
            if folder_key not in self._folder_index:
                self._folder_index[folder_key] = {}

            for ts in sorted(common_ts):
                info_path = os.path.join(dirpath, txt_by_ts[ts])
                csv_path = os.path.join(dirpath, csv_by_ts[ts])

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

                with open(info_path, "r", encoding="utf-8") as f:
                    info = json.load(f)
                driver = info.get("Driver")
                if driver:
                    drivers.add(driver)

        for driver in sorted(drivers):
            self.datasets[driver] = Dataset(driver, self.base_folder, lazy=self.lazy,
                                            downsample=self.downsample, smooth=self.smooth,
                                            normalize=self.normalize, **self.kwargs)

    def get(self, driver_name):
        return self.datasets.get(driver_name)

    def keys(self):
        return list(self.datasets.keys())

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, name):
        return self.datasets[name]

    def __iter__(self):
        return iter(self.datasets.items())

    def get_file_list(self, folder_path=None):
        if folder_path is None:
            folder_path = str(self.base_folder)
        folder_key = str(Path(folder_path))
        return sorted(self._folder_index.get(folder_key, {}).keys())

    def load_all(self, timestamp, folder_path=None):
        if folder_path is None:
            folder_path = str(self.base_folder)
        folder_key = str(Path(folder_path))

        file_info = self._folder_index[folder_key][timestamp]

        csv_data = pd.read_csv(file_info['csv_path'])
        with open(file_info['info_path'], 'r', encoding='utf-8') as f:
            info = json.load(f)

        sensors = {}
        if file_info['mat_path'] and os.path.exists(file_info['mat_path']):
            mat_data = scipy.io.loadmat(file_info['mat_path'])
            sensors = {k: v for k, v in mat_data.items() if not k.startswith('__')}

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

        return {'info': info, 'sensors': sensors, 'gps': gps_coords}

    def get_video_path(self, timestamp, folder_path=None):
        if folder_path is None:
            folder_path = str(self.base_folder)
        folder_key = str(Path(folder_path))

        if folder_key not in self._folder_index:
            return None
        if timestamp not in self._folder_index[folder_key]:
            return None

        return self._folder_index[folder_key][timestamp].get('video_path')
