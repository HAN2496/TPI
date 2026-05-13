import hashlib
import json
import math
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import train_test_split, StratifiedKFold

from .catalog import get_catalog
from .loader import DatasetManager, Dataset
from .transforms import apply_smoothing


RAW_CACHE_VERSION = 1


def _raw_cache_key(driver, episodes, root):
    records = []
    for ep in episodes:
        csv_path = root / ep["dirpath"] / ep["csv"]
        stat = csv_path.stat()
        records.append({
            "id": ep["id"],
            "csv": str(Path(ep["dirpath"]) / ep["csv"]),
            "n_timesteps": ep["n_timesteps"],
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        })
    payload = {
        "version": RAW_CACHE_VERSION,
        "driver": driver,
        "episodes": sorted(records, key=lambda r: r["id"]),
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _load_raw_driver_cache(root, driver, episodes):
    cache_dir = root / ".seqcache" / "raw" / f"v{RAW_CACHE_VERSION}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{_raw_cache_key(driver, episodes, root)}.npz"

    if cache_path.exists():
        try:
            arr = np.load(cache_path, allow_pickle=False)
            meta = json.loads(arr["meta_json"].item())
            return [
                {
                    "meta": rec,
                    "states": arr[f"states_{i}"],
                    "columns": arr[f"columns_{i}"].astype(str).tolist(),
                }
                for i, rec in enumerate(meta)
            ]
        except Exception:
            cache_path.unlink()

    meta = []
    arrays = {}
    for i, ep in enumerate(episodes):
        csv_path = root / ep["dirpath"] / ep["csv"]
        states = pd.read_csv(csv_path)
        meta.append({
            "id": ep["id"],
            "label": ep["label"],
            "dirpath": ep["dirpath"],
            "csv": ep["csv"],
            "n_timesteps": ep["n_timesteps"],
        })
        arrays[f"states_{i}"] = states.to_numpy(dtype=np.float32)
        arrays[f"columns_{i}"] = np.asarray(states.columns, dtype=str)

    arrays["meta_json"] = np.asarray(json.dumps(meta, ensure_ascii=False))
    np.savez(cache_path, **arrays)
    return [
        {
            "meta": rec,
            "states": arrays[f"states_{i}"],
            "columns": arrays[f"columns_{i}"].astype(str).tolist(),
        }
        for i, rec in enumerate(meta)
    ]


def _sequence_from_states(states, feature_cols, time_range, fill_value, pad):
    start_time, end_time = time_range
    duration = end_time - start_time
    seg = states[(states["Time"] >= start_time) & (states["Time"] < end_time)]
    if len(seg) == 0:
        return None, None

    time_vals = seg["Time"].values.astype(np.float32)
    cols = feature_cols if feature_cols is not None else [col for col in seg.columns if col != "Time"]
    seg_features = seg[cols].fillna(fill_value)
    values = seg_features.values.astype(np.float32)

    if pad:
        time_diffs = states["Time"].diff().dropna()
        dt = time_diffs.median() if len(time_diffs) else None
        if dt is not None and np.isfinite(dt) and dt > 0:
            expected_len = int(np.round(duration / dt))
            current_len = len(values)

            if current_len < expected_len:
                pad_len = expected_len - current_len
                pad_array = np.full((pad_len, values.shape[1]), fill_value, dtype=np.float32)
                values = np.vstack([values, pad_array])
                last_time = time_vals[-1] if len(time_vals) > 0 else start_time
                time_pad = np.arange(1, pad_len + 1) * dt + last_time
                time_vals = np.concatenate([time_vals, time_pad])
            elif current_len > expected_len:
                values = values[:expected_len]
                time_vals = time_vals[:expected_len]

    return time_vals, values


def load_sequences(driver, features, time_range, downsample, root="datasets",
                   smooth=False, smooth_cutoff=12.0, smooth_order=2):
    root = Path(root)
    catalog = get_catalog(root)
    episodes = catalog.query(drivers=[driver])
    assert episodes, f"No episodes for driver '{driver}' in catalog"

    t, X, y = [], [], []
    for item in _load_raw_driver_cache(root, driver, episodes):
        label = item["meta"]["label"]
        if label is None:
            continue

        states = pd.DataFrame(item["states"], columns=item["columns"])
        if features is not None:
            cols = ["Time"] + [col for col in features if col != "Time"]
            states = states[cols].copy()

        if smooth:
            states = apply_smoothing(states, cutoff=smooth_cutoff, order=smooth_order)
        if downsample > 1:
            states = states.iloc[::downsample].reset_index(drop=True)

        time_vals, values = _sequence_from_states(
            states, features, time_range, fill_value=0.0, pad=True)
        if values is None:
            continue

        t.append(time_vals)
        X.append(values)
        y.append(1 if label else 0)

    return np.stack(X), np.asarray(y, dtype=np.int64)


def _load_dataset_sequences(driver_name, time_range, downsample, config,
                             smooth=False, smooth_cutoff=12.0, smooth_order=2):
    return load_sequences(driver_name, config['features'], time_range, downsample,
                          smooth=smooth, smooth_cutoff=smooth_cutoff, smooth_order=smooth_order)


def _create_data_loaders(X_train, X_val, y_train, y_val, batch_size):
    train_ds = TensorDataset(
        torch.as_tensor(X_train, dtype=torch.float32),
        torch.as_tensor(y_train, dtype=torch.float32)
    )
    val_ds = TensorDataset(
        torch.as_tensor(X_val, dtype=torch.float32),
        torch.as_tensor(y_val, dtype=torch.float32)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)
    return train_loader, val_loader


def make_loaders(driver_name, config, time_range, train_downsample=1, val_downsample=None, normalize=False):
    if val_downsample is None:
        val_downsample = train_downsample

    load_downsample = math.gcd(train_downsample, val_downsample)
    X, y = load_sequences(driver_name, config['features'], time_range, load_downsample)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=config['test_size'], random_state=0, stratify=y)

    if train_downsample > load_downsample:
        X_train = X_train[:, ::train_downsample // load_downsample, :]
    if val_downsample > load_downsample:
        X_val = X_val[:, ::val_downsample // load_downsample, :]

    if normalize:
        mean = X_train.mean(axis=(0, 1), keepdims=True)
        std  = X_train.std(axis=(0, 1), keepdims=True) + 1e-6
        X_train = (X_train - mean) / std
        X_val   = (X_val   - mean) / std

    trainer_cfg = config.get("trainer", config)
    batch_size = trainer_cfg.get("batch_size", len(X_train))

    return _create_data_loaders(X_train, X_val, y_train, y_val, batch_size)


def make_kfold_loaders(driver_name, config, time_range, downsample=1, n_splits=5,
                       test_ratio=0.1, is_neural=True, random_state=42):
    X, y = load_sequences(driver_name, config['features'], time_range, downsample)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=random_state, stratify=y)

    trainval_ds = TensorDataset(
        torch.as_tensor(X_trainval, dtype=torch.float32),
        torch.as_tensor(y_trainval, dtype=torch.float32)
    )
    test_ds = TensorDataset(
        torch.as_tensor(X_test, dtype=torch.float32),
        torch.as_tensor(y_test, dtype=torch.float32)
    )
    test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_trainval_np = np.asarray(y_trainval)

    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y_trainval_np)), y_trainval_np)):
        train_subset = Subset(trainval_ds, train_idx)
        val_subset = Subset(trainval_ds, val_idx)

        batch_size = config["trainer"]["batch_size"] if is_neural else len(train_subset)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=len(val_subset), shuffle=False)

        folds.append((fold_idx, train_loader, val_loader))

    return test_loader, folds
