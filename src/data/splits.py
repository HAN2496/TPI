import hashlib
import json
import math
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import train_test_split, StratifiedKFold

from .catalog import get_catalog
from .loader import DatasetManager, Dataset


def _cache_key(driver, features, time_range, downsample, episodes, smooth, smooth_cutoff, smooth_order):
    payload = {
        "driver": driver,
        "features": list(features),
        "time_range": list(time_range),
        "downsample": downsample,
        "episodes": sorted((e["id"], e["n_timesteps"]) for e in episodes),
        "smooth": smooth,
        "smooth_cutoff": smooth_cutoff,
        "smooth_order": smooth_order,
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_sequences(driver, features, time_range, downsample, root="datasets",
                   smooth=False, smooth_cutoff=12.0, smooth_order=2):
    root = Path(root)
    catalog = get_catalog(root)
    episodes = catalog.query(drivers=[driver])
    assert episodes, f"No episodes for driver '{driver}' in catalog"

    key = _cache_key(driver, features, time_range, downsample, episodes, smooth, smooth_cutoff, smooth_order)
    cache_dir = root / ".seqcache"
    cache_path = cache_dir / f"{key}.npz"

    if cache_path.exists():
        try:
            arr = np.load(cache_path)
            return arr["X"], arr["y"]
        except Exception:
            cache_path.unlink()

    ds = Dataset(driver, base_folder=root, downsample=downsample, episodes=episodes,
                 smooth=smooth, smooth_cutoff=smooth_cutoff, smooth_order=smooth_order)
    _, X, y = ds.to_sequences(features, time_range, fill_value=0.0, pad=True)
    cache_dir.mkdir(exist_ok=True)
    np.savez(cache_path, X=X, y=y)
    return X, y


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
