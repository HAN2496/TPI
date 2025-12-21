from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch
import yaml
from pathlib import Path

from .data_loader import DatasetManager
from ..configs.utils import load_features


def convert_driver_name(driver_name):
    if driver_name == "a":
        return "강신길"
    elif driver_name == "b":
        return "박재일"
    elif driver_name == "c":
        return "한규택"
    return driver_name


def save_config(config, driver_name, model_name, save_path='src/configs/config.yaml'):
    with open(save_path, 'r', encoding='utf-8') as f:
        all_configs = yaml.safe_load(f)

    if driver_name not in all_configs:
        all_configs[driver_name] = {}
    all_configs[driver_name][model_name] = config

    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(all_configs, f, default_flow_style=False, allow_unicode=True)


def _load_dataset_sequences(driver_name, time_range, downsample, feature_version):
    manager = DatasetManager("datasets", downsample=downsample)
    dataset = manager.get(driver_name)
    feature_cols = load_features(feature_version)
    t, X, y = dataset.to_sequences(feature_cols, time_range, fill_value=0.0, pad=True)
    return X, y


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


def prepare_training_data(driver_name, feature_version, config, time_range, train_downsample=1, val_downsample=None):
    if val_downsample is None:
        val_downsample = train_downsample
        X, y = _load_dataset_sequences(driver_name, time_range, train_downsample, feature_version)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=config['test_size'],
            random_state=0,
            stratify=y
        )
    else:
        X, y = _load_dataset_sequences(driver_name, time_range, 1, feature_version)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=config['test_size'],
            random_state=0,
            stratify=y
        )

        if train_downsample > 1:
            X_train = X_train[:, ::train_downsample, :]
        if val_downsample > 1:
            X_val = X_val[:, ::val_downsample, :]

    trainer_cfg = config.get("trainer", config)
    batch_size = trainer_cfg.get("batch_size", len(X_train))

    return _create_data_loaders(X_train, X_val, y_train, y_val, batch_size)


def prepare_training_data_kfold(driver_name, feature_version, config, time_range, train_downsample=1, val_downsample=None, n_splits=5, random_state=42):
    if val_downsample is None:
        val_downsample = train_downsample
        X, y = _load_dataset_sequences(driver_name, time_range, train_downsample, feature_version)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        trainer_cfg = config.get("trainer", config)
        batch_size = trainer_cfg.get("batch_size", 128)

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            train_loader, val_loader = _create_data_loaders(X_train, X_val, y_train, y_val, batch_size)

            yield fold_idx, train_loader, val_loader
    else:
        X, y = _load_dataset_sequences(driver_name, time_range, 1, feature_version)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        trainer_cfg = config.get("trainer", config)
        batch_size = trainer_cfg.get("batch_size", 128)

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            if train_downsample > 1:
                X_train = X_train[:, ::train_downsample, :]
            if val_downsample > 1:
                X_val = X_val[:, ::val_downsample, :]

            train_loader, val_loader = _create_data_loaders(X_train, X_val, y_train, y_val, batch_size)

            yield fold_idx, train_loader, val_loader