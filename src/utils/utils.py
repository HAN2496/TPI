import math
import torch
from ruamel.yaml import YAML

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold

from .data_loader import DatasetManager

def convert_driver_name(driver_name):
    if driver_name == "a" or driver_name == "kang":
        return "강신길"
    elif driver_name == "b" or driver_name == "park":
        return "박재일"
    elif driver_name == "c" or driver_name == "han":
        return "한규택"
    return driver_name

def save_config(config, driver_name, model_type, model_name, save_path=None):
    if save_path is None:
        save_path = f'src/configs/drivers/{driver_name}.yaml'

    ryaml = YAML()
    ryaml.preserve_quotes = True
    ryaml.default_flow_style = None
    ryaml.width = 4096

    with open(save_path, 'r', encoding='utf-8') as f:
        all_configs = ryaml.load(f)

    if model_type not in all_configs:
        all_configs[model_type] = {}

    config_to_save = {k: v for k, v in config.items() if k != 'model_type'}

    all_configs[model_type][model_name] = config_to_save

    with open(save_path, 'w', encoding='utf-8') as f:
        ryaml.dump(all_configs, f)


def _load_dataset_sequences(driver_name, time_range, downsample, config):
    manager = DatasetManager("datasets", downsample=downsample)
    dataset = manager.get(driver_name)
    t, X, y = dataset.to_sequences(config['features'], time_range, fill_value=0.0, pad=True)
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


def prepare_training_data(driver_name, config, time_range, train_downsample=1, val_downsample=None):
    if val_downsample is None:
        val_downsample = train_downsample

    load_downsample = math.gcd(train_downsample, val_downsample)
    X, y = _load_dataset_sequences(driver_name, time_range, load_downsample, config)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=config['test_size'], random_state=0, stratify=y)

    if train_downsample > load_downsample:
        X_train = X_train[:, ::train_downsample // load_downsample, :]
    if val_downsample > load_downsample:
        X_val = X_val[:, ::val_downsample // load_downsample, :]

    trainer_cfg = config.get("trainer", config)
    batch_size = trainer_cfg.get("batch_size", len(X_train))

    return _create_data_loaders(X_train, X_val, y_train, y_val, batch_size)


def prepare_training_data_kfold_old(driver_name, config, time_range, train_downsample=1, val_downsample=None, n_splits=5, random_state=42):
    if val_downsample is None:
        val_downsample = train_downsample

    load_downsample = math.gcd(train_downsample, val_downsample)
    X, y = _load_dataset_sequences(driver_name, time_range, load_downsample, config)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    trainer_cfg = config.get("trainer", config)
    batch_size = trainer_cfg.get("batch_size", 128)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if train_downsample > load_downsample:
            X_train = X_train[:, ::train_downsample // load_downsample, :]
        if val_downsample > load_downsample:
            X_val = X_val[:, ::val_downsample // load_downsample, :]

        train_loader, val_loader = _create_data_loaders(X_train, X_val, y_train, y_val, batch_size)
        yield fold_idx, train_loader, val_loader


def prepare_training_data_kfold(driver_name, config, time_range, train_downsample=1, val_downsample=None, n_splits=5, test_ratio=0.1, random_state=42):
    if val_downsample is None:
        val_downsample = train_downsample

    load_downsample = math.gcd(train_downsample, val_downsample)
    X, y = _load_dataset_sequences(driver_name, time_range, load_downsample, config)

    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=test_ratio, random_state=random_state, stratify=y)

    if val_downsample > load_downsample:
        X_test = X_test[:, ::val_downsample // load_downsample, :]

    test_ds = TensorDataset(torch.as_tensor(X_test, dtype=torch.float32), torch.as_tensor(y_test, dtype=torch.float32))
    test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)

    def fold_generator():
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        trainer_cfg = config.get("trainer", config)
        batch_size = trainer_cfg.get("batch_size", 128)

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_trainval, y_trainval)):
            X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
            y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]

            if train_downsample > load_downsample:
                X_train = X_train[:, ::train_downsample // load_downsample, :]
            if val_downsample > load_downsample:
                X_val = X_val[:, ::val_downsample // load_downsample, :]

            train_loader, val_loader = _create_data_loaders(X_train, X_val, y_train, y_val, batch_size)
            yield fold_idx, train_loader, val_loader

    return test_loader, fold_generator()