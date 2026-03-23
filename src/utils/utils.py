import os
import math
import json
import random
import hashlib
import torch
import numpy as np
from ruamel.yaml import YAML

from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import train_test_split, StratifiedKFold

from .data_loader import DatasetManager

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(torch.cuda.current_device())

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


def get_cache_filename(drivers, time_range, downsample, features):
    config_dict = {
        'drivers': sorted(drivers) if isinstance(drivers, (list, tuple)) else [drivers],
        'time_range': time_range,
        'downsample': downsample,
        'features': sorted(features)
    }
    config_str = json.dumps(config_dict, sort_keys=True)
    hash_str = hashlib.md5(config_str.encode()).hexdigest()
    return f"dataset_{hash_str}.npz"


def load_cached_datasets(driver_names, time_range, downsample, config, cache_dir="datasets/cached"):
    os.makedirs(cache_dir, exist_ok=True)
    
    if isinstance(driver_names, str):
        driver_names = [driver_names]
        
    cache_file = get_cache_filename(driver_names, time_range, downsample, config['features'])
    cache_path = os.path.join(cache_dir, cache_file)
    
    if os.path.exists(cache_path):
        print(f" Loading cached dataset from: {cache_path}")
        data = np.load(cache_path)
        return data['X'], data['y']
        
    print(" Cache not found. Processing and merging raw data...")
    all_X, all_y = [], []
    for driver in driver_names:
        X, y = _load_dataset_sequences(driver, time_range, downsample, config)
        all_X.append(X)
        all_y.extend(y)
        
    X_concat = np.concatenate(all_X, axis=0)
    y_concat = np.array(all_y)
    
    print(f" Saving dataset to cache: {cache_path}")
    np.savez_compressed(cache_path, X=X_concat, y=y_concat)
    
    return X_concat, y_concat


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

def prepare_training_data_kfold(driver_name, config, time_range, downsample=1, n_splits=5, test_ratio=0.1, is_neural=True, random_state=42):
    X, y = _load_dataset_sequences(driver_name, time_range, downsample, config)

    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=test_ratio, random_state=random_state, stratify=y)

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

    folds = []
    y_trainval_np = np.asarray(y_trainval)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y_trainval_np)), y_trainval_np)):
        train_subset = Subset(trainval_ds, train_idx)
        val_subset   = Subset(trainval_ds, val_idx)

        batch_size = config["trainer"]["batch_size"] if is_neural else len(train_subset)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_subset, batch_size=len(val_subset), shuffle=False)

        folds.append((fold_idx, train_loader, val_loader))

    return test_loader, folds
