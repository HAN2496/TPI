import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from src.utils.data_loader import DatasetManager
from src.configs.utils import load_features

def prepare_training_data(driver_name, config, time_range, downsample, flatten=False):
    manager = DatasetManager("datasets", downsample=downsample)
    dataset = manager.get(driver_name)

    feature_cols = load_features(config['feature_version'])
    t, X, y = dataset.to_sequences(feature_cols, time_range, fill_value=0.0, pad=True)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=config['test_size'],
        random_state=0,
        stratify=y
    )

    if flatten:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_val = X_val.reshape(X_val.shape[0], -1)
        return None, X_train, y_train, X_val, y_val

    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).float()
    X_val_t = torch.from_numpy(X_val).float()
    y_val_t = torch.from_numpy(y_val).float()

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=config['trainer']['batch_size'], shuffle=True)

    return train_loader, X_train_t, y_train_t, X_val_t, y_val_t