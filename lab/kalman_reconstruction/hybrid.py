from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class ResidualLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, 2, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.SiLU(),
                                  nn.Linear(hidden_size // 2, 1))

    def forward(self, x):
        return self.head(self.lstm(x)[0])[..., 0]


def kf_features(state, az_g, state_space):
    predicted = np.zeros_like(state)
    predicted[:, 1:] = state[:, :-1] @ state_space.A.T
    innovation = (az_g - 1) * 9.81 - (predicted @ state_space.H.T)[..., 0]
    return np.concatenate((state, innovation[..., None]), -1).astype(np.float32)


def normalize(features, residual, train):
    x_mean = features[train].mean((0, 1))
    x_std = features[train].std((0, 1)) + 1e-6
    y_mean = residual[train].mean()
    y_std = residual[train].std() + 1e-6
    return (features - x_mean) / x_std, (residual - y_mean) / y_std, x_mean, x_std, y_mean, y_std


def _mse(model, loader, device):
    model.eval()
    total = count = 0
    with torch.no_grad():
        for x, y in loader:
            total += nn.functional.mse_loss(model(x.to(device)), y.to(device), reduction="sum").item()
            count += y.numel()
    return total / count


def fit(model, x, y, train, validation, device, epochs=30, batch_size=32):
    train_loader = DataLoader(TensorDataset(torch.from_numpy(x[train]), torch.from_numpy(y[train])),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(x[validation]), torch.from_numpy(y[validation])),
                            batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.2, patience=3)
    best, best_state, stale = np.inf, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(xb.to(device)), yb.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        value = _mse(model, val_loader, device)
        scheduler.step(value)
        print(f"epoch={epoch:02d} val_mse={value:.6f}")
        if value < best - 1e-5:
            best, best_state, stale = value, deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= 8:
                break
    model.load_state_dict(best_state)
    return best, epoch


def predict(model, x, device, batch_size=32):
    loader = DataLoader(torch.from_numpy(x), batch_size=batch_size)
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(batch.to(device)).cpu().numpy() for batch in loader])
