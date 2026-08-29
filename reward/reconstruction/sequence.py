from contextlib import nullcontext
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


MODEL_FREE_MODELS = (
    "lstm_online",
    "gru_online",
    "transformer_online",
    "bilstm_offline",
    "unet_offline",
    "transformer_offline",
)


class ResidualLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, 2, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        return self.head(self.lstm(x)[0])[..., 0]


class RecurrentSequence(nn.Module):
    def __init__(self, input_size, hidden_size, kind="lstm", bidirectional=False):
        super().__init__()
        recurrent = nn.LSTM if kind == "lstm" else nn.GRU
        self.core = recurrent(
            input_size,
            hidden_size,
            2,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.head = nn.Linear(hidden_size * (2 if bidirectional else 1), 1)

    def forward(self, x):
        return self.head(self.core(x)[0])[..., 0]


class SequenceTransformer(nn.Module):
    def __init__(self, input_size, length, hidden_size=64, causal=False):
        super().__init__()
        self.causal = causal
        self.input = nn.Linear(input_size, hidden_size)
        self.position = nn.Parameter(torch.zeros(1, length, hidden_size))
        layer = nn.TransformerEncoderLayer(
            hidden_size,
            4,
            2 * hidden_size,
            .1,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, 3, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1))
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(length, length, dtype=torch.bool), 1),
            persistent=False,
        )

    def forward(self, x):
        length = x.shape[1]
        value = self.input(x) + self.position[:, :length]
        value = self.encoder(
            value,
            mask=self.mask[:length, :length] if self.causal else None,
            is_causal=self.causal,
        )
        return self.head(value)[..., 0]


class _ConvBlock(nn.Sequential):
    def __init__(self, input_size, output_size):
        super().__init__(
            nn.Conv1d(input_size, output_size, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(output_size, output_size, 3, padding=1),
            nn.GELU(),
        )


class SequenceUNet(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super().__init__()
        self.enc1 = _ConvBlock(input_size, hidden_size)
        self.enc2 = _ConvBlock(hidden_size, 2 * hidden_size)
        self.pool = nn.MaxPool1d(2)
        self.center = _ConvBlock(2 * hidden_size, 4 * hidden_size)
        self.up2 = nn.ConvTranspose1d(4 * hidden_size, 2 * hidden_size, 2, stride=2)
        self.dec2 = _ConvBlock(4 * hidden_size, 2 * hidden_size)
        self.up1 = nn.ConvTranspose1d(2 * hidden_size, hidden_size, 2, stride=2)
        self.dec1 = _ConvBlock(2 * hidden_size, hidden_size)
        self.head = nn.Conv1d(hidden_size, 1, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        first = self.enc1(x)
        second = self.enc2(self.pool(first))
        value = self.dec2(torch.cat((self.up2(self.center(self.pool(second))), second), 1))
        value = self.dec1(torch.cat((self.up1(value), first), 1))
        return self.head(value)[:, 0]


def build_model_free(name, input_size, length, hidden_size=64):
    if name == "lstm_online":
        return RecurrentSequence(input_size, hidden_size, "lstm")
    if name == "gru_online":
        return RecurrentSequence(input_size, hidden_size, "gru")
    if name == "bilstm_offline":
        return RecurrentSequence(input_size, hidden_size, "lstm", True)
    if name == "unet_offline":
        return SequenceUNet(input_size, max(16, hidden_size // 2))
    if name in ("transformer_online", "transformer_offline"):
        return SequenceTransformer(input_size, length, hidden_size, name == "transformer_online")
    raise ValueError(f"unknown model: {name}")


def ou_features(state, az_g, state_space):
    predicted = np.zeros_like(state)
    predicted[:, 1:] = state[:, :-1] @ state_space.A.T
    innovation = (np.asarray(az_g) - 1) * 9.81 - (predicted @ state_space.H.T)[..., 0]
    return np.concatenate((state, innovation[..., None]), -1).astype(np.float32)
    predicted = np.zeros_like(state)
    predicted[:, 1:] = state[:, :-1] @ state_space.A.T
    innovation = (np.asarray(az_g) - 1) * 9.81 - (predicted @ state_space.H.T)[..., 0]
    return np.concatenate((state, innovation[..., None]), -1).astype(np.float32)


def normalize_residual(features, residual, train):
    x_mean = features[train].mean((0, 1))
    x_std = features[train].std((0, 1)) + 1e-6
    y_mean = residual[train].mean()
    y_std = residual[train].std() + 1e-6
    return (
        (features - x_mean) / x_std,
        (residual - y_mean) / y_std,
        x_mean,
        x_std,
        y_mean,
        y_std,
    )


def normalize_model_free(features, target, train):
    x_mean = features[train].mean((0, 1))
    x_std = features[train].std((0, 1)) + 1e-6
    y_mean = target[train].mean()
    y_std = target[train].std() + 1e-6
    return (
        ((features - x_mean) / x_std).astype(np.float32),
        ((target - y_mean) / y_std).astype(np.float32),
        x_mean,
        x_std,
        y_mean,
        y_std,
    )


def _validation_loss(model, loader, device, amp):
    model.eval()
    total = count = 0
    with torch.no_grad():
        for features, target in loader:
            context = torch.amp.autocast("cuda") if amp else nullcontext()
            with context:
                value = nn.functional.mse_loss(
                    model(features.to(device)),
                    target.to(device),
                    reduction="sum",
                )
            total += value.item()
            count += target.numel()
    return total / count


def _fit_sequence(
    model,
    features,
    target,
    train,
    validation,
    device,
    epochs,
    batch_size,
    model_free,
    verbose,
):
    model.to(device)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(features[train]), torch.from_numpy(target[train])),
        batch_size=batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        TensorDataset(torch.from_numpy(features[validation]), torch.from_numpy(target[validation])),
        batch_size=batch_size,
    )
    optimizer = (
        torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        if model_free
        else torch.optim.Adam(model.parameters(), lr=1e-3)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.2, patience=3)
    amp = model_free and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best, best_state, stale, best_epoch = np.inf, None, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_features, batch_target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            context = torch.amp.autocast("cuda") if amp else nullcontext()
            with context:
                loss = nn.functional.mse_loss(
                    model(batch_features.to(device)),
                    batch_target.to(device),
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            scaler.step(optimizer)
            scaler.update()
        value = _validation_loss(model, validation_loader, device, amp)
        scheduler.step(value)
        if verbose:
            print(f"epoch={epoch:02d} val_mse={value:.6f}")
        if value < best - 1e-5:
            best, best_state, stale, best_epoch = value, deepcopy(model.state_dict()), 0, epoch
        else:
            stale += 1
            if stale >= 8:
                break
    if best_state is None:
        raise ValueError("epochs must be positive")
    model.load_state_dict(best_state)
    return best, best_epoch


def fit_residual_lstm(
    model,
    features,
    residual,
    train,
    validation,
    device,
    epochs=30,
    batch_size=32,
    verbose=False,
):
    return _fit_sequence(
        model,
        features,
        residual,
        train,
        validation,
        device,
        epochs,
        batch_size,
        False,
        verbose,
    )


def fit_model_free(
    model,
    features,
    target,
    train,
    validation,
    device,
    epochs=30,
    batch_size=32,
    verbose=False,
):
    return _fit_sequence(
        model,
        features,
        target,
        train,
        validation,
        device,
        epochs,
        batch_size,
        True,
        verbose,
    )


def _predict_sequence(model, features, device, batch_size, amp):
    loader = DataLoader(torch.from_numpy(features), batch_size=batch_size)
    output = []
    amp = amp and device.type == "cuda"
    model.to(device).eval()
    with torch.no_grad():
        for batch in loader:
            context = torch.amp.autocast("cuda") if amp else nullcontext()
            with context:
                output.append(model(batch.to(device)).float().cpu().numpy())
    return np.concatenate(output)


def predict_residual_lstm(model, features, device, batch_size=32):
    return _predict_sequence(model, features, device, batch_size, False)


def predict_model_free(model, features, device, batch_size=32):
    return _predict_sequence(model, features, device, batch_size, True)


def causality_error(model, input_size, length=40):
    model.eval()
    device = next(model.parameters()).device
    original = torch.randn(2, length, input_size, device=device)
    changed = original.clone()
    changed[:, length // 2 + 1:] += torch.randn_like(changed[:, length // 2 + 1:])
    with torch.no_grad():
        left = model(original)[:, :length // 2 + 1]
        right = model(changed)[:, :length // 2 + 1]
    return float(torch.max(torch.abs(left - right)))
