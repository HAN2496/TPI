from copy import deepcopy
from contextlib import nullcontext

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


MODELS = ("lstm_online", "gru_online", "transformer_online",
          "bilstm_offline", "unet_offline", "transformer_offline")


class Recurrent(nn.Module):
    def __init__(self, input_size, hidden_size, kind="lstm", bidirectional=False):
        super().__init__()
        recurrent = nn.LSTM if kind == "lstm" else nn.GRU
        self.core = recurrent(input_size, hidden_size, 2, batch_first=True, bidirectional=bidirectional)
        self.head = nn.Linear(hidden_size * (2 if bidirectional else 1), 1)

    def forward(self, x):
        return self.head(self.core(x)[0])[..., 0]


class Transformer(nn.Module):
    def __init__(self, input_size, length, hidden_size=64, causal=False):
        super().__init__()
        self.causal = causal
        self.input = nn.Linear(input_size, hidden_size)
        self.position = nn.Parameter(torch.zeros(1, length, hidden_size))
        layer = nn.TransformerEncoderLayer(hidden_size, 4, 2 * hidden_size, .1,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, 3, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1))
        self.register_buffer("mask", torch.triu(torch.ones(length, length, dtype=torch.bool), 1), persistent=False)

    def forward(self, x):
        length = x.shape[1]
        value = self.input(x) + self.position[:, :length]
        value = self.encoder(value, mask=self.mask[:length, :length] if self.causal else None,
                             is_causal=self.causal)
        return self.head(value)[..., 0]


class ConvBlock(nn.Sequential):
    def __init__(self, input_size, output_size):
        super().__init__(nn.Conv1d(input_size, output_size, 3, padding=1), nn.GELU(),
                         nn.Conv1d(output_size, output_size, 3, padding=1), nn.GELU())


class UNet(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super().__init__()
        self.enc1, self.enc2 = ConvBlock(input_size, hidden_size), ConvBlock(hidden_size, 2 * hidden_size)
        self.pool = nn.MaxPool1d(2)
        self.center = ConvBlock(2 * hidden_size, 4 * hidden_size)
        self.up2 = nn.ConvTranspose1d(4 * hidden_size, 2 * hidden_size, 2, stride=2)
        self.dec2 = ConvBlock(4 * hidden_size, 2 * hidden_size)
        self.up1 = nn.ConvTranspose1d(2 * hidden_size, hidden_size, 2, stride=2)
        self.dec1 = ConvBlock(2 * hidden_size, hidden_size)
        self.head = nn.Conv1d(hidden_size, 1, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        value = self.dec2(torch.cat((self.up2(self.center(self.pool(e2))), e2), 1))
        value = self.dec1(torch.cat((self.up1(value), e1), 1))
        return self.head(value)[:, 0]


def build(name, input_size, length, hidden_size=64):
    if name == "lstm_online":
        return Recurrent(input_size, hidden_size, "lstm")
    if name == "gru_online":
        return Recurrent(input_size, hidden_size, "gru")
    if name == "bilstm_offline":
        return Recurrent(input_size, hidden_size, "lstm", True)
    if name == "unet_offline":
        return UNet(input_size, max(16, hidden_size // 2))
    if name in ("transformer_online", "transformer_offline"):
        return Transformer(input_size, length, hidden_size, name == "transformer_online")
    raise ValueError(name)


def normalize(x, y, train):
    x_mean, x_std = x[train].mean((0, 1)), x[train].std((0, 1)) + 1e-6
    y_mean, y_std = y[train].mean(), y[train].std() + 1e-6
    return ((x - x_mean) / x_std).astype(np.float32), ((y - y_mean) / y_std).astype(np.float32), \
        x_mean, x_std, y_mean, y_std


def _loss(model, loader, device, amp):
    model.eval()
    total = count = 0
    with torch.no_grad():
        for x, y in loader:
            context = torch.amp.autocast("cuda") if amp else nullcontext()
            with context:
                value = nn.functional.mse_loss(model(x.to(device)), y.to(device), reduction="sum")
            total += value.item()
            count += y.numel()
    return total / count


def fit(model, x, y, train, validation, device, epochs=30, batch_size=32):
    train_loader = DataLoader(TensorDataset(torch.from_numpy(x[train]), torch.from_numpy(y[train])),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(x[validation]), torch.from_numpy(y[validation])),
                            batch_size=batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.2, patience=3)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best, best_state, stale, best_epoch = np.inf, None, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            context = torch.amp.autocast("cuda") if amp else nullcontext()
            with context:
                loss = nn.functional.mse_loss(model(xb.to(device)), yb.to(device))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            scaler.step(optimizer)
            scaler.update()
        value = _loss(model, val_loader, device, amp)
        scheduler.step(value)
        print(f"epoch={epoch:02d} val_mse={value:.6f}")
        if value < best - 1e-5:
            best, best_state, stale, best_epoch = value, deepcopy(model.state_dict()), 0, epoch
        else:
            stale += 1
            if stale >= 8:
                break
    model.load_state_dict(best_state)
    return best, best_epoch


def predict(model, x, device, batch_size=32):
    loader = DataLoader(torch.from_numpy(x), batch_size=batch_size)
    output, amp = [], device.type == "cuda"
    model.eval()
    with torch.no_grad():
        for batch in loader:
            context = torch.amp.autocast("cuda") if amp else nullcontext()
            with context:
                output.append(model(batch.to(device)).float().cpu().numpy())
    return np.concatenate(output)


def causal_check(model, input_size, length=40):
    model.eval()
    x = torch.randn(2, length, input_size)
    changed = x.clone()
    changed[:, length // 2 + 1:] += torch.randn_like(changed[:, length // 2 + 1:])
    with torch.no_grad():
        left = model(x)[:, :length // 2 + 1]
        right = model(changed)[:, :length // 2 + 1]
    return float(torch.max(torch.abs(left - right)))
