import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils.vpl_dataset import create_vpl_dataset, VPLDataset
from src.model.vpl.vae import VAEModel
from src.model.vpl.trainer import VPLTrainer, Annealer

driver_names = ['강신길']
time_range = [5, 7]
downsample = 5
features = ['IMU_VerAccelVal', 'Bounce_rate_6D', 'Pitch_rate_6D', 'IMU_LongAccelVal']
set_len = 10

print("Creating VPL dataset...")
dataset, driver_map = create_vpl_dataset(
    driver_names=driver_names,
    time_range=time_range,
    downsample=downsample,
    features=features,
    set_len=set_len
)

n_total = len(dataset['observations'])
n_val = max(1, int(n_total * 0.1))
n_train = n_total - n_val

indices = np.random.permutation(n_total)
train_idx, val_idx = indices[:n_train], indices[n_train:]

train_dataset = VPLDataset({
    k: v[train_idx] for k, v in dataset.items()
})
val_dataset = VPLDataset({
    k: v[val_idx] for k, v in dataset.items()
})

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

_, set_len, T, d = dataset['observations'].shape
encoder_input = set_len * (T * d + 1)
decoder_input = d + 32
latent_dim = 32
hidden_dim = 64

print(f"\nModel dimensions:")
print(f"  Encoder input: {encoder_input}")
print(f"  Decoder input: {decoder_input}")
print(f"  Latent dim: {latent_dim}")
print(f"  Hidden dim: {hidden_dim}")

annealer = Annealer(
    total_steps=50 // 4,
    shape='cosine',
    baseline=0.0,
    cyclical=True
)

model = VAEModel(
    encoder_input=encoder_input,
    decoder_input=decoder_input,
    latent_dim=latent_dim,
    hidden_dim=hidden_dim,
    annotation_size=set_len,
    size_segment=T,
    kl_weight=1.0,
    annealer=annealer,
    reward_scaling=T
)

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters())}")

config = {
    'learning_rate': 1e-3,
    'weight_decay': 0.0,
    'epochs': 50,
    'early_stop': False
}

trainer = VPLTrainer(model, config, device='cpu')

print("\nTraining for 50 epochs...")
history, val_accuracy = trainer.train(
    train_loader, val_loader,
    epochs=50,
    verbose=1
)

print(f"\nTraining completed!")
print(f"Final Val Accuracy: {val_accuracy:.4f}")
print(f"Final Val Loss: {history['val_loss'][-1]:.4f}")
