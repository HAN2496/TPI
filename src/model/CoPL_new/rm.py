import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# Reward Model (time-series, Linear encoder)
# =========================
class RewardModel(nn.Module):
    """
    time-series RM:
      - per-timestep linear projection
      - user conditioning via linear projection
      - nonlinearity + mean pooling
      - head -> logit
    """
    def __init__(self, obs_dim: int, user_dim: int, hidden: int = 128, mlp_hidden: int = 128):
        super().__init__()
        self.obs_proj = nn.Linear(obs_dim, hidden)
        self.user_proj = nn.Linear(user_dim, hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, user_emb: torch.Tensor, obs: torch.Tensor):
        """
        user_emb: (B, user_dim)
        obs:      (B, T, obs_dim)
        """
        # project
        h_obs = self.obs_proj(obs)  # (B,T,H)
        h_u = self.user_proj(user_emb).unsqueeze(1)  # (B,1,H)
        h = torch.nn.LeakyReLU()(h_obs + h_u)  # (B,T,H)
        # h = torch.tanh(h_obs + h_u)  # (B,T,H)
        h = h.mean(dim=1)            # (B,H)
        h = self.mlp(h)              # (B,H)
        return self.head(h).squeeze(-1)  # (B,)


class CNNRewardModel(nn.Module):
    """
    CNN-based RM:
      - Temporal CNN (Non-Markovian patterns)
      - Global Max Pooling (Once-per-episode feedback)
      - Emphasizes critical moments in the sequence
    """
    def __init__(self, obs_dim: int, user_dim: int, hidden: int = 128, mlp_hidden: int = 128, kernel_size: int = 3, layers: int = 2):
        super().__init__()
        # Projections
        self.obs_proj = nn.Linear(obs_dim, hidden)
        self.user_proj = nn.Linear(user_dim, hidden)

        # Temporal CNN
        conv_layers = []
        for _ in range(layers):
            conv_layers.append(nn.Conv1d(hidden, hidden, kernel_size=kernel_size, padding=kernel_size//2))
            conv_layers.append(nn.LeakyReLU())
        self.conv = nn.Sequential(*conv_layers)
        
        # Head (Once-per-episode)
        self.head = nn.Sequential(
            nn.Linear(hidden, mlp_hidden),
            nn.LeakyReLU(),
            nn.Linear(mlp_hidden, 1)
        )

    def forward(self, user_emb: torch.Tensor, obs: torch.Tensor):
        """
        user_emb: (B, user_dim)
        obs:      (B, T, obs_dim)
        """
        B, T, D = obs.shape
        
        # 1. Project & Condition
        h_obs = self.obs_proj(obs)                           # (B, T, H)
        h_u = self.user_proj(user_emb).unsqueeze(1)          # (B, 1, H)
        x = h_obs + h_u                                      # (B, T, H)
        x = F.leaky_relu(x)
        
        # 2. Permute for Conv1d: (B, H, T)
        x = x.permute(0, 2, 1)                               # (B, H, T)
        
        # 3. Temporal Processing
        x = self.conv(x)                                     # (B, H, T)
        
        # 4. Global Max Pooling (Critical Moment extraction)
        # This aligns with "Once-per-episode feedback" where the outcome 
        # depends on the single most significant event in the trajectory.
        x = F.max_pool1d(x, kernel_size=T).squeeze(2)        # (B, H)

        return self.head(x).squeeze(-1)                      # (B,)


def weighted_bce_logits(logits, labels, pos_weight=None):
    labels = labels.float()
    if pos_weight is not None:
        return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
    return F.binary_cross_entropy_with_logits(logits, labels)

class RMEdgeDataset(torch.utils.data.Dataset):
    def __init__(self, uids, iids, labels, item_series):
        self.uids = uids.astype(np.int64)
        self.iids = iids.astype(np.int64)
        self.labels = labels.astype(np.int64)
        self.item_series = item_series.astype(np.float32)

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        u = self.uids[idx]
        i = self.iids[idx]
        y = self.labels[idx]
        obs = self.item_series[i]  # (T,d)
        return u, obs, y

def rm_collate(batch):
    u = torch.tensor([b[0] for b in batch], dtype=torch.long)
    obs = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.float32)
    y = torch.tensor([b[2] for b in batch], dtype=torch.float32)
    return u, obs, y
