import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardModel(nn.Module):
    def __init__(self, obs_dim, user_dim, hidden=128, mlp_hidden=128):
        super().__init__()
        self.obs_proj = nn.Linear(obs_dim, hidden)
        self.user_proj = nn.Linear(user_dim, hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, mlp_hidden), nn.ReLU(),
            nn.Linear(mlp_hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, user_emb, obs):
        h_obs = self.obs_proj(obs)
        h_u = self.user_proj(user_emb).unsqueeze(1)
        h = nn.LeakyReLU()(h_obs + h_u).mean(dim=1)
        return self.head(self.mlp(h)).squeeze(-1)


class CNNRewardModel(nn.Module):
    def __init__(self, obs_dim, user_dim, hidden=128, mlp_hidden=128, kernel_size=3, layers=2):
        super().__init__()
        self.obs_proj = nn.Linear(obs_dim, hidden)
        self.user_proj = nn.Linear(user_dim, hidden)

        conv_layers = []
        for _ in range(layers):
            conv_layers.append(nn.Conv1d(hidden, hidden, kernel_size=kernel_size, padding=kernel_size // 2))
            conv_layers.append(nn.LeakyReLU())
        self.conv = nn.Sequential(*conv_layers)

        self.head = nn.Sequential(
            nn.Linear(hidden, mlp_hidden), nn.LeakyReLU(),
            nn.Linear(mlp_hidden, 1)
        )

    def forward(self, user_emb, obs):
        B, T, D = obs.shape
        h = F.leaky_relu(self.obs_proj(obs) + self.user_proj(user_emb).unsqueeze(1))
        h = self.conv(h.permute(0, 2, 1))
        h = F.max_pool1d(h, kernel_size=T).squeeze(2)
        return self.head(h).squeeze(-1)


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
        return self.uids[idx], self.item_series[self.iids[idx]], self.labels[idx]


def rm_collate(batch):
    u = torch.tensor([b[0] for b in batch], dtype=torch.long)
    obs = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.float32)
    y = torch.tensor([b[2] for b in batch], dtype=torch.float32)
    return u, obs, y


class MoLELinear(nn.Module):
    def __init__(self, in_features, out_features, num_experts=8, rank=8):
        super().__init__()
        self.base_layer = nn.Linear(in_features, out_features)
        self.num_experts = num_experts
        self.lora_A_s = nn.Parameter(torch.randn(in_features, rank) / np.sqrt(in_features))
        self.lora_B_s = nn.Parameter(torch.zeros(rank, out_features))
        self.lora_A_e = nn.Parameter(torch.randn(num_experts, in_features, rank) / np.sqrt(in_features))
        self.lora_B_e = nn.Parameter(torch.zeros(num_experts, rank, out_features))

    def forward(self, x, routing_weights):
        base_out = self.base_layer(x)
        shared_out = (x @ self.lora_A_s) @ self.lora_B_s
        expert_out = 0
        reshape_dim = [-1] + [1] * (x.dim() - 1)
        for i in range(self.num_experts):
            w_i = routing_weights[:, i].view(reshape_dim)
            expert_out += w_i * ((x @ self.lora_A_e[i]) @ self.lora_B_e[i])
        return base_out + shared_out + expert_out


class MoLECNNRewardModel(nn.Module):
    def __init__(self, obs_dim, user_dim, hidden=128, mlp_hidden=128,
                 kernel_size=3, layers=2, num_experts=4, rank=6, tau=2.0):
        super().__init__()
        self.tau = tau
        self.gating = nn.Sequential(
            nn.Linear(user_dim, 128), nn.ReLU(),
            nn.Linear(128, num_experts)
        )
        self.obs_proj = MoLELinear(obs_dim, hidden, num_experts, rank)

        conv_layers = []
        for _ in range(layers):
            conv_layers.append(nn.Conv1d(hidden, hidden, kernel_size=kernel_size, padding=kernel_size // 2))
            conv_layers.append(nn.LeakyReLU())
        self.conv = nn.Sequential(*conv_layers)

        self.head_1 = MoLELinear(hidden, mlp_hidden, num_experts, rank)
        self.head_act = nn.LeakyReLU()
        self.head_2 = MoLELinear(mlp_hidden, 1, num_experts, rank)

    def forward(self, user_emb, obs):
        B, T, D = obs.shape
        soft_w = F.softmax(self.gating(user_emb) / self.tau, dim=-1)
        idx = soft_w.argmax(dim=-1, keepdim=True)
        hard_w = torch.zeros_like(soft_w).scatter_(-1, idx, 1.0)
        routing_weights = (hard_w - soft_w.detach() + soft_w) if self.training else hard_w

        x = F.leaky_relu(self.obs_proj(obs, routing_weights))
        x = F.max_pool1d(self.conv(x.permute(0, 2, 1)), kernel_size=T).squeeze(2)
        return self.head_2(self.head_act(self.head_1(x, routing_weights)), routing_weights).squeeze(-1)
