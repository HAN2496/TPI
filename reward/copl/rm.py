import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardModel(nn.Module):
    uses_user_embedding = True

    def __init__(self, obs_dim, user_dim, hidden=128, mlp_hidden=128, dropout=0.0):
        super().__init__()
        self.obs_proj = nn.Linear(obs_dim, hidden)
        self.user_proj = nn.Linear(user_dim, hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, mlp_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, user_emb, obs):
        h_obs = self.obs_proj(obs)
        h_u = self.user_proj(user_emb).unsqueeze(1)
        h = nn.LeakyReLU()(h_obs + h_u).mean(dim=1)
        return self.head(self.mlp(h)).squeeze(-1)

    def forward_no_user(self, obs):
        h = nn.LeakyReLU()(self.obs_proj(obs)).mean(dim=1)
        return self.head(self.mlp(h)).squeeze(-1)


class ObsOnlyRewardModel(nn.Module):
    uses_user_embedding = False

    def __init__(self, obs_dim, hidden=128, mlp_hidden=128):
        super().__init__()
        self.obs_proj = nn.Linear(obs_dim, hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, mlp_hidden), nn.ReLU(),
            nn.Linear(mlp_hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, obs):
        h = nn.LeakyReLU()(self.obs_proj(obs)).mean(dim=1)
        return self.head(self.mlp(h)).squeeze(-1)


class CNNRewardModel(nn.Module):
    uses_user_embedding = True

    def __init__(self, obs_dim, user_dim, hidden=128, mlp_hidden=128, kernel_size=3, layers=2, dropout=0.0):
        super().__init__()
        self.obs_proj = nn.Linear(obs_dim, hidden)
        self.user_proj = nn.Linear(user_dim, hidden)

        conv_layers = []
        for _ in range(layers):
            conv_layers.append(nn.Conv1d(hidden, hidden, kernel_size=kernel_size, padding=kernel_size // 2))
            conv_layers.append(nn.LeakyReLU())
            conv_layers.append(nn.Dropout(dropout))
        self.conv = nn.Sequential(*conv_layers)

        self.head = nn.Sequential(
            nn.Linear(hidden, mlp_hidden), nn.LeakyReLU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1)
        )

    def forward(self, user_emb, obs):
        B, T, D = obs.shape
        h = F.leaky_relu(self.obs_proj(obs) + self.user_proj(user_emb).unsqueeze(1))
        h = self.conv(h.permute(0, 2, 1))
        h = F.max_pool1d(h, kernel_size=T).squeeze(2)
        return self.head(h).squeeze(-1)

    def forward_no_user(self, obs):
        B, T, D = obs.shape
        h = F.leaky_relu(self.obs_proj(obs))
        h = self.conv(h.permute(0, 2, 1))
        h = F.max_pool1d(h, kernel_size=T).squeeze(2)
        return self.head(h).squeeze(-1)


class ObsOnlyCNNRewardModel(nn.Module):
    uses_user_embedding = False

    def __init__(self, obs_dim, hidden=128, mlp_hidden=128, kernel_size=3, layers=2):
        super().__init__()
        self.obs_proj = nn.Linear(obs_dim, hidden)

        conv_layers = []
        for _ in range(layers):
            conv_layers.append(nn.Conv1d(hidden, hidden, kernel_size=kernel_size, padding=kernel_size // 2))
            conv_layers.append(nn.LeakyReLU())
        self.conv = nn.Sequential(*conv_layers)

        self.head = nn.Sequential(
            nn.Linear(hidden, mlp_hidden), nn.LeakyReLU(),
            nn.Linear(mlp_hidden, 1)
        )

    def forward(self, obs):
        B, T, D = obs.shape
        h = F.leaky_relu(self.obs_proj(obs))
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
    uses_user_embedding = True

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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class PreferenceTransformerRewardModel(nn.Module):
    uses_user_embedding = True

    def __init__(self, obs_dim, user_dim, hidden=128, num_heads=4, num_layers=2, max_len=1000):
        super().__init__()
        self.obs_proj = nn.Linear(obs_dim, hidden)
        self.user_proj = nn.Linear(user_dim, hidden)
        self.pos_encoder = PositionalEncoding(hidden, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=num_heads, dim_feedforward=hidden * 2,
            dropout=0.2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Preference Transformer 특유의 구조: 궤적 내 각 step의 보상(r_t)과 가중치(w_t)를 분리
        self.reward_head = nn.Linear(hidden, 1)
        self.attn_head = nn.Linear(hidden, 1)

    def forward(self, user_emb, obs):
        # h: (B, T, hidden)
        h = F.leaky_relu(self.obs_proj(obs) + self.user_proj(user_emb).unsqueeze(1))
        h = self.pos_encoder(h)
        
        # Transformer를 통해 과거 state들의 문맥(Context)이 반영된 sequence 추출
        h = self.transformer(h) 
        
        r_t = self.reward_head(h).squeeze(-1) # (B, T)
        w_t = F.softmax(self.attn_head(h).squeeze(-1), dim=1) # (B, T)
        
        # 가중치가 반영된 궤적 전체의 스칼라 보상 (Weighted sum)
        return torch.sum(w_t * r_t, dim=1)

    def forward_no_user(self, obs):
        h = F.leaky_relu(self.obs_proj(obs))
        h = self.pos_encoder(h)
        h = self.transformer(h)
        r_t = self.reward_head(h).squeeze(-1)
        w_t = F.softmax(self.attn_head(h).squeeze(-1), dim=1)
        return torch.sum(w_t * r_t, dim=1)


class ObsOnlyPreferenceTransformerRewardModel(nn.Module):
    uses_user_embedding = False

    def __init__(self, obs_dim, hidden=128, num_heads=4, num_layers=2, max_len=1000):
        super().__init__()
        self.obs_proj = nn.Linear(obs_dim, hidden)
        self.pos_encoder = PositionalEncoding(hidden, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=num_heads, dim_feedforward=hidden * 2,
            dropout=0.2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.reward_head = nn.Linear(hidden, 1)
        self.attn_head = nn.Linear(hidden, 1)

    def forward(self, obs):
        h = F.leaky_relu(self.obs_proj(obs))
        h = self.pos_encoder(h)
        h = self.transformer(h)
        r_t = self.reward_head(h).squeeze(-1)
        w_t = F.softmax(self.attn_head(h).squeeze(-1), dim=1)
        return torch.sum(w_t * r_t, dim=1)
