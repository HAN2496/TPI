import torch
import torch.nn as nn
import torch.nn.functional as F


class MoECNNModel(nn.Module):
    def __init__(self, obs_dim, n_train_users, user_dim=16, hidden_dim=64, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        
        # Train user embedding
        self.user_emb = nn.Embedding(n_train_users, user_dim)
        
        # Shared Trajectory Encoder (1D CNN)
        self.obs_proj = nn.Linear(obs_dim, hidden_dim)
        self.encoder = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )
        
        # Experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(num_experts)
        ])
        
        # Gate
        self.gate = nn.Sequential(
            nn.Linear(user_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, num_experts)
        )

    def forward(self, obs, uids=None, custom_user_emb=None):
        B, T, D = obs.shape
        
        # Encoder
        h = F.leaky_relu(self.obs_proj(obs))
        h = self.encoder(h.permute(0, 2, 1))
        h = F.max_pool1d(h, kernel_size=T).squeeze(-1)  # [B, hidden_dim]
        
        # Experts output
        expert_outputs = torch.cat([expert(h) for expert in self.experts], dim=1) # [B, num_experts]
        
        # Gating
        if custom_user_emb is not None:
            u_e = custom_user_emb
        else:
            u_e = self.user_emb(uids)
            
        gate_weights = F.softmax(self.gate(u_e), dim=-1) # [B, num_experts]
        
        # Weighted sum
        logits = (expert_outputs * gate_weights).sum(dim=1)
        return logits