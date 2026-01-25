import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Simple MLP Encoder for Trajectories (Flattened)"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2)
        )
    def forward(self, x):
        return self.net(x)

class RM(nn.Module):
    """
    Mixture-of-Experts Reward Model for Control.
    Replaces the LLM + LoRA from the paper with MLP + Experts.
    """
    def __init__(self, input_dim, hidden_dim, user_emb_dim, num_experts=4, topk=1):
        super().__init__()
        self.num_experts = num_experts
        self.topk = topk
        
        # 1. Shared Feature Extractor (Encoder)
        self.encoder = Encoder(input_dim, hidden_dim)
        
        # 2. Gating Network (User Embedding -> Expert Weights)
        # Paper Eq (10): Gating based on User Embedding
        self.gating_net = nn.Sequential(
            nn.Linear(user_emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_experts)
        )
        
        # 3. Experts (Heads)
        # Each expert takes the hidden representation and outputs a scalar reward
        self.experts = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_experts)
        ])
        
        # Shared expert (Optional, similar to MoLE's shared path)
        self.shared_expert = nn.Linear(hidden_dim, 1)

    def forward(self, x, user_emb):
        """
        x: Trajectory observations (Batch, Flattened_Dim)
        user_emb: User embeddings from GCF (Batch, User_Dim)
        """
        # Feature Extraction
        features = self.encoder(x) # (B, H)
        
        # Gating
        gate_logits = self.gating_net(user_emb) # (B, K)
        gate_weights = F.softmax(gate_logits, dim=-1)
        
        # Top-K Gating (Hard routing like Paper)
        if self.topk < self.num_experts:
            topk_vals, topk_indices = torch.topk(gate_weights, self.topk, dim=-1)
            mask = torch.zeros_like(gate_weights).scatter_(1, topk_indices, 1.0)
            gate_weights = gate_weights * mask
            gate_weights = gate_weights / (gate_weights.sum(dim=-1, keepdim=True) + 1e-9)

        # Expert Execution
        expert_outputs = torch.stack([exp(features) for exp in self.experts], dim=1) # (B, K, 1)
        
        # Weighted Sum
        weighted_expert_output = (expert_outputs * gate_weights.unsqueeze(-1)).sum(dim=1) # (B, 1)
        
        # Shared Path
        shared_output = self.shared_expert(features)
        
        # Final Reward
        total_reward = weighted_expert_output + shared_output
        
        return total_reward, gate_weights