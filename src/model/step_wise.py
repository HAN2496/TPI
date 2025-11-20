import torch
import torch.nn as nn

from src.model.base import BaseModel

class OnlineMLP(BaseModel):
    def __init__(self, input_dim, hidden_dims, dropout_rate, act_name, use_batchnorm=False):
        super().__init__()

        self.input_dim = input_dim

        if act_name == "relu":
            act = nn.ReLU()
        elif act_name == "tanh":
            act = nn.Tanh()
        else:
            act = nn.GELU()

        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))

            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(act)

            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        batch_size, seq_len, feat_dim = x.shape

        x_flat = x.reshape(-1, feat_dim)
        reward_flat = self.network(x_flat).squeeze(-1)
        rewards = reward_flat.reshape(batch_size, seq_len)
        total_reward = rewards.sum(dim=1)

        return total_reward

    def get_step_rewards(self, x):
        batch_size, seq_len, feat_dim = x.shape

        x_flat = x.reshape(-1, feat_dim)
        reward_flat = self.network(x_flat).squeeze(-1)
        rewards = reward_flat.reshape(batch_size, seq_len)

        return rewards

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            total_reward = self.forward(x)
            probs = torch.sigmoid(total_reward)
        return (probs > 0.5).long()