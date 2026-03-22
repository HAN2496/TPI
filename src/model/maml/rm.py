import torch.nn as nn
import torch.nn.functional as F

from ..base import NeuralModel

class MetaCNNModel(NeuralModel):
    def __init__(self, obs_dim, hidden_dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(obs_dim, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = F.max_pool1d(x, kernel_size=x.shape[-1]).squeeze(-1)
        return self.head(x).squeeze(-1)