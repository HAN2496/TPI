import torch
import torch.nn as nn
import torch.nn.functional as F

class DriverRewardModel(nn.Module):
    def __init__(self, input_dim, user_emb_dim, hidden_dim=64):
        super(DriverRewardModel, self).__init__()
        
        # 1D CNN for time-series feature extraction
        # Input shape: (B, Input_Dim, Time)
        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1) # Global pooling to get (B, 64)
        
        self.feature_mlp = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # User Embedding projection
        self.user_proj = nn.Linear(user_emb_dim, hidden_dim)
        
        # Fusion and Prediction
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Scalar reward
        )
        
    def forward(self, x, user_emb):
        """
        x: (B, T, D) Driver data sequence
        user_emb: (B, User_Emb_Dim) User embedding vector
        """
        # Permute to (B, D, T) for Conv1d
        x = x.permute(0, 2, 1)
        
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = self.pool(h).squeeze(-1) # (B, 64)
        
        h_x = self.feature_mlp(h)
        
        # Process User Embedding
        h_u = self.user_proj(user_emb)
        
        # Concatenate
        combined = torch.cat([h_x, h_u], dim=1)
        
        reward = self.head(combined)
        return reward
