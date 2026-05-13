import torch
import torch.nn as nn
import torch.nn.functional as F

class RAPRewardModel(nn.Module):
    def __init__(self, obs_dim, n_train_users, user_dim=16, hidden_dim=64, attn_dim=64):
        super().__init__()
        
        # 유저 임베딩
        self.user_emb = nn.Embedding(n_train_users, user_dim)
        
        # 궤적 인코더 (Dropout 추가로 과적합 방지)
        self.obs_proj = nn.Linear(obs_dim, hidden_dim)
        self.encoder = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )
        
        # Attention Projections
        # Query: [Target 궤적, Target 유저]
        self.wq = nn.Linear(hidden_dim + user_dim, attn_dim)
        # Key: [Memory 궤적, Memory 유저]
        self.wk = nn.Linear(hidden_dim + user_dim, attn_dim)
        # Value: [Memory 궤적, Memory 유저, Memory 라벨(0 or 1)]
        self.wv = nn.Linear(hidden_dim + user_dim + 1, attn_dim)
        
        # 최종 예측기 (Target 정보 + 검색된 Context 결합)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim + user_dim + attn_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )

    def encode_obs(self, obs):
        B, T, D = obs.shape
        h = F.leaky_relu(self.obs_proj(obs))
        h = self.encoder(h.permute(0, 2, 1))
        h = F.max_pool1d(h, kernel_size=T).squeeze(-1)
        return h

    def forward(self, obs, uids, mem_obs, mem_uids, mem_y, custom_user_emb=None, custom_mem_u_emb=None):
        """
        obs, uids: 평가할 타겟 데이터
        mem_obs, mem_uids, mem_y: 컨닝할 과거 메모리 데이터 (Memory Bank)
        """
        # 1. Target Encoding
        z_q = self.encode_obs(obs)
        e_q = custom_user_emb if custom_user_emb is not None else self.user_emb(uids)
        
        # 2. Memory Encoding
        z_m = self.encode_obs(mem_obs)
        if custom_mem_u_emb is not None:
            e_m = custom_mem_u_emb
        else:
            e_m = self.user_emb(mem_uids)
            
        # 3. Cross-Attention (Retrieval)
        Q = self.wq(torch.cat([z_q, e_q], dim=-1)) # [B, attn_dim]
        K = self.wk(torch.cat([z_m, e_m], dim=-1)) # [M, attn_dim]
        V = self.wv(torch.cat([z_m, e_m, mem_y.unsqueeze(-1)], dim=-1)) # [M, attn_dim]
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.shape[-1] ** 0.5) # [B, M]
        attn_weights = F.softmax(scores, dim=-1)
        
        context = torch.matmul(attn_weights, V) # [B, attn_dim]
        
        # 4. Final Prediction
        logits = self.predictor(torch.cat([z_q, e_q, context], dim=-1)).squeeze(-1)
        return logits