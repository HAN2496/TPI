import torch
import torch.nn as nn

from .base import NeuralModel, build_mlp

class OnlineMLP(NeuralModel):
    def __init__(self, input_dim, hidden_dims, dropout_rates, act_name, use_batchnorm=False, reduce='mean'):
        super().__init__(reduce=reduce)

        act = self._get_activation(act_name)
        self.net = build_mlp(input_dim, hidden_dims, act, dropout_rates, use_batchnorm)

    def step_rewards(self, x, detach=False):
        b, t, d = x.shape # (Batch size, Seq len, Dim)
        r = self.net(x.reshape(-1, d)).squeeze(-1).reshape(b, t)  # (B,T)
        if detach:
            r = r.detach().cpu().numpy()
        return r.reshape(b, t)

    def forward(self, x):
        r = self.step_rewards(x)  # (B,T)
        if self.reduce == 'sum':
            return r.sum(dim=1)
        if self.reduce == 'mean':
            return r.mean(dim=1)
        raise ValueError(f"Unknown reduce: {self.reduce}")

class OnlineLSTM(NeuralModel):
    def __init__(self, input_dim,
                 lstm_hidden_dim, lstm_layers, lstm_dropout,
                 hidden_dims, dropout_rates, act_name, use_batchnorm=False, reduce='mean'):
        super().__init__(reduce=reduce)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
            batch_first=True
        )

        act = self._get_activation(act_name)
        self.head = build_mlp(lstm_hidden_dim, hidden_dims, act, dropout_rates, use_batchnorm)

    def step_rewards(self, x, detach=False):
        h, _ = self.lstm(x) # (Batch size, Seq len, Dim)
        b, t, hdim = h.shape
        r = self.head(h.reshape(-1, hdim)).squeeze(-1).reshape(b, t)  # (B,T)
        if detach:
            r = r.detach().cpu().numpy()
        return r

    def forward(self, x):
        r = self.step_rewards(x)
        if self.reduce == 'sum':
            return r.sum(dim=1)
        if self.reduce == 'mean':
            return r.mean(dim=1)
        raise ValueError(f"Unknown reduce: {self.reduce}")


class OnlineAttention(NeuralModel):
    def __init__(self, input_dim, attn_dim, attn_heads, attn_dropout,
                 ffn_dim, ffn_dropout, hidden_dims, dropout_rates, act_name,
                 use_batchnorm=False, reduce='mean', use_ffn=True):
        super().__init__(reduce=reduce)

        if attn_dim % attn_heads != 0:
            raise ValueError(f"attn_dim({attn_dim}) must be divisible by attn_heads({attn_heads})")

        self.use_ffn = use_ffn

        self.in_proj = nn.Linear(input_dim, attn_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=attn_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        self.attn_ln = nn.LayerNorm(attn_dim)
        self.attn_drop = nn.Dropout(attn_dropout)

        if use_ffn:
            self.ffn = nn.Sequential(
                nn.Linear(attn_dim, ffn_dim),
                self._get_activation(act_name),
                nn.Dropout(ffn_dropout),
                nn.Linear(ffn_dim, attn_dim),
                nn.Dropout(ffn_dropout),
            )
            self.ffn_ln = nn.LayerNorm(attn_dim)

        act = self._get_activation(act_name)
        self.head = build_mlp(attn_dim, hidden_dims, act, dropout_rates, use_batchnorm)

    def _causal_attn_mask(self, T, device):
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def step_rewards(self, x, detach=False):
        b, t, _ = x.shape
        h = self.in_proj(x)

        attn_mask = self._causal_attn_mask(t, h.device)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)

        h = self.attn_ln(h + self.attn_drop(attn_out))

        if self.use_ffn:
            h2 = self.ffn(h)
            h = self.ffn_ln(h + h2)

        r = self.head(h.reshape(b * t, -1)).squeeze(-1).reshape(b, t)

        if detach:
            return r.detach().cpu().numpy()
        return r

    def forward(self, x):
        r = self.step_rewards(x)
        if self.reduce == 'sum':
            return r.sum(dim=1)
        if self.reduce == 'mean':
            return r.mean(dim=1)
        raise ValueError(f"Unknown reduce: {self.reduce}")
