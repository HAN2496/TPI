import torch.nn as nn

from .base import BaseModel, build_mlp

class OnlineMLP(BaseModel):
    def __init__(self, input_dim, hidden_dims, dropout_rate, act_name, use_batchnorm=False, reduce='mean'):
        super().__init__(reduce=reduce)

        act = self._get_activation(act_name)
        self.net = build_mlp(input_dim, hidden_dims, act, dropout_rate, use_batchnorm)

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

class OnlineLSTM(BaseModel):
    def __init__(self, input_dim,
                 lstm_hidden_dim, lstm_layers,
                 hidden_dims, dropout_rate, act_name, use_batchnorm=False, reduce='mean'):
        super().__init__(reduce=reduce)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )

        act = self._get_activation(act_name)
        self.head = build_mlp(lstm_hidden_dim, hidden_dims, act, dropout_rate, use_batchnorm)

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