import torch.nn as nn

from .base import BaseModel, build_mlp

class OfflineLSTM(BaseModel):
    def __init__(self, input_dim,
                 lstm_hidden_dim, lstm_layers, bidirectional,
                 hidden_dims, dropout_rate, act_name, use_batchnorm=False, reduce='mean'):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        act = self._get_activation(act_name)
        self.head = build_mlp(lstm_hidden_dim, hidden_dims, act, dropout_rate, use_batchnorm)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h = h[-1]
        out = self.head(h)
        return out.squeeze(-1)