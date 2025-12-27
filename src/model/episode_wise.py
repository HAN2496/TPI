import torch.nn as nn

from .base import NeuralModel, build_mlp

class OfflineLSTM(NeuralModel):
    def __init__(self, input_dim,
                 lstm_hidden_dim, lstm_layers, lstm_dropout, bidirectional,
                 hidden_dims, dropout_rates, act_name, use_batchnorm=False, reduce='mean'):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
            batch_first=True,
            bidirectional=bidirectional
        )

        act = self._get_activation(act_name)
        self.head = build_mlp(lstm_hidden_dim, hidden_dims, act, dropout_rates, use_batchnorm)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h = h[-1]
        out = self.head(h)
        return out.squeeze(-1)