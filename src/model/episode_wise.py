import torch
import torch.nn as nn

from src.model.base import BaseModel

class OfflineLSTM(BaseModel):
    def __init__(self,
                 input_dim,
                 lstm_hidden, lstm_layers,
                 mlp_hiddens, mlp_dropout,
                 act_name, use_batchnorm=False):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, lstm_hidden, num_layers=lstm_layers, batch_first=True)

        if act_name == "relu":
            act = nn.ReLU()
        elif act_name == "tanh":
            act = nn.Tanh()
        else:
            act = nn.GELU()

        layers = []
        in_dim = lstm_hidden
        for hidden_dim in mlp_hiddens:
            layers.append(nn.Linear(in_dim, hidden_dim))

            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            if mlp_dropout > 0:
                layers.append(nn.Dropout(mlp_dropout))

            layers.append(act)
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h = h[-1]
        out = self.mlp(h)
        return out.squeeze(-1)
    
    def predict_reward(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
        return logits
    
    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
        return probs

    def predict(self, x, threshold=0.5):
        probs = self.predict_proba(x)
        return (probs >= threshold).long()