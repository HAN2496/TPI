from src.model import OfflineLSTM, OfflineRegression, OnlineMLP, OnlineRegression, OnlineLSTM, OnlineAttention

MODELS = {
    "offline_lstm": OfflineLSTM,
    "offline_regression": OfflineRegression,
    "online_mlp": OnlineMLP,
    "online_lstm": OnlineLSTM,
    "online_attention": OnlineAttention,
    "online_regression": OnlineRegression,
}
