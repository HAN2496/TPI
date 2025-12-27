from src.model import OfflineLSTM, OnlineMLP, OnlineCombination, OnlineLSTM, OnlineAttention

MODELS = {
    "offline_lstm": OfflineLSTM,
    "online_lstm": OnlineLSTM,
    "online_mlp": OnlineMLP,
    "linear_quad": OnlineCombination,
    "quad_only": OnlineCombination,
    "quad_exp": OnlineCombination,
    "online_combination": OnlineCombination,
    "online_attention": OnlineAttention,
}
