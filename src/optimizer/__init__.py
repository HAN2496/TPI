from src.configs.utils import load_config

from .base import BaseOptimizer, BayesianOptimizer, ExhaustiveOptimizer, FeatureSelectionDataset
from .bayesian import MLPOptimizer, LSTMOptimizer, AttentionOptimizer, OnlineRegressionOptimizer, OfflineRegressionOptimizer
from .exhaustive import (ExhaustiveMLPOptimizer, ExhaustiveLSTMOptimizer, ExhaustiveAttentionOptimizer,
                         ExhaustiveOnlineRegressionOptimizer, ExhaustiveOfflineRegressionOptimizer)

OPTIMIZERS={
    'bayesian': {
        'online_mlp': MLPOptimizer,
        'offline_lstm': LSTMOptimizer,
        'online_lstm': LSTMOptimizer,
        'online_attention': AttentionOptimizer,
        'online_regression': OnlineRegressionOptimizer,
        'offline_regression': OfflineRegressionOptimizer
    },
    'exhaustive': {
        'online_mlp': ExhaustiveMLPOptimizer,
        'offline_lstm': ExhaustiveLSTMOptimizer,
        'online_lstm': ExhaustiveLSTMOptimizer,
        'online_attention': ExhaustiveAttentionOptimizer,
        'online_regression': ExhaustiveOnlineRegressionOptimizer,
        'offline_regression': ExhaustiveOfflineRegressionOptimizer
    }
}

def create_optimizer(driver_name, model_type, time_range, downsample, n_splits, test_ratio,
                     use_feature_selection=False, device="cpu", optimizer_type="bayesian"):
    config = load_config(driver_name, model_type, 'base')
    config_model_type = config['model_type']

    optimizer = OPTIMIZERS[optimizer_type][config_model_type](driver_name, model_type, time_range, downsample, n_splits, test_ratio,
                                                          use_feature_selection=use_feature_selection, device=device)
    return optimizer