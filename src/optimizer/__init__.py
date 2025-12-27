from src.configs.utils import load_config

from .base import BaseOptimizer, BayesianOptimizer, ExhaustiveOptimizer, FeatureSelectionDataset
from .bayesian import MLPOptimizer, LSTMOptimizer, AttentionOptimizer, LinearCombinationOptimizer
from .exhaustive import ExhaustiveMLPOptimizer, ExhaustiveLSTMOptimizer, ExhaustiveAttentionOptimizer, ExhaustiveLinearCombinationOptimizer

def create_optimizer(driver_name, model_type, time_range, downsample, n_splits,
                     use_feature_selection=False, device="cpu", optimizer_type="bayesian"):
    config = load_config(driver_name, model_type, 'base')
    config_model_type = config['model_type']

    if optimizer_type == "bayesian":
        if config_model_type == 'online_mlp':
            return MLPOptimizer(driver_name, model_type, time_range, downsample, n_splits, use_feature_selection=use_feature_selection, device=device)
        elif config_model_type in ('offline_lstm', 'online_lstm'):
            return LSTMOptimizer(driver_name, model_type, time_range, downsample, n_splits, use_feature_selection=use_feature_selection, device=device)
        elif config_model_type == 'online_attention':
            return AttentionOptimizer(driver_name, model_type, time_range, downsample, n_splits, use_feature_selection=use_feature_selection, device=device)
        elif config_model_type in ('quad_only', 'linear_quad', 'quad_exp'):
            return LinearCombinationOptimizer(driver_name, model_type, time_range, downsample, n_splits, use_feature_selection=use_feature_selection, device=device)

    elif optimizer_type == "exhaustive":
        if config_model_type == 'online_mlp':
            return ExhaustiveMLPOptimizer(driver_name, model_type, time_range, downsample, n_splits, use_feature_selection=use_feature_selection, device=device)
        elif config_model_type in ('offline_lstm', 'online_lstm'):
            return ExhaustiveLSTMOptimizer(driver_name, model_type, time_range, downsample, n_splits, use_feature_selection=use_feature_selection, device=device)
        elif config_model_type == 'online_attention':
            return ExhaustiveAttentionOptimizer(driver_name, model_type, time_range, downsample, n_splits, use_feature_selection=use_feature_selection, device=device)
        elif config_model_type in ('quad_only', 'linear_quad', 'quad_exp'):
            return ExhaustiveLinearCombinationOptimizer(driver_name, model_type, time_range, downsample, n_splits, use_feature_selection=use_feature_selection, device=device)
    raise ValueError(f"Unknown model_type: {config_model_type} or optimizer_type: {optimizer_type}")
