from .base_2 import BayesianOptimizer

class MLPOptimizer(BayesianOptimizer):
    def _suggest_model_params(self, trial, config):
        n_layers = trial.suggest_int('n_layers', 3, 8)
        config['args']['hidden_dims']   = [trial.suggest_categorical(f'h{i}', [32, 64, 128, 256]) for i in range(n_layers)]
        config['args']['dropout_rates'] = [trial.suggest_categorical(f'dropout_{i}', [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]) for i in range(n_layers)]
        config['args']['act_name']      = trial.suggest_categorical('act', ['relu', 'gelu', 'silu'])
        config['args']['use_batchnorm'] = trial.suggest_categorical('batchnorm', [True, False])

        config['trainer']['learning_rate'] = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
        config['trainer']['weight_decay']  = trial.suggest_float('wd', 0.0, 0.01)

class LSTMOptimizer(BayesianOptimizer):
    def _suggest_model_params(self, trial, config):
        config['args']['lstm_hidden_dim']  = trial.suggest_categorical('lstm_h', [32, 64, 128, 256])
        config['args']['lstm_layers']      = trial.suggest_int('lstm_l', 1, 4)
        if config['args']['lstm_layers'] > 1:
            config['args']['lstm_dropout'] = trial.suggest_float('lstm_dropout', 0.0, 0.5)
        else:
            config['args']['lstm_dropout'] = 0.0

        n_mlp_layers = trial.suggest_int('n_mlp_layers', 1, 3)
        config['args']['hidden_dims']   = [trial.suggest_categorical(f'mlp_h{i}', [32, 64, 128]) for i in range(n_mlp_layers)]
        config['args']['dropout_rates'] = [trial.suggest_categorical(f'mlp_dropout_{i}', [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]) for i in range(n_mlp_layers)]
        config['args']['act_name']      = trial.suggest_categorical('act', ['relu', 'gelu', 'tanh'])
        config['args']['use_batchnorm'] = trial.suggest_categorical('batchnorm', [True, False])

        config['trainer']['learning_rate'] = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
        config['trainer']['weight_decay']  = trial.suggest_float('wd', 0.0, 0.01)

class AttentionOptimizer(BayesianOptimizer):
    def _suggest_model_params(self, trial, config):
        attn_dim = trial.suggest_categorical('attn_dim', [64, 128, 256])
        attn_heads = trial.suggest_categorical('attn_heads', [2, 4, 8])

        while attn_dim % attn_heads != 0:
            attn_heads = trial.suggest_categorical(f'attn_heads_retry', [2, 4, 8])

        config['args']['attn_dim'] = attn_dim
        config['args']['attn_heads'] = attn_heads
        config['args']['attn_dropout'] = trial.suggest_float('attn_dropout', 0.0, 0.3)

        config['args']['use_ffn'] = trial.suggest_categorical('use_ffn', [True, False])
        if config['args']['use_ffn']:
            ffn_mult = trial.suggest_categorical('ffn_mult', [2, 4])
            config['args']['ffn_dim'] = attn_dim * ffn_mult
            config['args']['ffn_dropout'] = trial.suggest_float('ffn_dropout', 0.0, 0.3)

        n_mlp_layers = trial.suggest_int('n_mlp_layers', 1, 3)
        config['args']['hidden_dims'] = [trial.suggest_categorical(f'mlp_h{i}', [32, 64, 128]) for i in range(n_mlp_layers)]
        config['args']['dropout_rates'] = [trial.suggest_categorical(f'mlp_dropout_{i}', [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]) for i in range(n_mlp_layers)]
        config['args']['act_name'] = trial.suggest_categorical('act', ['relu', 'gelu', 'silu'])
        config['args']['use_batchnorm'] = trial.suggest_categorical('batchnorm', [True, False])

        config['trainer']['learning_rate'] = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
        config['trainer']['weight_decay'] = trial.suggest_float('wd', 0.0, 0.01)

class OnlineRegressionOptimizer(BayesianOptimizer):
    n_startup_trials = 100

    def _suggest_model_params(self, trial, config):
        config['args']['C'] = trial.suggest_float('C', 0.1, 100.0, log=True)

        available_ops = ['abs', 'quad', 'cube', 'shifted_exp']
        basis = {}
        shifted_exp_features = []

        for feat_name in config['features']:
            selected_ops = [op for op in available_ops if trial.suggest_categorical(f'basis_{feat_name}_{op}', [0, 1])]
            if not selected_ops:
                selected_ops = ['quad']
            basis[feat_name] = selected_ops
            if 'shifted_exp' in selected_ops:
                shifted_exp_features.append(feat_name)

        config['args']['basis'] = basis

        if shifted_exp_features:
            w3_dict = {}
            w4_dict = {}
            for feat_name in shifted_exp_features:
                w3_dict[feat_name] = trial.suggest_float(f'w3_{feat_name}', 0.1, 50.0, log=True)
                w4_dict[feat_name] = trial.suggest_float(f'w4_{feat_name}', 0.0, 50.0)

            config['args']['w3'] = w3_dict
            config['args']['w4'] = w4_dict

class OfflineRegressionOptimizer(BayesianOptimizer):
    n_startup_trials = 100

    def _suggest_model_params(self, trial, config):
        config['args']['C'] = trial.suggest_float('C', 0.1, 100.0, log=True)

        available_ops = ['mean', 'std', 'max', 'min', 'abs_mean', 'sqrt_mean', 'rmse']
        basis = {}

        for feat_name in config['features']:
            selected_ops = [op for op in available_ops if trial.suggest_categorical(f'basis_{feat_name}_{op}', [0, 1])]
            if not selected_ops:
                selected_ops = ['mean']
            basis[feat_name] = selected_ops

        config['args']['basis'] = basis

OPTIMIZERS_V2 = {
    'online_mlp': MLPOptimizer,
    'offline_lstm': LSTMOptimizer,
    'online_attention': AttentionOptimizer,
    'online_regression': OnlineRegressionOptimizer,
    'offline_regression': OfflineRegressionOptimizer,
}
