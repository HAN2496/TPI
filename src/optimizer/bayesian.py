from .base import BayesianOptimizer

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

class LinearCombinationOptimizer(BayesianOptimizer):
    n_startup_trials = 100

    def __init__(self, driver_name, model_type, time_range, downsample, n_splits,
                 use_feature_selection=False, device="cpu", verbose=1):
        super().__init__(driver_name, model_type, time_range, downsample, n_splits,
                        use_feature_selection, device, verbose)
        self.n_orig_features = len(self.base_config['features'])

    def _suggest_model_params(self, trial, config):
        config['args']['C'] = trial.suggest_float('C', 0.01, 10.0, log=True)

        if config['args']['form'] == 'quad_exp':
            w3_list = [trial.suggest_float(f'w3_{i}', 0.1, 10.0, log=True) for i in range(self.n_orig_features)]
            w4_list = [trial.suggest_float(f'w4_{i}', 0.0, 5.0) for i in range(self.n_orig_features)]
            config['args']['w3'] = w3_list
            config['args']['w4'] = w4_list
