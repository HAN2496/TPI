import itertools
from .base import ExhaustiveOptimizer

class ExhaustiveMLPOptimizer(ExhaustiveOptimizer):
    def _suggest_model_params(self, trial, config):
        pass

    def _get_search_space(self):
        return {
            'n_layers': [3, 4, 5, 6, 7],
            'hidden_dim_options': [32, 64, 128, 256],
            'dropout_options': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            'act': ['relu', 'gelu', 'silu'],
            'batchnorm': [True, False],
        }

    def _generate_all_configs(self):
        space = self._get_search_space()
        all_configs = []
        for n_layers in space['n_layers']:
            for hidden_tuple in itertools.product(space['hidden_dim_options'], repeat=n_layers):
                for dropout_tuple in itertools.product(space['dropout_options'], repeat=n_layers):
                    for act, bn in itertools.product(space['act'], space['batchnorm']):
                        config_dict = {
                            'hidden_dims': list(hidden_tuple),
                            'dropout_rates': list(dropout_tuple),
                            'act_name': act,
                            'use_batchnorm': bn,
                        }
                        all_configs.append(config_dict)
        return all_configs

    def _apply_config_dict(self, config, config_dict):
        config['args']['hidden_dims'] = config_dict['hidden_dims']
        config['args']['dropout_rates'] = config_dict['dropout_rates']
        config['args']['act_name'] = config_dict['act_name']
        config['args']['use_batchnorm'] = config_dict['use_batchnorm']

class ExhaustiveLSTMOptimizer(ExhaustiveOptimizer):
    def _suggest_model_params(self, trial, config):
        pass

    def _get_search_space(self):
        return {
            'lstm_hidden_dim': [32, 64, 128, 256],
            'lstm_layers': [1, 2, 3, 4],
            'lstm_dropout': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            'n_mlp_layers': [1, 2, 3],
            'mlp_hidden_dim_options': [32, 64, 128],
            'dropout_options': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            'act': ['relu', 'gelu', 'tanh'],
            'batchnorm': [True, False],
        }

    def _generate_all_configs(self):
        space = self._get_search_space()
        all_configs = []
        for lstm_h, lstm_l, lstm_drop in itertools.product(
            space['lstm_hidden_dim'], space['lstm_layers'], space['lstm_dropout']
        ):
            if lstm_l == 1 and lstm_drop != 0.0:
                continue
            for n_mlp in space['n_mlp_layers']:
                for mlp_hidden_tuple in itertools.product(space['mlp_hidden_dim_options'], repeat=n_mlp):
                    for dropout_tuple in itertools.product(space['dropout_options'], repeat=n_mlp):
                        for act, bn in itertools.product(space['act'], space['batchnorm']):
                            config_dict = {
                                'lstm_hidden_dim': lstm_h,
                                'lstm_layers': lstm_l,
                                'lstm_dropout': lstm_drop,
                                'hidden_dims': list(mlp_hidden_tuple),
                                'dropout_rates': list(dropout_tuple),
                                'act_name': act,
                                'use_batchnorm': bn,
                            }
                            all_configs.append(config_dict)
        return all_configs

    def _apply_config_dict(self, config, config_dict):
        config['args']['lstm_hidden_dim'] = config_dict['lstm_hidden_dim']
        config['args']['lstm_layers'] = config_dict['lstm_layers']
        config['args']['lstm_dropout'] = config_dict['lstm_dropout']
        config['args']['hidden_dims'] = config_dict['hidden_dims']
        config['args']['dropout_rates'] = config_dict['dropout_rates']
        config['args']['act_name'] = config_dict['act_name']
        config['args']['use_batchnorm'] = config_dict['use_batchnorm']

class ExhaustiveAttentionOptimizer(ExhaustiveOptimizer):
    def _suggest_model_params(self, trial, config):
        pass

    def _get_search_space(self):
        return {
            'attn_dim': [64, 128, 256],
            'attn_heads': [2, 4, 8],
            'attn_dropout': [0.0, 0.1, 0.2, 0.3],
            'use_ffn': [True, False],
            'ffn_multiplier': [2, 4],
            'ffn_dropout': [0.0, 0.1, 0.2, 0.3],
            'n_mlp_layers': [1, 2, 3],
            'mlp_hidden_dim_options': [32, 64, 128],
            'dropout_options': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            'act': ['relu', 'gelu', 'silu'],
            'batchnorm': [True, False],
        }

    def _generate_all_configs(self):
        space = self._get_search_space()
        all_configs = []
        for attn_dim, attn_heads in itertools.product(space['attn_dim'], space['attn_heads']):
            if attn_dim % attn_heads != 0:
                continue
            for attn_drop, use_ffn in itertools.product(space['attn_dropout'], space['use_ffn']):
                ffn_configs = [(None, None)] if not use_ffn else [
                    (attn_dim * mult, ffn_drop)
                    for mult, ffn_drop in itertools.product(space['ffn_multiplier'], space['ffn_dropout'])
                ]
                for ffn_dim, ffn_drop in ffn_configs:
                    for n_mlp in space['n_mlp_layers']:
                        for mlp_hidden_tuple in itertools.product(space['mlp_hidden_dim_options'], repeat=n_mlp):
                            for dropout_tuple in itertools.product(space['dropout_options'], repeat=n_mlp):
                                for act, bn in itertools.product(space['act'], space['batchnorm']):
                                    config_dict = {
                                        'attn_dim': attn_dim,
                                        'attn_heads': attn_heads,
                                        'attn_dropout': attn_drop,
                                        'use_ffn': use_ffn,
                                        'ffn_dim': ffn_dim,
                                        'ffn_dropout': ffn_drop,
                                        'hidden_dims': list(mlp_hidden_tuple),
                                        'dropout_rates': list(dropout_tuple),
                                        'act_name': act,
                                        'use_batchnorm': bn,
                                    }
                                    all_configs.append(config_dict)
        return all_configs

    def _apply_config_dict(self, config, config_dict):
        config['args']['attn_dim'] = config_dict['attn_dim']
        config['args']['attn_heads'] = config_dict['attn_heads']
        config['args']['attn_dropout'] = config_dict['attn_dropout']
        config['args']['use_ffn'] = config_dict['use_ffn']
        if config_dict['use_ffn']:
            config['args']['ffn_dim'] = config_dict['ffn_dim']
            config['args']['ffn_dropout'] = config_dict['ffn_dropout']
        config['args']['hidden_dims'] = config_dict['hidden_dims']
        config['args']['dropout_rates'] = config_dict['dropout_rates']
        config['args']['act_name'] = config_dict['act_name']
        config['args']['use_batchnorm'] = config_dict['use_batchnorm']

class ExhaustiveOnlineRegressionOptimizer(ExhaustiveOptimizer):
    def _suggest_model_params(self, trial, config):
        pass

    def _get_search_space(self):
        return {
            'C': [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
            'ops_per_feature': [
                ['quad'],
                ['abs', 'quad'],
                ['quad', 'cube'],
                ['abs', 'quad', 'shifted_exp'],
            ]
        }

    def _generate_all_configs(self):
        space = self._get_search_space()
        all_configs = []
        for C in space['C']:
            for ops in space['ops_per_feature']:
                config_dict = {
                    'C': C,
                    'ops': ops
                }
                all_configs.append(config_dict)
        return all_configs

    def _apply_config_dict(self, config, config_dict):
        config['args']['C'] = config_dict['C']
        basis = {feat_name: config_dict['ops'] for feat_name in config['features']}
        config['args']['basis'] = basis

        if 'shifted_exp' in config_dict['ops']:
            config['args']['w3'] = {feat_name: 1.0 for feat_name in config['features']}
            config['args']['w4'] = {feat_name: 0.0 for feat_name in config['features']}

class ExhaustiveOfflineRegressionOptimizer(ExhaustiveOptimizer):
    def _suggest_model_params(self, trial, config):
        pass

    def _get_search_space(self):
        return {
            'C': [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
            'ops_per_feature': [
                ['mean'],
                ['mean', 'std'],
                ['mean', 'max', 'min'],
                ['mean', 'std', 'max', 'min'],
                ['mean', 'abs_mean', 'rmse'],
            ]
        }

    def _generate_all_configs(self):
        space = self._get_search_space()
        all_configs = []
        for C in space['C']:
            for ops in space['ops_per_feature']:
                config_dict = {
                    'C': C,
                    'ops': ops
                }
                all_configs.append(config_dict)
        return all_configs

    def _apply_config_dict(self, config, config_dict):
        config['args']['C'] = config_dict['C']
        basis = {feat_name: config_dict['ops'] for feat_name in config['features']}
        config['args']['basis'] = basis
