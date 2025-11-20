import yaml
import json
import optuna
from abc import ABC, abstractmethod
from copy import deepcopy
from torch.utils.data import DataLoader
from src.configs.utils import load_config, create_model
from src.utils import prepare_training_data, ExperimentPaths
from src.utils.trainer import BaseTrainer

class BaseOptimizer(ABC):
    def __init__(self, driver_name, model_name, time_range, downsample, device="cpu"):
        self.driver_name = driver_name
        self.model_name = model_name
        self.time_range = time_range
        self.downsample = downsample
        self.device = device
        self.base_config = load_config(driver_name, model_name)

        self.train_loader = None
        self.X_val = None
        self.y_val = None
        self._load_data()

    def _load_data(self):
        self.train_loader, _, _, self.X_val, self.y_val = prepare_training_data(
            self.driver_name,
            self.base_config,
            self.time_range,
            self.downsample
        )
        self.X_val = self.X_val.to(self.device)
        self.y_val = self.y_val.to(self.device)

    @abstractmethod
    def _suggest_model_params(self, trial, config):
        raise NotImplementedError

    def _create_trial_config(self, trial):
        config = deepcopy(self.base_config)

        config['trainer']['learning_rate'] = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        config['trainer']['weight_decay'] = trial.suggest_float('wd', 0.0, 0.01)
        config['trainer']['batch_size'] = trial.suggest_categorical('batch_size', [32, 64, 128, 256])

        self._suggest_model_params(trial, config)

        return config

    def _train_and_evaluate(self, config, trial):
        train_data = self.train_loader.dataset
        new_loader = DataLoader(train_data, batch_size=config['trainer']['batch_size'], shuffle=True)

        model = create_model(self.driver_name, self.model_name, device=self.device)
        trainer = BaseTrainer(model, config['trainer'])

        _, auroc, _ = trainer.fit(new_loader, self.X_val, self.y_val, epochs=30, verbose=False)

        return auroc

    def objective(self, trial):
        config = self._create_trial_config(trial)
        auroc = self._train_and_evaluate(config, trial)
        return auroc

    def optimize(self, n_trials=100, tag="optuna"):
        paths = ExperimentPaths(self.driver_name, self.model_name,
                               self.base_config['feature_version'], self.time_range, tag=tag)

        storage_path = paths.get("study.db", create=True)
        study = optuna.create_study(
            study_name=f"{self.model_name}_optimization",
            direction="maximize",
            storage=f"sqlite:///{storage_path}",
            load_if_exists=True
        )

        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)

        best_params_path = paths.get("best_params.json", create=True)
        with open(best_params_path, 'w') as f:
            json.dump(study.best_params, f, indent=2)

        best_config = self._create_trial_config(study.best_trial)
        best_config_path = paths.get("best_config.yaml", create=True)
        with open(best_config_path, 'w') as f:
            yaml.dump(best_config, f, default_flow_style=False)

        print(f"\nBest AUROC: {study.best_value:.4f}")
        print(f"Best params saved to: {best_params_path}")
        print(f"Best config saved to: {best_config_path}")

        return study


class MLPOptimizer(BaseOptimizer):
    def _suggest_model_params(self, trial, config):
        n_layers = trial.suggest_int('n_layers', 1, 4)
        hidden_dims = [trial.suggest_categorical(f'h{i}', [32, 64, 128, 256]) for i in range(n_layers)]

        config['args']['hidden_dims']   = hidden_dims
        config['args']['dropout_rate']  = trial.suggest_float('dropout', 0.0, 0.5)
        config['args']['act_name']      = trial.suggest_categorical('act', ['relu', 'gelu', 'tanh'])
        config['args']['use_batchnorm'] = trial.suggest_categorical('batchnorm', [True, False])


class LSTMOptimizer(BaseOptimizer):
    def _suggest_model_params(self, trial, config):
        config['args']['lstm_hidden']  = trial.suggest_categorical('lstm_h', [32, 64, 128, 256])
        config['args']['lstm_layers']  = trial.suggest_int('lstm_l', 1, 3)
        config['args']['lstm_dropout'] = trial.suggest_float('lstm_dropout', 0.0, 0.5)

        n_mlp_layers = trial.suggest_int('n_mlp_layers', 1, 3)
        mlp_hiddens = [trial.suggest_categorical(f'mlp_h{i}', [32, 64, 128]) for i in range(n_mlp_layers)]

        config['args']['mlp_hiddens']   = mlp_hiddens
        config['args']['mlp_dropout']   = trial.suggest_float('mlp_dropout', 0.0, 0.5)
        config['args']['act_name']      = trial.suggest_categorical('act', ['relu', 'gelu', 'tanh'])
        config['args']['use_batchnorm'] = trial.suggest_categorical('batchnorm', [True, False])


class LinearCombinationOptimizer(BaseOptimizer):
    def _suggest_model_params(self, trial, config):
        config['args']['C']        = trial.suggest_float('C', 0.01, 10.0, log=True)
        config['args']['solver']   = trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'saga'])
        config['args']['max_iter'] = trial.suggest_int('max_iter', 100, 1000)

    def _load_data(self):
        self.train_loader, self.X_train, self.y_train, self.X_val, self.y_val = prepare_training_data(
            self.driver_name,
            self.base_config,
            self.time_range,
            self.downsample,
            flatten=True
        )

    def _train_and_evaluate(self, config, trial):
        from src.utils.trainer import CombinationTrainer

        model = create_model(self.driver_name, self.model_name, device=self.device)
        trainer = CombinationTrainer(model, config['trainer'])
        _, auroc, _ = trainer.fit(self.X_train, self.y_train, self.X_val, self.y_val, verbose=False)

        return auroc


def create_optimizer(driver_name, model_name, time_range, downsample, device="cpu"):
    config = load_config(driver_name, model_name)
    model_type = config['model_type']

    if model_type == 'online_mlp':
        return MLPOptimizer(driver_name, model_name, time_range, downsample, device)
    elif model_type == 'offline_lstm':
        return LSTMOptimizer(driver_name, model_name, time_range, downsample, device)
    elif model_type == 'online_combination':
        return LinearCombinationOptimizer(driver_name, model_name, time_range, downsample, device)

    raise ValueError(f"Unknown model_type: {model_type}")
