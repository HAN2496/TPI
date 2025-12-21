import yaml
import json
import torch
import optuna
from abc import ABC, abstractmethod
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader, TensorDataset

from src.configs.utils import load_config, create_model, load_features
from src.utils import prepare_training_data, ExperimentPaths
from src.utils.utils import save_config
from src.utils.trainer import BaseTrainer, CombinationTrainer
from src.model.linear_combination import OnlineCombination

class FeatureSelectionDataset(Dataset):
    def __init__(self, base_dataset, active_indices, n_features):
        self.base_dataset = base_dataset
        self.active = sorted(active_indices)
        self.inactive = [i for i in range(n_features) if i not in self.active]

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]
        if self.inactive:
            x = x.clone()
            x[..., self.inactive] = 0.0
        return x, y


class BaseOptimizer(ABC):
    def __init__(self, driver_name, model_type, model_name, time_range, feature_version, downsample,
                 use_feature_selection=False, device="cpu", verbose=1):
        self.driver_name = driver_name
        self.model_type = model_type
        self.model_name = model_name
        self.time_range = time_range
        self.feature_version = feature_version
        self.downsample = downsample
        self.use_feature_selection = use_feature_selection
        self.device = device
        self.verbose = verbose
        self.base_config = load_config(driver_name, model_name)

        self.train_loader = None
        self.X_val = None
        self.y_val = None
        self._load_data()

    def _load_data(self):
        self.train_loader, self.val_loader = prepare_training_data(
            self.driver_name,
            self.base_config,
            self.time_range,
            self.downsample,
            feature_version=self.feature_version
        )
        X_val, y_val = self.val_loader.dataset.tensors
        self.X_val = X_val.to(self.device)
        self.y_val = y_val.to(self.device)
        self.n_features = self.X_val.shape[-1]

    def _create_trial_config(self, trial):
        config = deepcopy(self.base_config)

        self._suggest_model_params(trial, config)

        selected_indices = None
        if self.use_feature_selection:
            mask = [trial.suggest_categorical(f"use_feat_{i}", [0, 1]) for i in range(self.n_features)]
            if sum(mask) == 0:
                mask[0] = 1
            selected_indices = [i for i, v in enumerate(mask) if v == 1]

        config.setdefault('features', {})
        config['features']['selected_indices'] = selected_indices
        config['features']['use_feature_selection'] = self.use_feature_selection

        return config

    @abstractmethod
    def _suggest_model_params(self, trial, config):
        raise NotImplementedError

    def _train_and_evaluate(self, config, trial):
        train_data = self.train_loader.dataset
        feature_cfg = config.get('features', {})
        feature_indices = feature_cfg.get('selected_indices')

        if feature_cfg.get('use_feature_selection', False) and feature_indices:
            train_dataset = FeatureSelectionDataset(train_data, feature_indices, self.n_features)
            val_dataset = FeatureSelectionDataset(self.val_loader.dataset, feature_indices, self.n_features)
            new_train_loader = DataLoader(train_dataset, batch_size=config['trainer']['batch_size'], shuffle=True)
            new_val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
        else:
            new_train_loader = DataLoader(train_data, batch_size=config['trainer']['batch_size'], shuffle=True)
            new_val_loader = self.val_loader

        model = create_model(self.driver_name, self.model_name, self.feature_version, device=self.device)
        trainer = BaseTrainer(model, config['trainer'], device=self.device)

        trainer.train(new_train_loader, new_val_loader, epochs=30, verbose=False)
        return trainer.best_auroc

    def objective(self, trial):
        config = self._create_trial_config(trial)
        score = self._train_and_evaluate(config, trial)
        return score

    def optimize(self, n_trials=100, tag="optuna"):
        paths = ExperimentPaths(self.driver_name, self.model_name,
                               self.feature_version, self.time_range, tag=tag)

        storage_path = paths.get("study.db", create=True)
        direction = "maximize"

        n_startup_trials = 100 if self.model_type == 'online_combination' else 30
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=n_startup_trials,
            n_ei_candidates=128,
            multivariate=True,
            group=True,
            constant_liar=True,
            seed=42
        )

        study = optuna.create_study(
            study_name=f"{self.model_name}_optimization",
            direction=direction,
            storage=f"sqlite:///{storage_path}",
            load_if_exists=True,
            sampler=sampler,
        )

        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)

        best_params_path = paths.get("best_params.json", create=True)
        with open(best_params_path, 'w') as f:
            json.dump(study.best_params, f, indent=2)

        best_config = self._create_trial_config(study.best_trial)
        best_config_path = paths.get("best_config.yaml", create=True)
        save_config(best_config, self.driver_name, self.model_name, best_config_path)

        print(f"\nBest AUROC: {study.best_value:.4f}")
        print(f"Best params saved to: {best_params_path}")
        print(f"Best config saved to: {best_config_path}")

        return study


class MLPOptimizer(BaseOptimizer):
    def _suggest_model_params(self, trial, config):
        n_layers = trial.suggest_int('n_layers', 3, 8)
        hidden_dims = [trial.suggest_categorical(f'h{i}', [16, 32, 64, 128, 256]) for i in range(n_layers)]

        config['args']['hidden_dims']   = hidden_dims
        config['args']['dropout_rate']  = trial.suggest_float('dropout', 0.0, 0.5)
        config['args']['act_name']      = trial.suggest_categorical('act', ['relu', 'gelu', 'silu'])
        config['args']['use_batchnorm'] = trial.suggest_categorical('batchnorm', [True, False])

        config['trainer']['learning_rate'] = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        config['trainer']['weight_decay']  = trial.suggest_float('wd', 0.0, 0.01)
        config['trainer']['batch_size']    = trial.suggest_categorical('batch_size', [32, 64, 128, 256])

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

        config['trainer']['learning_rate'] = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        config['trainer']['weight_decay']  = trial.suggest_float('wd', 0.0, 0.01)
        config['trainer']['batch_size']    = trial.suggest_categorical('batch_size', [32, 64, 128, 256])

class LinearCombinationOptimizer(BaseOptimizer):
    def _suggest_model_params(self, trial, config):
        config['args']['C'] = trial.suggest_float('C', 0.01, 10.0, log=True)

        if config['args']['form'] == 'quad_exp':
            w3_list = [trial.suggest_float(f'w3_{i}', 0.1, 10.0, log=True) for i in range(self.n_orig_features)]
            w4_list = [trial.suggest_float(f'w4_{i}', 0.0, 5.0) for i in range(self.n_orig_features)]
            config['args']['w3'] = w3_list
            config['args']['w4'] = w4_list

    def _load_data(self):
        self.train_loader, self.val_loader = prepare_training_data(
            self.driver_name,
            self.base_config,
            self.time_range,
            self.downsample,
            feature_version=self.feature_version
        )

        X_train, y_train = self.train_loader.dataset.tensors
        X_val, y_val = self.val_loader.dataset.tensors

        B_train, T, F = X_train.shape
        B_val = X_val.shape[0]

        self.X_train = X_train.reshape(B_train, T * F).numpy()
        self.y_train = y_train.numpy()
        self.X_val = X_val.reshape(B_val, T * F).numpy()
        self.y_val = y_val.numpy()

        feature_cols = load_features(self.feature_version)
        self.n_orig_features = len(feature_cols)
        self.seq_len = T
        self.n_features_flat = T * F

        self.n_features = self.n_orig_features

    def _train_and_evaluate(self, config, trial):
        feature_cfg = config.get('features', {})
        feature_indices = feature_cfg.get('selected_indices')

        if feature_cfg.get('use_feature_selection', False) and feature_indices:
            active_flat = []
            for i in range(self.n_orig_features):
                if i in feature_indices:
                    for t in range(self.seq_len):
                        active_flat.append(t * self.n_orig_features + i)
            X_train = self.X_train[:, active_flat]
            X_val = self.X_val[:, active_flat]
            n_features = len(feature_indices)
        else:
            X_train = self.X_train
            X_val = self.X_val
            n_features = self.n_orig_features

        X_train = X_train.reshape(-1, self.seq_len, n_features)
        X_val = X_val.reshape(-1, self.seq_len, n_features)

        args = config['args']
        model = OnlineCombination(
            input_dim=n_features,
            form=args['form'],
            max_iter=args['max_iter'],
            C=args['C'],
            solver=args['solver'],
            random_state=42,
            w3=args.get('w3'),
            w4=args.get('w4'),
        )

        train_ds = TensorDataset(torch.as_tensor(X_train, dtype=torch.float32), torch.as_tensor(self.y_train, dtype=torch.float32))
        val_ds = TensorDataset(torch.as_tensor(X_val, dtype=torch.float32), torch.as_tensor(self.y_val, dtype=torch.float32))
        train_loader = DataLoader(train_ds, batch_size=len(train_ds), shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)

        trainer = CombinationTrainer(model, config['trainer'])
        _, best_auroc, _ = trainer.train(train_loader, val_loader, verbose=False)

        return best_auroc


def create_optimizer(driver_name, model_name, time_range, feature_version, downsample,
                     use_feature_selection=False, device="cpu"):
    config = load_config(driver_name, model_name)
    model_type = config['model_type']

    if model_type == 'online_mlp':
        return MLPOptimizer(driver_name, model_type, model_name, time_range, feature_version, downsample, use_feature_selection=use_feature_selection, device=device)
    elif model_type == 'offline_lstm':
        return LSTMOptimizer(driver_name, model_type, model_name, time_range, feature_version, downsample, use_feature_selection=use_feature_selection, device=device)
    elif model_type == 'online_combination':
        return LinearCombinationOptimizer(driver_name, model_type, model_name, time_range, feature_version, downsample, use_feature_selection=use_feature_selection, device=device)
    raise ValueError(f"Unknown model_type: {model_type}")
