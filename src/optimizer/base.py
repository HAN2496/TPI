import sys
import json
import optuna
import subprocess
import webbrowser
from abc import ABC, abstractmethod
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader, TensorDataset

from src.configs.utils import load_config
from src.configs.registries import MODELS
from src.model.base import RegressionModel, NeuralModel
from src.utils import ExperimentPaths, prepare_training_data_kfold, save_config
from model.trainer import BaseTrainer, RegressionTrainer, create_trainer

from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
import torch
from src.utils.utils import _load_dataset_sequences

class FeatureSelectionDataset(Dataset):
    def __init__(self, base_dataset, active_indices, n_features=None, mode='slice'):
        self.ds = base_dataset
        self.active = active_indices
        self.mode = mode
        if mode == 'mask':
            if n_features is None:
                raise ValueError("n_features must be provided for 'mask' mode.")
            self.inactive = [i for i in range(n_features) if i not in set(active_indices)]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, y = self.ds[idx]

        if self.mode == 'slice':
            return x[:, self.active], y
        elif self.mode == 'mask':
            x = x.clone()
            x[..., self.inactive] = 0.0
            return x, y

class BaseOptimizer(ABC):
    def __init__(self, driver_name, model_type, time_range, downsample, n_splits, test_ratio,
                 use_feature_selection=False, device="cpu", verbose=1):
        self.driver_name = driver_name
        self.model_type = model_type
        self.time_range = time_range
        self.device = device
        self.verbose = verbose

        self.base_config = load_config(driver_name, model_type, 'base')
        self.all_features = self.base_config['features']
        self.n_features = len(self.base_config['features'])

        self.use_feature_selection = use_feature_selection

        self._setup_data(downsample, n_splits, test_ratio)

    def _setup_data(self, downsample, n_splits, test_ratio):
        X, y = _load_dataset_sequences(self.driver_name, self.time_range, downsample, self.base_config)

        X_trainval, _, y_trainval, _ = train_test_split(X, y, test_size=test_ratio, random_state=42, stratify=y)

        self.X_data = torch.as_tensor(X_trainval, dtype=torch.float32)
        self.y_data = torch.as_tensor(y_trainval, dtype=torch.float32)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        y_np = np.asarray(y_trainval)
        
        self.folds_indices = list(skf.split(np.zeros(len(y_np)), y_np))

    @abstractmethod
    def _suggest_model_params(self, trial, config):
        raise NotImplementedError

    def _suggest_features(self, trial):
        if not self.use_feature_selection:
            return list(range(self.n_features))

        mask = [trial.suggest_categorical(f"use_feat_{i}", [0, 1]) for i in range(self.n_features)]

        if sum(mask) == 0:
            mask[0] = 1

        selected_indices = [i for i, v in enumerate(mask) if v == 1]
        return selected_indices

    def _create_trial_config(self, trial):
        config = deepcopy(self.base_config)
        self._suggest_model_params(trial, config)

        if self.use_feature_selection:
            selected_indices = self._suggest_features(trial)
            config['features'] = [self.all_features[i] for i in selected_indices]
            config['active_feature_indices'] = selected_indices # for internal use
        else:
            config['active_feature_indices'] = list(range(self.n_features))
        return config

    def _run_fold(self, train_idx, val_idx, config):
        active_indices = config['active_feature_indices']

        X_train = self.X_data[train_idx][:, :, active_indices]
        y_train = self.y_data[train_idx]

        X_val = self.X_data[val_idx][:, :, active_indices]
        y_val = self.y_data[val_idx]

        is_neural = issubclass(MODELS[config['model_type']], NeuralModel)
        batch_size = config['trainer']['batch_size'] if is_neural else len(X_train)

        train_ds = TensorDataset(X_train, y_train)
        val_ds = TensorDataset(X_val, y_val)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)

        model_args = {**config['args']}
        if is_neural:
            model_args['input_dim'] = len(active_indices)

        model = MODELS[config['model_type']](**model_args)

        trainer = create_trainer(model, config, device=self.device, is_neural=is_neural)
        _, best_score = trainer.train(train_loader, val_loader, epochs=30, verbose=False)

        return best_score

    def evaluate(self, config):
        scores = []
        for train_idx, val_idx in self.folds_indices:
            score = self._run_fold(train_idx, val_idx, config)
            scores.append(score)
        
        return sum(scores) / len(scores)

class BayesianOptimizer(BaseOptimizer):
    n_startup_trials = 30

    def optimize(self, n_trials=100, tag="optuna"):
        paths = ExperimentPaths(self.driver_name, self.model_type, None,
                               self.time_range, tag=tag)
        storage_path = paths.get("study.db", create=True)

        sampler = optuna.samplers.TPESampler(
            n_startup_trials=self.n_startup_trials,
            n_ei_candidates=128,
            multivariate=True,
            group=True,
            constant_liar=True,
            seed=42
        )

        study = optuna.create_study(
            study_name=f"{self.model_type}_optimization",
            direction="maximize",
            storage=f"sqlite:///{storage_path}",
            load_if_exists=True,
            sampler=sampler,
        )

        def objective(trial):
            config = self._create_trial_config(trial)
            score = self.evaluate(config)
            return score

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        return self._save_results(study, paths)

    def _save_results(self, study, paths):
        best_params_path = paths.get("best_params.json", create=True)
        with open(best_params_path, 'w') as f:
            json.dump(study.best_params, f, indent=2)

        best_config = self._create_trial_config(study.best_trial)
        save_name = 'bo_fs' if self.use_feature_selection else 'bo'

        is_neural = not issubclass(MODELS[best_config['model_type']], RegressionModel)
        if is_neural:
            best_config['args']['input_dim'] = len(best_config['features'])

        save_config(best_config, self.driver_name, self.model_type, save_name)

        print(f"\nBest AUROC: {study.best_value:.4f}")
        print(f"Best params saved to: {best_params_path}")
        print(f"Best config saved to: src/configs/config.yaml under {self.driver_name}/{save_name}")
        return study, save_name

    def open_dashboard(self, port=8081, host="127.0.0.1", open_browser=True):
        storage_path = getattr(self, "_storage_path", None)
        if storage_path is None:
            raise RuntimeError("No storage_path found. Run optimize() first or set self._storage_path.")

        storage_url = f"sqlite:///{storage_path}"
        cmd = [sys.executable, "-m", "optuna_dashboard", storage_url, "--host", host, "--port", str(port)]
        proc = subprocess.Popen(cmd)

        if open_browser:
            webbrowser.open(f"http://{host}:{port}")

        return proc

class ExhaustiveOptimizer(BaseOptimizer):
    @abstractmethod
    def _get_search_space(self):
        raise NotImplementedError

    @abstractmethod
    def _generate_all_configs(self):
        raise NotImplementedError

    @abstractmethod
    def _apply_config_dict(self, config, config_dict):
        raise NotImplementedError

    def optimize(self, tag="exhaustive"):
        all_configs = self._generate_all_configs()
        print(f"Total configurations to evaluate: {len(all_configs)}")

        paths = ExperimentPaths(self.driver_name, self.model_type, None, self.time_range, tag=tag)
        trials_path = paths.get('trials.json', create=True)

        trials_data = {
            'trials': [],
            'best_trial': None,
            'total_configs': len(all_configs)
        }

        best_config = None
        best_score = 0.0
        best_trial_id = -1

        for i, config_dict in enumerate(all_configs):
            config = deepcopy(self.base_config)
            self._apply_config_dict(config, config_dict)

            score = self.evaluate(config)

            trial_data = {
                'trial_id': i + 1,
                'score': score,
                'config': config
            }
            trials_data['trials'].append(trial_data)

            if score > best_score:
                best_score = score
                best_config = config
                best_trial_id = i + 1
                trials_data['best_trial'] = {
                    'trial_id': best_trial_id,
                    'score': best_score,
                    'config': best_config
                }

            with open(trials_path, 'w') as f:
                json.dump(trials_data, f, indent=2)

            print(f"[{i+1}/{len(all_configs)}] Score: {score:.4f} | Best: {best_score:.4f} (Trial {best_trial_id})")

        new_model_name = 'exhaust_fs' if self.use_feature_selection else 'exhaust'
        save_config(best_config, self.driver_name, self.model_type, new_model_name)

        print(f"\nBest AUROC: {best_score:.4f}")
        print(f"Trials saved to: {trials_path}")
        print(f"Best config saved to driver YAML under {self.model_type}/{new_model_name}")

        return None, new_model_name
