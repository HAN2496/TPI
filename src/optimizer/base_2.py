import sys
import json
import optuna
import subprocess
import webbrowser
from abc import ABC, abstractmethod
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader

from src.configs.utils import load_config
from src.configs.registries import MODELS
from src.model.base import RegressionModel
from src.utils import prepare_training_data_kfold, ExperimentPaths
from src.utils.utils import save_config
from model.trainer import BaseTrainer, RegressionTrainer

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
                 fold_idx, use_feature_selection=False, device="cpu", verbose=1):
        self.driver_name = driver_name
        self.model_type = model_type
        self.time_range = time_range
        self.downsample = downsample
        self.n_splits = n_splits
        self.test_ratio = test_ratio
        self.fold_idx = fold_idx
        self.use_feature_selection = use_feature_selection
        self.device = device
        self.verbose = verbose
        self.base_config = load_config(driver_name, model_type, 'base')
        self.n_features = len(self.base_config['features'])

    def _create_trial_config(self, trial):
        config = deepcopy(self.base_config)

        self._suggest_model_params(trial, config)

        if self.use_feature_selection:
            base_features = config['features']
            mask = [trial.suggest_categorical(f"use_feat_{i}", [0, 1]) for i in range(len(base_features))]
            if sum(mask) == 0:
                mask[0] = 1
            selected_features = [base_features[i] for i, v in enumerate(mask) if v == 1]
            config['features'] = selected_features

        return config

    @abstractmethod
    def _suggest_model_params(self, trial, config):
        raise NotImplementedError

    def _train_and_evaluate(self, config, trial):
        load_config = self.base_config if self.use_feature_selection else config
        batch_size = config['trainer']['batch_size']
        input_dim = len(config['features'])
        is_regression = issubclass(MODELS[config['model_type']], RegressionModel)

        feature_indices = None
        if self.use_feature_selection:
            base_features = self.base_config['features']
            feature_indices = [base_features.index(f) for f in config['features']]

        _, fold_gen = prepare_training_data_kfold(
            self.driver_name, load_config, self.time_range, self.downsample, n_splits=self.n_splits, test_ratio=self.test_ratio
        )

        for current_fold_idx, train_loader, val_loader in fold_gen:
            if current_fold_idx != self.fold_idx:
                continue

            if feature_indices:
                train_ds = FeatureSelectionDataset(train_loader.dataset, feature_indices)
                val_ds = FeatureSelectionDataset(val_loader.dataset, feature_indices)
                train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
                val_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False, pin_memory=True)

            model_args = {**config['args']}

            if not is_regression:
                model_args['input_dim'] = input_dim

            model = MODELS[config['model_type']](**model_args)

            if is_regression:
                trainer = RegressionTrainer(model, config['trainer'])
                _, best_auroc, _ = trainer.train(train_loader, val_loader, verbose=False)
            else:
                model.to(self.device)
                trainer = BaseTrainer(model, config['trainer'], device=self.device)
                trainer.train(train_loader, val_loader, epochs=30, verbose=False)
                best_auroc = trainer.best_auroc

            return best_auroc

class BayesianOptimizer(BaseOptimizer):
    n_startup_trials = 30

    def optimize(self, n_trials=100, tag="optuna"):
        fold_tag = f"{tag}/fold_{self.fold_idx}"
        paths = ExperimentPaths(self.driver_name, self.model_type, None,
                               self.time_range, tag=fold_tag)

        storage_path = paths.get("study.db", create=True)
        direction = "maximize"

        sampler = optuna.samplers.TPESampler(
            n_startup_trials=self.n_startup_trials,
            n_ei_candidates=128,
            multivariate=True,
            group=True,
            constant_liar=True,
            seed=42 + self.fold_idx
        )

        study = optuna.create_study(
            study_name=f"{self.model_type}_optimization_fold_{self.fold_idx}",
            direction=direction,
            storage=f"sqlite:///{storage_path}",
            load_if_exists=True,
            sampler=sampler,
        )

        def objective(trial):
            config = self._create_trial_config(trial)
            score = self._train_and_evaluate(config, trial)
            return score

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params_path = paths.get("best_params.json", create=True)
        with open(best_params_path, 'w') as f:
            json.dump(study.best_params, f, indent=2)

        best_config = self._create_trial_config(study.best_trial)

        print(f"\nFold {self.fold_idx} - Best AUROC: {study.best_value:.4f}")
        print(f"Best params saved to: {best_params_path}")

        return study, best_config

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
