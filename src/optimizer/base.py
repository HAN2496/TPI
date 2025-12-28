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
from src.model import OnlineCombination
from src.utils import prepare_training_data_kfold, ExperimentPaths
from src.utils.utils import save_config
from src.utils.trainer import BaseTrainer, CombinationTrainer

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
    def __init__(self, driver_name, model_type, time_range, downsample, n_splits,
                 use_feature_selection=False, device="cpu", verbose=1):
        self.driver_name = driver_name
        self.model_type = model_type
        self.time_range = time_range
        self.downsample = downsample
        self.n_splits = n_splits
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
        fold_scores = []
        # # OOF AUROC 최대화 방식 (현재는 fold별 AUROC 평균 사용)
        # fold_predictions = []  # (y_true, y_probs) 저장

        load_config = self.base_config if self.use_feature_selection else config
        batch_size = config['trainer']['batch_size']
        input_dim = len(config['features'])
        is_combination = MODELS[config['model_type']] is OnlineCombination

        feature_indices = None
        if self.use_feature_selection:
            base_features = self.base_config['features']
            feature_indices = [base_features.index(f) for f in config['features']]

        for fold_idx, train_loader, val_loader in prepare_training_data_kfold(
            self.driver_name, load_config, self.time_range, self.downsample, n_splits=self.n_splits
        ):
            if feature_indices:
                train_loader = DataLoader(
                    FeatureSelectionDataset(train_loader.dataset, feature_indices),
                    batch_size=batch_size, shuffle=True, pin_memory=True
                )
                val_loader = DataLoader(
                    FeatureSelectionDataset(val_loader.dataset, feature_indices),
                    batch_size=len(val_loader.dataset), shuffle=False, pin_memory=True
                )

            model_args = {**config['args'], 'input_dim': input_dim}
            model = MODELS[config['model_type']](**model_args)

            if is_combination:
                trainer = CombinationTrainer(model, config['trainer'])
                _, best_auroc, _ = trainer.train(train_loader, val_loader, verbose=False)
            else:
                model.to(self.device)
                trainer = BaseTrainer(model, config['trainer'], device=self.device)
                trainer.train(train_loader, val_loader, epochs=30, verbose=False)
                best_auroc = trainer.best_auroc

            fold_scores.append(best_auroc)

            # # OOF 방식: 각 fold의 예측 수집
            # import torch
            # import numpy as np
            # model.eval()
            # with torch.no_grad():
            #     X_val, y_val = next(iter(val_loader))
            #     if not is_combination:
            #         X_val = X_val.to(self.device)
            #     y_probs = model.predict_probability(X_val)
            #     if torch.is_tensor(y_probs):
            #         y_probs = y_probs.cpu().numpy()
            #     if torch.is_tensor(y_val):
            #         y_val = y_val.cpu().numpy()
            #     fold_predictions.append((y_val, y_probs))

        return sum(fold_scores) / len(fold_scores)

        # # OOF 방식: 모든 fold 예측을 합쳐서 AUROC 계산
        # from sklearn.metrics import roc_auc_score
        # y_true_all = np.concatenate([y for y, _ in fold_predictions])
        # y_probs_all = np.concatenate([p for _, p in fold_predictions])
        # return roc_auc_score(y_true_all, y_probs_all)

class BayesianOptimizer(BaseOptimizer):
    n_startup_trials = 30

    def optimize(self, n_trials=100, tag="optuna"):
        paths = ExperimentPaths(self.driver_name, self.model_type, None,
                               self.time_range, tag=tag)

        storage_path = paths.get("study.db", create=True)
        direction = "maximize"

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

        new_model_name = 'bo_fs' if self.use_feature_selection else 'bo'

        save_config(best_config, self.driver_name, self.model_type, new_model_name)

        print(f"\nBest AUROC: {study.best_value:.4f}")
        print(f"Best params saved to: {best_params_path}")
        print(f"Best config saved to: src/configs/config.yaml under {self.driver_name}/{new_model_name}")

        return study, new_model_name

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

            score = self._train_and_evaluate(config, trial=None)

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
