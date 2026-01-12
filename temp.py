import sys
import json
import optuna
import subprocess
import webbrowser
import numpy as np
import torch
from abc import ABC, abstractmethod
from copy import deepcopy
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.configs.utils import load_config
from src.configs.registries import MODELS
from src.model.base import RegressionModel
from src.utils import ExperimentPaths, save_config
from model.trainer import BaseTrainer, RegressionTrainer

# 기존의 prepare_training_data_kfold 대신 내부 메서드 혹은 _load_dataset_sequences를 직접 사용한다고 가정
# (여기서는 코드 문맥상 필요한 데이터 로딩 부분만 간소화하여 클래스 내부에 통합했습니다)
from src.utils import _load_dataset_sequences 

class BaseOptimizer(ABC):
    def __init__(self, driver_name, model_type, time_range, downsample, n_splits, test_ratio,
                 use_feature_selection=False, device="cpu", verbose=1):
        self.driver_name = driver_name
        self.model_type = model_type
        self.time_range = time_range
        self.device = device
        self.verbose = verbose
        
        # 설정 로드
        self.base_config = load_config(driver_name, model_type, 'base')
        self.all_features = self.base_config['features']
        self.n_features = len(self.all_features)
        
        # Feature Selection 설정
        self.use_feature_selection = use_feature_selection
        
        # 데이터 미리 준비 (핵심 최적화: Trial마다 로드하지 않음)
        self._setup_data(downsample, n_splits, test_ratio)

    def _setup_data(self, downsample, n_splits, test_ratio):
        """데이터를 메모리에 로드하고 K-Fold 인덱스를 미리 계산합니다."""
        if self.verbose > 0:
            print("Loading and splitting data for optimization...")
            
        # 1. 원본 데이터 로드 (외부 함수 사용)
        X, y = _load_dataset_sequences(self.driver_name, self.time_range, downsample, self.base_config)
        
        # 2. Train/Test 분리
        X_trainval, _, y_trainval, _ = train_test_split(
            X, y, test_size=test_ratio, random_state=42, stratify=y
        )

        # 3. Tensor 변환 (GPU 사용 시 미리 옮길 수도 있지만, 메모리 이슈 고려하여 CPU 유지 추천)
        self.X_data = torch.as_tensor(X_trainval, dtype=torch.float32)
        self.y_data = torch.as_tensor(y_trainval, dtype=torch.float32)

        # 4. K-Fold 인덱스 미리 생성 (Fixed Folds)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        y_np = np.asarray(y_trainval)
        
        # [(train_idx, val_idx), (train_idx, val_idx), ...] 형태로 저장
        self.folds_indices = list(skf.split(np.zeros(len(y_np)), y_np))

    @abstractmethod
    def _suggest_model_params(self, trial, config):
        """모델 하이퍼파라미터 제안 로직 (자식 클래스 구현)"""
        pass

    def _suggest_features(self, trial):
        """Feature Selection 마스크 생성"""
        if not self.use_feature_selection:
            return list(range(self.n_features))

        # Categorical로 0/1 선택 (Bit mask)
        mask = [trial.suggest_categorical(f"use_feat_{i}", [0, 1]) for i in range(self.n_features)]
        
        # 모든 Feature가 꺼지는 경우 방지
        if sum(mask) == 0:
            mask[0] = 1
            
        selected_indices = [i for i, v in enumerate(mask) if v == 1]
        return selected_indices

    def _create_trial_config(self, trial):
        config = deepcopy(self.base_config)
        self._suggest_model_params(trial, config)
        
        # Feature Selection 적용
        if self.use_feature_selection:
            selected_indices = self._suggest_features(trial)
            config['features'] = [self.all_features[i] for i in selected_indices]
            config['active_feature_indices'] = selected_indices # 내부 로직용 저장
        else:
            config['active_feature_indices'] = list(range(self.n_features))
            
        return config

    def _run_fold(self, train_idx, val_idx, config):
        """단일 Fold 학습 및 평가"""
        active_indices = config['active_feature_indices']
        
        # 1. 텐서 슬라이싱 (여기서 필요한 Feature만 딱 잘라냅니다 - 속도 최적화)
        # X_data: (N, Seq, Feat) -> 슬라이싱 -> (N, Seq, Selected_Feat)
        X_train = self.X_data[train_idx][:, :, active_indices]
        y_train = self.y_data[train_idx]
        
        X_val = self.X_data[val_idx][:, :, active_indices]
        y_val = self.y_data[val_idx]

        # 2. 데이터로더 생성
        is_regression = issubclass(MODELS[config['model_type']], RegressionModel)
        batch_size = config['trainer']['batch_size'] if not is_regression else len(X_train)
        
        train_ds = TensorDataset(X_train, y_train)
        val_ds = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)

        # 3. 모델 초기화
        model_args = {**config['args']}
        if not is_regression:
            model_args['input_dim'] = len(active_indices) # 동적으로 입력 차원 변경

        model = MODELS[config['model_type']](**model_args)

        # 4. 학습
        if is_regression:
            trainer = RegressionTrainer(model)
            _, best_score, _ = trainer.train(train_loader, val_loader, verbose=False)
        else:
            model.to(self.device)
            trainer = BaseTrainer(model, config, device=self.device)
            trainer.train(train_loader, val_loader, epochs=30, verbose=False)
            best_score = trainer.best_auroc
            
        return best_score

    def evaluate(self, config):
        """전체 Fold에 대해 평가 수행"""
        scores = []
        for train_idx, val_idx in self.folds_indices:
            score = self._run_fold(train_idx, val_idx, config)
            scores.append(score)
        
        return sum(scores) / len(scores)


class BayesianOptimizer(BaseOptimizer):
    n_startup_trials = 30

    def optimize(self, n_trials=100, tag="optuna"):
        paths = ExperimentPaths(self.driver_name, self.model_type, None, self.time_range, tag=tag)
        storage_path = paths.get("study.db", create=True)

        sampler = optuna.samplers.TPESampler(
            n_startup_trials=self.n_startup_trials,
            n_ei_candidates=128,
            multivariate=True,
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
            return self.evaluate(config)

        if self.verbose > 0:
            print(f"Starting Bayesian Optimization: {n_trials} trials")

        study.optimize(objective, n_trials=n_trials, show_progress_bar=(self.verbose > 0))
        return self._save_results(study, paths)

    def _save_results(self, study, paths):
        best_params_path = paths.get("best_params.json", create=True)
        with open(best_params_path, 'w') as f:
            json.dump(study.best_params, f, indent=2)

        # Best Config 재생성 및 저장
        best_config = self._create_trial_config(study.best_trial)
        
        # Feature Selection 여부에 따른 이름 결정
        suffix = 'fs' if self.use_feature_selection else 'bo'
        save_name = f"{suffix}" 

        save_config(best_config, self.driver_name, self.model_type, save_name)

        if self.verbose > 0:
            print(f"\nBest AUROC: {study.best_value:.4f}")
            print(f"Config saved as: {save_name}")
            
        return study, save_name


class ExhaustiveOptimizer(BaseOptimizer):
    @abstractmethod
    def _get_search_space(self):
        """Return list of config dictionaries to try"""
        pass

    def optimize(self, tag="exhaustive"):
        configs = self._get_search_space()
        
        # Feature Selection이 켜져있다면, 각 Config에 대해 Feature 조합도 추가해야 함
        # 하지만 Exhaustive에서 Feature 조합까지 다 돌리는 건 불가능에 가까우므로 
        # 여기서는 하이퍼파라미터만 서치하거나, 로직을 추가해야 함.
        # 일단은 하이퍼파라미터 서치만 수행한다고 가정.
        
        if self.verbose > 0:
            print(f"Starting Exhaustive Search: {len(configs)} configurations")

        best_score = -1.0
        best_config = None
        
        # 결과 저장을 위한 경로 설정
        paths = ExperimentPaths(self.driver_name, self.model_type, None, self.time_range, tag=tag)
        
        for i, config_dict in enumerate(configs):
            # Base Config에 파라미터 덮어씌우기
            config = deepcopy(self.base_config)
            
            # config_dict 병합 (재귀적 업데이트 필요시 유틸 함수 사용 권장)
            for k, v in config_dict.items():
                if isinstance(v, dict) and k in config:
                    config[k].update(v)
                else:
                    config[k] = v
            
            # Feature Selection은 Exhaustive에서는 보통 고정하거나 따로 처리하므로
            # 여기서는 전체 Feature 사용으로 가정 (필요시 수정)
            config['active_feature_indices'] = list(range(self.n_features))

            score = self.evaluate(config)

            if score > best_score:
                best_score = score
                best_config = config
                
            if self.verbose > 0:
                print(f"[{i+1}/{len(configs)}] Score: {score:.4f} (Best: {best_score:.4f})")

        # 저장 로직
        suffix = 'exhaust_fs' if self.use_feature_selection else 'exhaust'
        save_config(best_config, self.driver_name, self.model_type, suffix)
        
        return None, suffix