import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ..experiment import BaseExperiment
from .pipeline import TSFreshPipeline
from ...data.splits import _load_dataset_sequences
from ...evaluation import evaluate_predictions, save_metrics_txt


class TSFreshExperiment(BaseExperiment):

    def build(self):
        cfg = self.cfg
        self._config_dict = {"features": cfg.features}
        self._smooth_kw = {"smooth": cfg.smooth, "smooth_cutoff": cfg.smooth_cutoff, "smooth_order": cfg.smooth_order}
        self._log("[1] Loading train drivers...")
        self.X_train_all = np.concatenate([
            _load_dataset_sequences(d, cfg.time_range, cfg.downsample, self._config_dict, **self._smooth_kw)[0]
            for d in cfg.train_driver_names
        ], axis=0)
        self.norm_mean = self.norm_std = None
        if cfg.normalize:
            self.norm_mean = self.X_train_all.mean(axis=(0, 1), keepdims=True)
            self.norm_std  = self.X_train_all.std(axis=(0, 1), keepdims=True) + 1e-6
            self.X_train_all = (self.X_train_all - self.norm_mean) / self.norm_std
        self.pipeline = TSFreshPipeline(cfg.features, n_components=cfg.n_components,
                                        extraction_settings=cfg.extraction_settings, verbose=cfg.verbose)

    def train(self, out_dir: Path) -> dict:
        self._log("[2] Fitting TSFresh pipeline...")
        self.pipeline.fit(self.X_train_all)
        self.pipeline.save(out_dir / "pipeline.joblib")
        return {}

    def evaluate(self, out_dir: Path) -> dict:
        cfg = self.cfg
        self._log(f"[3] Evaluating on {cfg.test_driver_name} ({cfg.k_folds}-fold CV)...")
        X_test, y_test = _load_dataset_sequences(
            cfg.test_driver_name, cfg.time_range, cfg.downsample, self._config_dict, **self._smooth_kw)
        y_test = y_test.astype(int)
        if self.norm_mean is not None:
            X_test = (X_test - self.norm_mean) / self.norm_std
        Z_test = self.pipeline.transform(X_test)

        skf = StratifiedKFold(n_splits=cfg.k_folds, shuffle=True, random_state=cfg.seed)
        oof_probs = np.zeros(len(y_test))
        for fold, (train_idx, val_idx) in enumerate(skf.split(Z_test, y_test)):
            clf = LogisticRegression(C=cfg.C, class_weight="balanced", max_iter=1000, random_state=cfg.seed)
            clf.fit(Z_test[train_idx], y_test[train_idx])
            oof_probs[val_idx] = clf.predict_proba(Z_test[val_idx])[:, 1]
            self._log(f"  Fold {fold+1}: AUROC={roc_auc_score(y_test[val_idx], oof_probs[val_idx]):.4f}")

        m = evaluate_predictions(y_test, oof_probs, out_dir / "plots", cfg.test_driver_name,
                                  title=f"TSFresh - {cfg.test_driver_name} (OOF)")
        save_metrics_txt({f"test/{cfg.test_driver_name}": m}, out_dir / "metrics.txt")
        self._log(f"  OOF AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}  Brier={m['brier']:.4f}")
        return {f"test/{cfg.test_driver_name}": m}

    def save(self, out_dir: Path) -> None:
        pass  # train()에서 pipeline.joblib 저장

    def load(self, out_dir: Path) -> None:
        self.pipeline = TSFreshPipeline.load(out_dir / "pipeline.joblib")

    def make_summary(self, train_metrics: dict, eval_metrics: dict) -> dict:
        s = super().make_summary(train_metrics, eval_metrics)
        s["test_driver_name"] = self.cfg.test_driver_name
        return s
