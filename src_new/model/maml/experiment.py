import numpy as np
import torch
from pathlib import Path

from ..experiment import BaseExperiment
from .rm import MetaCNNModel
from .trainer import MAMLTrainer
from ...data.splits import load_sequences
from ...evaluation import evaluate_predictions, save_metrics_txt


class MAMLExperiment(BaseExperiment):

    def build(self):
        cfg = self.cfg
        self.per_user_items = {}
        item_series_list = []
        item_offset = 0

        for uid, uname in enumerate(cfg.train_driver_names):
            X, y = load_sequences(uname, cfg.features, cfg.time_range, cfg.downsample,
                                   smooth=cfg.smooth, smooth_cutoff=cfg.smooth_cutoff, smooth_order=cfg.smooth_order)
            n = X.shape[0]
            item_ids = np.arange(item_offset, item_offset + n, dtype=np.int64)
            self.per_user_items[uid] = (item_ids, y.astype(np.int64))
            item_series_list.append(X.astype(np.float32))
            item_offset += n
            if cfg.verbose > 0:
                print(f"  - {uname}: items={n}, pos={int(y.sum())}, neg={int((1-y).sum())}")

        self.item_series = np.concatenate(item_series_list, axis=0)
        self.norm_mean = self.norm_std = None
        if cfg.normalize:
            self.norm_mean = self.item_series.mean(axis=(0, 1), keepdims=True)
            self.norm_std  = self.item_series.std(axis=(0, 1), keepdims=True) + 1e-6
            self.item_series = (self.item_series - self.norm_mean) / self.norm_std
        obs_dim = self.item_series.shape[2]
        self.model = MetaCNNModel(obs_dim=obs_dim, hidden_dim=cfg.hidden_dim)
        self.trainer = MAMLTrainer(self.model, {
            'device': cfg.device, 'outer_lr': cfg.outer_lr, 'inner_lr': cfg.inner_lr,
            'inner_steps': cfg.inner_steps, 'n_support': cfg.n_support, 'n_query': cfg.n_query,
            'n_tasks_per_epoch': cfg.n_tasks_per_epoch, 'meta_epochs': cfg.meta_epochs,
        })

    def train(self, out_dir: Path) -> dict:
        self.trainer.log_dir = out_dir
        print("\n[1] Starting MAML Training...")
        self.trainer.train(self.item_series, self.per_user_items,
                           len(self.cfg.train_driver_names), verbose=self.cfg.verbose)
        return {}

    def evaluate(self, out_dir: Path) -> dict:
        cfg = self.cfg
        print(f"\n[2] Evaluating on Test Driver: {cfg.test_driver_name}")
        X_test, y_test = load_sequences(cfg.test_driver_name, cfg.features, cfg.time_range, cfg.downsample,
                                         smooth=cfg.smooth, smooth_cutoff=cfg.smooth_cutoff, smooth_order=cfg.smooth_order)
        X_test = X_test.astype(np.float32)
        if self.norm_mean is not None:
            X_test = (X_test - self.norm_mean) / self.norm_std
        y_test = y_test.astype(np.int64)

        best_auroc, final_probs, que_y, _ = self.trainer.adapt_and_evaluate(X_test, y_test, verbose=cfg.verbose)

        all_metrics = {}
        if final_probs is not None:
            m = evaluate_predictions(que_y, final_probs, out_dir / "plots", cfg.test_driver_name,
                                      title=f"MAML - {cfg.test_driver_name} (test)")
            all_metrics[f"test/{cfg.test_driver_name}"] = m
            print(f"  Test AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}  Brier={m['brier']:.4f}")

        print("\n[3] Evaluating training drivers (base model)...")
        device = torch.device(cfg.device)
        self.model.eval()
        for uid, uname in enumerate(cfg.train_driver_names):
            item_ids, labels = self.per_user_items[uid]
            X_u = self.item_series[item_ids]
            with torch.no_grad():
                u_probs = torch.sigmoid(
                    self.model(torch.tensor(X_u, dtype=torch.float32).to(device))
                ).cpu().numpy()
            m = evaluate_predictions(labels, u_probs, out_dir / "plots" / "train", uname,
                                      title=f"MAML - {uname} (train)")
            all_metrics[f"train/{uname}"] = m
            print(f"  {uname}: AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}  Brier={m['brier']:.4f}")

        save_metrics_txt(all_metrics, out_dir / "metrics.txt")
        return all_metrics

    def save(self, out_dir: Path) -> None:
        pass  # MAMLTrainer가 best_maml.pt 저장

    def load(self, out_dir: Path) -> None:
        self.model.load_state_dict(
            torch.load(out_dir / "best_maml.pt", map_location=self.cfg.device, weights_only=True))

    def make_summary(self, train_metrics: dict, eval_metrics: dict) -> dict:
        s = super().make_summary(train_metrics, eval_metrics)
        s["test_driver"] = self.cfg.test_driver_name
        return s
