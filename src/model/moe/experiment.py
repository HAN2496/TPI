import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from ..experiment import BaseExperiment
from .rm import MoECNNModel
from .trainer import MoETrainer
from ...data.splits import load_sequences
from ...evaluation import evaluate_predictions, save_metrics_txt, plot_sequential_auroc, plot_training_curves

class MoEExperiment(BaseExperiment):

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
            self._log(f"  - {uname}: items={n}, pos={int(y.sum())}, neg={int((1-y).sum())}")

        self.item_series = np.concatenate(item_series_list, axis=0)
        self.norm_mean = self.norm_std = None
        if cfg.normalize:
            self.norm_mean = self.item_series.mean(axis=(0, 1), keepdims=True)
            self.norm_std  = self.item_series.std(axis=(0, 1), keepdims=True) + 1e-6
            self.item_series = (self.item_series - self.norm_mean) / self.norm_std

        val_x_list, val_y_list, val_uid_list = [], [], []
        self.per_user_items_train = {}
        for uid, (item_ids, labels) in self.per_user_items.items():
            tr_idx, va_idx = train_test_split(np.arange(len(item_ids)), test_size=cfg.val_size, random_state=cfg.seed)
            self.per_user_items_train[uid] = (item_ids[tr_idx], labels[tr_idx])
            val_x_list.append(self.item_series[item_ids[va_idx]])
            val_y_list.append(labels[va_idx])
            val_uid_list.append(np.full(len(va_idx), uid, dtype=np.int64))
        self.val_x = np.concatenate(val_x_list)
        self.val_y = np.concatenate(val_y_list)
        self.val_uids = np.concatenate(val_uid_list)

        obs_dim = self.item_series.shape[2]
        self.model = MoECNNModel(
            obs_dim=obs_dim, 
            n_train_users=len(cfg.train_driver_names), 
            user_dim=cfg.user_dim, 
            hidden_dim=cfg.hidden_dim, 
            num_experts=cfg.num_experts
        )
        self.trainer = MoETrainer(self.model, {
            'device': cfg.device, 'lr': cfg.lr, 'weight_decay': cfg.weight_decay,
            'epochs': cfg.epochs, 'batch_size': cfg.batch_size,
            'adapt_lr': cfg.adapt_lr, 'adapt_steps': cfg.adapt_steps
        })

    def train(self, out_dir: Path) -> dict:
        self.trainer.log_dir = out_dir
        self._log("\n[1] Starting MoE Training...")
        metrics = self.trainer.train(self.item_series, self.per_user_items_train,
                                     val_x=self.val_x, val_y=self.val_y, val_uids=self.val_uids,
                                     verbose=self.cfg.verbose)
        plot_training_curves(dict(metrics), out_dir / "plots" / "training_curves.png", title="MoE Training")
        return {}

    def evaluate(self, out_dir: Path) -> dict:
        cfg = self.cfg
        self._log(f"\n[2] Evaluating on Test Driver: {cfg.test_driver_name}")
        X_test, y_test = load_sequences(cfg.test_driver_name, cfg.features, cfg.time_range, cfg.downsample,
                                         smooth=cfg.smooth, smooth_cutoff=cfg.smooth_cutoff, smooth_order=cfg.smooth_order)
        X_test = X_test.astype(np.float32)
        if self.norm_mean is not None:
            X_test = (X_test - self.norm_mean) / self.norm_std
        y_test = y_test.astype(np.int64)

        ctx_sizes, test_aurocs, final_probs, holdout_y, snapshots = self.trainer.adapt_sequential(X_test, y_test)

        all_metrics = {}
        if final_probs is not None:
            plot_sequential_auroc(ctx_sizes, test_aurocs, out_dir / "plots", cfg.test_driver_name)

            for pct, probs in snapshots.items():
                snap_dir = out_dir / "plots" / "snapshots" / f"context_{pct}pct"
                snap_dir.mkdir(parents=True, exist_ok=True)
                evaluate_predictions(holdout_y, probs, snap_dir, "metrics",
                                     title=f"MoE - {cfg.test_driver_name} (Context {pct}%)")

            m = evaluate_predictions(holdout_y, final_probs, out_dir / "plots", cfg.test_driver_name,
                                      title=f"MoE - {cfg.test_driver_name} (test)")
            all_metrics[f"test/{cfg.test_driver_name}"] = m
            self._log(f"  Test AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}  Brier={m['brier']:.4f}")

        self._log("\n[3] Evaluating training drivers (base model)...")
        device = torch.device(cfg.device)
        self.model.eval()
        for uid, uname in enumerate(cfg.train_driver_names):
            item_ids, labels = self.per_user_items[uid]
            X_u = torch.tensor(self.item_series[item_ids], dtype=torch.float32).to(device)
            uids_t = torch.full((len(item_ids),), uid, dtype=torch.long).to(device)
            with torch.no_grad():
                u_probs = torch.sigmoid(self.model(X_u, uids=uids_t)).cpu().numpy()
            m = evaluate_predictions(labels, u_probs, out_dir / "plots" / "train", uname,
                                      title=f"MoE - {uname} (train)")
            all_metrics[f"train/{uname}"] = m
            self._log(f"  {uname}: AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}  Brier={m['brier']:.4f}")

        save_metrics_txt(all_metrics, out_dir / "metrics.txt")
        return all_metrics

    def save(self, out_dir: Path) -> None:
        pass  # Trainer 안에서 저장함

    def load(self, out_dir: Path) -> None:
        self.model.load_state_dict(
            torch.load(out_dir / "best_moe.pt", map_location=self.cfg.device, weights_only=True))

    def make_summary(self, train_metrics: dict, eval_metrics: dict) -> dict:
        s = super().make_summary(train_metrics, eval_metrics)
        s["test_driver"] = self.cfg.test_driver_name
        return s