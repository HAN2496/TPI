import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from ..experiment import BaseExperiment
from .episode_wise import OfflineLSTM, OfflineRegression
from .step_wise import OnlineMLP, OnlineLSTM, OnlineAttention, OnlineNNRegression, OnlineRegression
from .trainer import BaseTrainer, RegressionTrainer
from ...data.splits import load_sequences, make_loaders
from ...evaluation import evaluate_predictions, save_metrics_txt, plot_training_curves


class IndependentExperiment(BaseExperiment):

    def build(self):
        cfg = self.cfg
        self.train_loader, self.val_loader = make_loaders(
            cfg.driver_name, {"features": cfg.features, "test_size": cfg.test_size, "trainer": cfg.trainer},
            cfg.time_range, cfg.downsample, normalize=cfg.normalize)
        xb, _ = next(iter(self.train_loader))
        self.model = self._build_model(xb.shape[-1])
        self.is_neural = not hasattr(self.model, "fit")

    def _build_model(self, input_dim):
        cfg = self.cfg
        if cfg.model_type == "online_mlp":
            return OnlineMLP(input_dim, cfg.hidden_dims, cfg.dropout_rates, cfg.act_name, reduce=cfg.reduce)
        if cfg.model_type == "online_lstm":
            return OnlineLSTM(input_dim, 64, 1, 0.0, cfg.hidden_dims, cfg.dropout_rates, cfg.act_name, reduce=cfg.reduce)
        if cfg.model_type == "online_attention":
            return OnlineAttention(input_dim, 64, 4, 0.1, 128, 0.1, cfg.hidden_dims, cfg.dropout_rates, cfg.act_name, reduce=cfg.reduce)
        if cfg.model_type == "offline_lstm":
            return OfflineLSTM(input_dim, 64, 1, 0.0, False, cfg.hidden_dims, cfg.dropout_rates, cfg.act_name)
        raise ValueError(f"Unknown model_type: {cfg.model_type}")

    def train(self, out_dir: Path) -> dict:
        cfg = self.cfg
        if self.is_neural:
            self._trainer = BaseTrainer(self.model, cfg.trainer,
                                        best_model_path=str(out_dir / "best.pt"), device=cfg.device)
            history, best_auroc = self._trainer.train(self.train_loader, self.val_loader, verbose=cfg.verbose)
            plot_training_curves(history, out_dir / "training_curves.png")
        else:
            self._trainer = RegressionTrainer(self.model)
            _, best_auroc = self._trainer.train(self.train_loader, self.val_loader)
        return {"val_auroc": float(best_auroc)}

    def evaluate(self, out_dir: Path) -> dict:
        if self.is_neural:
            _, best_auroc = self._trainer.evaluate(self.val_loader)
        else:
            _, best_auroc = self._trainer.train(self.train_loader, self.val_loader)
        m = {"val_auroc": float(best_auroc)}
        save_metrics_txt({f"val/{self.cfg.driver_name}": m}, out_dir / "metrics.txt")
        self._log(f"  {self.cfg.driver_name}: val AUROC={best_auroc:.4f}")
        return {f"val/{self.cfg.driver_name}": m}

    def save(self, out_dir: Path) -> None:
        pass  # BaseTrainer가 best.pt 저장

    def load(self, out_dir: Path) -> None:
        self.model.load_state_dict(
            torch.load(out_dir / "best.pt", map_location=self.cfg.device, weights_only=True))
        self._trainer = BaseTrainer(self.model, self.cfg.trainer, device=self.cfg.device)

    def make_summary(self, train_metrics: dict, eval_metrics: dict) -> dict:
        s = super().make_summary(train_metrics, eval_metrics)
        s["driver_name"] = self.cfg.driver_name
        s["model_type"] = self.cfg.model_type
        return s


class BaselineExperiment(BaseExperiment):
    """All train drivers combined, no per-user adaptation."""

    def build(self):
        cfg = self.cfg
        X_parts, y_parts = [], []
        for name in cfg.train_driver_names:
            X, y = load_sequences(name, cfg.features, cfg.time_range, cfg.downsample,
                                   smooth=cfg.smooth, smooth_cutoff=cfg.smooth_cutoff, smooth_order=cfg.smooth_order)
            X_parts.append(X.astype(np.float32))
            y_parts.append(y.astype(np.float32))
        X_all = np.concatenate(X_parts)
        y_all = np.concatenate(y_parts)
        self.X_test, self.y_test = load_sequences(
            cfg.test_driver, cfg.features, cfg.time_range, cfg.downsample,
            smooth=cfg.smooth, smooth_cutoff=cfg.smooth_cutoff, smooth_order=cfg.smooth_order)

        if cfg.normalize:
            norm_mean = X_all.mean(axis=(0, 1), keepdims=True)
            norm_std  = X_all.std(axis=(0, 1), keepdims=True) + 1e-6
            X_all = (X_all - norm_mean) / norm_std
            self.X_test = (self.X_test - norm_mean) / norm_std

        X_tr, X_va, y_tr, y_va = train_test_split(
            X_all, y_all, test_size=cfg.val_size, random_state=cfg.seed, stratify=y_all)
        self.train_loader = DataLoader(
            TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
            batch_size=cfg.batch_size, shuffle=True)
        self.val_loader = DataLoader(
            TensorDataset(torch.tensor(X_va), torch.tensor(y_va)),
            batch_size=len(X_va), shuffle=False)

        self.model = self._build_model(X_all.shape[2])

    def _build_model(self, input_dim):
        cfg = self.cfg
        if cfg.model_type == "online_mlp":
            return OnlineMLP(input_dim, cfg.hidden_dims, cfg.dropout_rates, cfg.act_name, reduce=cfg.reduce)
        if cfg.model_type == "online_lstm":
            return OnlineLSTM(input_dim, cfg.lstm_hidden, cfg.lstm_layers, cfg.lstm_dropout,
                              cfg.hidden_dims, cfg.dropout_rates, cfg.act_name, reduce=cfg.reduce)
        if cfg.model_type == "online_attention":
            return OnlineAttention(input_dim, cfg.attn_dim, cfg.attn_heads, cfg.attn_dropout,
                                   cfg.ffn_dim, cfg.ffn_dropout,
                                   cfg.hidden_dims, cfg.dropout_rates, cfg.act_name, reduce=cfg.reduce)
        if cfg.model_type == "offline_lstm":
            return OfflineLSTM(input_dim, cfg.lstm_hidden, cfg.lstm_layers, cfg.lstm_dropout,
                               False, cfg.hidden_dims, cfg.dropout_rates, cfg.act_name)
        raise ValueError(f"Unknown model_type: {cfg.model_type}")

    def train(self, out_dir: Path) -> dict:
        cfg = self.cfg
        trainer_cfg = {"learning_rate": cfg.lr, "weight_decay": cfg.weight_decay, "epochs": cfg.epochs}
        self._trainer = BaseTrainer(self.model, trainer_cfg,
                                    best_model_path=str(out_dir / "best.pt"), device=cfg.device)
        history, best_auroc = self._trainer.train(self.train_loader, self.val_loader, verbose=cfg.verbose)
        plot_training_curves(history, out_dir / "training_curves.png")
        return {"val_auroc": float(best_auroc)}

    def evaluate(self, out_dir: Path) -> dict:
        cfg = self.cfg
        device = torch.device(cfg.device)
        self.model.eval()
        all_metrics = {}

        self._log(f"\n[2] Test-time evaluation on {cfg.test_driver}...")
        X_t = torch.tensor(self.X_test, dtype=torch.float32).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(self.model(X_t)).cpu().numpy()
        m = evaluate_predictions(self.y_test, probs, out_dir / "plots", cfg.test_driver,
                                  title=f"Baseline - {cfg.test_driver} (test)")
        all_metrics[f"test/{cfg.test_driver}"] = m
        self._log(f"  Test AUROC: {m['auroc']:.4f}  AUPRC: {m['auprc']:.4f}  Brier: {m['brier']:.4f}")

        self._log("\n[3] Evaluating per train driver...")
        for name in cfg.train_driver_names:
            X_u, y_u = load_sequences(name, cfg.features, cfg.time_range, cfg.downsample,
                                       smooth=cfg.smooth, smooth_cutoff=cfg.smooth_cutoff, smooth_order=cfg.smooth_order)
            X_ut = torch.tensor(X_u, dtype=torch.float32).to(device)
            with torch.no_grad():
                p_u = torch.sigmoid(self.model(X_ut)).cpu().numpy()
            m_u = evaluate_predictions(y_u, p_u, out_dir / "plots" / "train", name,
                                        title=f"Baseline - {name} (train)")
            all_metrics[f"train/{name}"] = m_u
            self._log(f"  {name}: AUROC={m_u['auroc']:.4f}  AUPRC={m_u['auprc']:.4f}  Brier={m_u['brier']:.4f}")

        save_metrics_txt(all_metrics, out_dir / "metrics.txt")
        return all_metrics

    def save(self, out_dir: Path) -> None:
        pass  # BaseTrainer가 best.pt 저장

    def load(self, out_dir: Path) -> None:
        self.model.load_state_dict(
            torch.load(out_dir / "best.pt", map_location=self.cfg.device, weights_only=True))
        trainer_cfg = {"learning_rate": self.cfg.lr, "weight_decay": self.cfg.weight_decay,
                       "epochs": self.cfg.epochs}
        self._trainer = BaseTrainer(self.model, trainer_cfg, device=self.cfg.device)

    def make_summary(self, train_metrics: dict, eval_metrics: dict) -> dict:
        s = super().make_summary(train_metrics, eval_metrics)
        s["test_driver"] = self.cfg.test_driver
        s["model_type"] = self.cfg.model_type
        return s
