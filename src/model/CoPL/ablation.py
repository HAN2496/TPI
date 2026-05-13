import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from .rm import (ObsOnlyRewardModel, ObsOnlyCNNRewardModel,
                 ObsOnlyPreferenceTransformerRewardModel,
                 RMEdgeDataset, rm_collate)
from .trainer import CoPLRMTrainer
from .visualization import plot_ablation_bars
from ...evaluation import evaluate_predictions

ABL_ADAPTED    = "Proposed"
ABL_NO_USER    = "No User Emb."
ABL_TRAIN_OBS  = "Pooled"
ABL_TARGET_OBS = "Target Only"

_TAG = {ABL_ADAPTED: "w/ CoPL", ABL_NO_USER: "w/ CoPL",
        ABL_TRAIN_OBS: "w/o CoPL", ABL_TARGET_OBS: "w/o CoPL"}


def xtick_labels(names):
    return [f"{n}\n{_TAG[n]}" for n in names]


def build_obs_only_rm(cfg, obs_dim, device):
    common = dict(obs_dim=obs_dim)
    if cfg.rm_model == "mlp":
        return ObsOnlyRewardModel(**common, hidden=cfg.rm_mlp_hidden).to(device)
    if cfg.rm_model == "cnn":
        return ObsOnlyCNNRewardModel(**common, hidden=cfg.rm_hidden, mlp_hidden=cfg.rm_mlp_hidden,
                                     kernel_size=cfg.rm_kernel_size, layers=cfg.rm_layers).to(device)
    if cfg.rm_model == "preference_transformer":
        return ObsOnlyPreferenceTransformerRewardModel(
            **common, hidden=cfg.rm_hidden, num_heads=cfg.rm_num_heads,
            num_layers=cfg.rm_layers, max_len=cfg.rm_max_len).to(device)
    raise ValueError(f"Obs-only RM not implemented for rm_model={cfg.rm_model!r}")


def predict_no_user_branch(rm, obs):
    if not hasattr(rm, "forward_no_user"):
        return None
    with torch.no_grad():
        return torch.sigmoid(rm.forward_no_user(obs)).cpu().numpy()


def _split_ctx(y_ctx, cfg):
    y_ctx = np.asarray(y_ctx, dtype=np.int64)
    n_ctx = len(y_ctx)
    if n_ctx < 2:
        return np.arange(n_ctx), np.array([], dtype=np.int64), "no_val_too_small"
    val_size = float(cfg.val_size)
    if val_size <= 0.0:
        return np.arange(n_ctx), np.array([], dtype=np.int64), "no_val_disabled"
    n_val = int(np.ceil(n_ctx * min(max(val_size, 0.0), 0.9)))
    n_val = min(max(1, n_val), n_ctx - 1)
    indices = np.arange(n_ctx)
    stratify, split_mode = None, "random"
    classes, counts = np.unique(y_ctx, return_counts=True)
    if len(classes) == 2 and counts.min() >= 2 and n_val >= 2 and (n_ctx - n_val) >= 2:
        stratify, split_mode = y_ctx, "stratified"
    try:
        tr_idx, va_idx = train_test_split(indices, test_size=n_val, random_state=cfg.seed, stratify=stratify)
    except ValueError:
        tr_idx, va_idx = train_test_split(indices, test_size=n_val, random_state=cfg.seed, stratify=None)
        split_mode = "random_fallback"
    return tr_idx.astype(np.int64), va_idx.astype(np.int64), split_mode


def fit_ctx_obs_only_rm(cfg, obs_dim, device, X_ctx, y_ctx, X_eval):
    model = build_obs_only_rm(cfg, obs_dim, device)
    model.train()
    y_ctx = np.asarray(y_ctx, dtype=np.int64)
    tr_idx, va_idx, split_mode = _split_ctx(y_ctx, cfg)
    X_tr, y_tr = X_ctx[tr_idx], y_ctx[tr_idx]
    X_va, y_va = X_ctx[va_idx], y_ctx[va_idx]

    train_loader = DataLoader(
        RMEdgeDataset(np.zeros(len(X_tr), dtype=np.int64), np.arange(len(X_tr)), y_tr, X_tr),
        batch_size=min(cfg.rm_batch_size, max(1, len(X_tr))), shuffle=True, collate_fn=rm_collate)
    val_loader = None
    if len(X_va) > 0:
        val_loader = DataLoader(
            RMEdgeDataset(np.zeros(len(X_va), dtype=np.int64), np.arange(len(X_va)), y_va, X_va),
            batch_size=min(cfg.rm_batch_size, len(X_va)), shuffle=False, collate_fn=rm_collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.rm_lr, weight_decay=cfg.rm_weight_decay)

    pos_weight = None
    if cfg.use_pos_weight:
        pos_cnt = int(np.sum(y_tr))
        neg_cnt = int(len(y_tr) - pos_cnt)
        if pos_cnt > 0 and neg_cnt > 0:
            pos_weight = torch.tensor([neg_cnt / pos_cnt], dtype=torch.float32, device=device)

    epochs = int(cfg.rm_epochs)
    patience = int(cfg.ctx_obs_only_patience)
    last_loss, best_score = float("nan"), -float("inf")
    best_state, best_epoch = None, -1
    best_val_loss, best_val_auc = float("nan"), float("nan")
    bad_epochs, epochs_run = 0, 0
    for epoch in range(max(1, epochs)):
        epochs_run = epoch + 1
        for _, obs_b, y_b in train_loader:
            obs_b = obs_b.to(device); y_b = y_b.float().to(device)
            logits = model(obs_b)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_b, pos_weight=pos_weight)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            last_loss = float(loss.item())
        if val_loader is None:
            continue
        model.eval()
        val_logits, val_labels = [], []
        val_loss_total, n_val_seen = 0.0, 0
        with torch.no_grad():
            for _, obs_b, y_b in val_loader:
                obs_b = obs_b.to(device); y_b_dev = y_b.float().to(device)
                logits = model(obs_b)
                val_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_b_dev)
                val_loss_total += float(val_loss.item()) * len(y_b)
                n_val_seen += len(y_b)
                val_logits.append(logits.cpu()); val_labels.append(y_b)
        model.train()
        val_probs = torch.sigmoid(torch.cat(val_logits)).numpy()
        val_y = torch.cat(val_labels).numpy().astype(np.int64)
        val_loss_mean = val_loss_total / max(1, n_val_seen)
        val_auc = roc_auc_score(val_y, val_probs) if len(np.unique(val_y)) > 1 else float("nan")
        score = val_auc if not np.isnan(val_auc) else -val_loss_mean
        if score > best_score:
            best_score, best_epoch = score, epoch + 1
            best_val_loss, best_val_auc = val_loss_mean, val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if patience > 0 and bad_epochs >= patience:
                break
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    eval_obs = torch.tensor(X_eval, dtype=torch.float32, device=device)
    with torch.no_grad():
        probs = torch.sigmoid(model(eval_obs)).cpu().numpy()
    info = {
        "train_n": int(len(X_tr)), "val_n": int(len(X_va)),
        "split": split_mode, "epochs_run": int(epochs_run),
        "best_epoch": int(best_epoch if best_epoch > 0 else epochs_run),
        "train_loss_last": float(last_loss),
        "val_loss": float(best_val_loss), "val_auroc": float(best_val_auc),
    }
    return probs, info


class Ablation:
    """ablation 전체 책임 (Pooled obs-only fit/load/predict + No User + Target Only + 그림/메트릭)."""

    def __init__(self, experiment):
        self.exp = experiment
        self.cfg = experiment.cfg
        self.device = experiment.device
        self.pooled_rm = None

    def build(self):
        self.pooled_rm = build_obs_only_rm(self.cfg, self.exp.dataset.obs_dim, self.device)

    def fit(self, out_dir, train_loader, val_loader):
        exp, cfg, ds = self.exp, self.cfg, self.exp.dataset
        exp._log(f"\n[2b] Training {ABL_TRAIN_OBS}...")
        trainer = CoPLRMTrainer(self.pooled_rm, {
            'device': cfg.device, 'rm_lr': cfg.rm_lr,
            'rm_weight_decay': cfg.rm_weight_decay, 'rm_lambda_reg': cfg.rm_lambda_reg,
            'rm_epochs': cfg.rm_epochs, 'use_pos_weight': cfg.use_pos_weight,
        }, log_dir=out_dir, checkpoint_name="best_obs_rm.pt")
        best_auc, metrics = trainer.train(train_loader, val_loader, None, ds.tr_y, verbose=cfg.verbose)
        exp._log(f"  {ABL_TRAIN_OBS} Best Val AUC: {best_auc:.4f}")
        return best_auc, metrics

    def load(self, out_dir):
        ckpt = out_dir / "best_obs_rm.pt"
        if ckpt.exists():
            self.pooled_rm.load_state_dict(torch.load(ckpt, map_location=self.device, weights_only=True))
            return
        ds, cfg = self.exp.dataset, self.cfg
        train_loader = DataLoader(RMEdgeDataset(ds.tr_u, ds.tr_i, ds.tr_y, ds.item_series),
                                   batch_size=cfg.rm_batch_size, shuffle=True, collate_fn=rm_collate)
        val_loader = DataLoader(RMEdgeDataset(ds.va_u, ds.va_i, ds.va_y, ds.item_series),
                                 batch_size=cfg.rm_batch_size, shuffle=False, collate_fn=rm_collate)
        self.exp._log(f"\n[Eval-only] {ABL_TRAIN_OBS} checkpoint missing; training baseline...")
        self.fit(out_dir, train_loader, val_loader)

    def _pooled_probs(self, holdout_X):
        if self.pooled_rm is None:
            return None
        self.pooled_rm.eval()
        obs = torch.tensor(holdout_X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return torch.sigmoid(self.pooled_rm(obs)).cpu().numpy()

    def run(self, drivers, plots_dir, all_metrics):
        exp, ds, log = self.exp, self.exp.dataset, self.exp._log
        ablation_dir = plots_dir / "ablation"
        names = [d["name"] for d in drivers]
        ys = [d["holdout_y"] for d in drivers]

        for d in drivers:
            holdout_obs = torch.tensor(d["holdout_X"], dtype=torch.float32, device=self.device)
            d["no_user_probs"] = predict_no_user_branch(exp.rm, holdout_obs)
            d["train_obs_probs"] = self._pooled_probs(d["holdout_X"])
            d["ctx_obs_probs"], d["ctx_obs_info"] = fit_ctx_obs_only_rm(
                self.cfg, ds.obs_dim, self.device,
                d["X_test"][:d["best"]["ctx"]], d["y_test"][:d["best"]["ctx"]], d["holdout_X"])
            info = d["ctx_obs_info"]
            val_auc = info["val_auroc"]
            val_auc_text = f"{val_auc:.4f}" if not np.isnan(val_auc) else "N/A"
            log(f"  {ABL_TARGET_OBS} [{d['name']}]: ctx={d['best']['ctx']}  "
                f"train={info['train_n']}  val={info['val_n']}  "
                f"split={info['split']}  val_auc={val_auc_text}  "
                f"val_loss={info['val_loss']:.4f}")

        for probs_key, sub, title, mkey in [
            ("no_user_probs",   "wo_user_emb",       ABL_NO_USER,    "wo_user_emb"),
            ("train_obs_probs", "train_user_pooled", ABL_TRAIN_OBS,  "train_user_pooled"),
            ("ctx_obs_probs",   "target_user_only",  ABL_TARGET_OBS, "target_user_only"),
        ]:
            if not any(d[probs_key] is not None for d in drivers):
                continue
            ms = evaluate_predictions(ys, [d[probs_key] for d in drivers],
                                       ablation_dir / sub, names, save_name="metrics", title=title)
            for d, m in zip(drivers, ms):
                if probs_key == "ctx_obs_probs" and d["ctx_obs_info"] is not None:
                    m.update(d["ctx_obs_info"])
                all_metrics[f"{mkey}/{d['name']}"] = m
                log(f"  {title} [{d['name']}] AUROC={m['auroc']:.4f}  "
                    f"AUPRC={m['auprc']:.4f}  Brier={m['brier']:.4f}")

        for pct in sorted({pct for d in drivers for pct in d["snapshots"]}):
            ds_with = [d for d in drivers if pct in d["snapshots"]]
            self._plot_bars_pct(ds_with, pct, plots_dir / "snapshots" / f"context_{pct}pct")

        snap_dir = plots_dir / "snapshots" / "context_best"
        best_names_list, best_vals_list = [], []
        for d in drivers:
            n_, v_ = [ABL_ADAPTED], [d["best"]["auroc"]]
            for key, lab in [("no_user_probs", ABL_NO_USER),
                             ("ctx_obs_probs", ABL_TARGET_OBS),
                             ("train_obs_probs", ABL_TRAIN_OBS)]:
                if d[key] is not None:
                    n_.append(lab); v_.append(roc_auc_score(d["holdout_y"], d[key]))
            best_names_list.append(n_); best_vals_list.append(v_)
        plot_ablation_bars(best_names_list, best_vals_list, names,
                            snap_dir / "user_emb_ablation.png",
                            title="Model Ablation (best ctx)", xtick_fn=xtick_labels)

        log("\n  [Ablation] Model comparison (best ctx):")
        for d, n_, v_ in zip(drivers, best_names_list, best_vals_list):
            log(f"   [{d['name']}]")
            ablation = {k: round(v, 4) for k, v in zip(n_, v_)}
            for k, v in ablation.items():
                log(f"    {k:>10}: AUROC={v:.4f}")
            all_metrics[f"ablation/user_emb/{d['name']}"] = ablation

    def _plot_bars_pct(self, ds_with, pct, snap_dir):
        names_list, vals_list = [], []
        for d in ds_with:
            snap = d["snapshots"][pct]
            adapted = roc_auc_score(d["holdout_y"], snap["probs"])
            n_, v_ = [ABL_ADAPTED], [adapted]
            if d["no_user_probs"] is not None:
                n_.append(ABL_NO_USER); v_.append(roc_auc_score(d["holdout_y"], d["no_user_probs"]))
            if d["train_obs_probs"] is not None:
                n_.append(ABL_TRAIN_OBS); v_.append(roc_auc_score(d["holdout_y"], d["train_obs_probs"]))
            names_list.append(n_); vals_list.append(v_)
        plot_ablation_bars(names_list, vals_list, [d["name"] for d in ds_with],
                            snap_dir / "user_emb_ablation.png",
                            title=f"Model Ablation - Context {pct}%", xtick_fn=xtick_labels)
