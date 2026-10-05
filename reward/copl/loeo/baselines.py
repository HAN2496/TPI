"""Baselines for the CoPL LOEO protocol: Pooled-CNN (fine-tuned per budget), Indep-CNN, k-NN vote."""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..rm import ObsOnlyCNNRewardModel, ObsOnlyRewardModel, RMEdgeDataset, rm_collate
from ..trainer import CoPLRMTrainer
from . import encoders as E


def obs_only_model(cfg, obs_dim):
    if cfg.rm_model == "mlp":
        return ObsOnlyRewardModel(obs_dim, hidden=cfg.rm_mlp_hidden, mlp_hidden=cfg.rm_mlp_hidden)
    return ObsOnlyCNNRewardModel(obs_dim=obs_dim, hidden=cfg.rm_hidden, mlp_hidden=cfg.rm_mlp_hidden,
                                 kernel_size=cfg.rm_kernel_size, layers=cfg.rm_layers)


def _rm_cfg(cfg, device, lr=None, epochs=None):
    return {"device": str(device), "rm_lr": cfg.rm_lr if lr is None else lr, "rm_weight_decay": cfg.rm_weight_decay,
            "rm_lambda_reg": 0.0, "rm_epochs": cfg.rm_epochs if epochs is None else epochs,
            "use_pos_weight": cfg.use_pos_weight, "rm_select": getattr(cfg, "rm_select", "auc")}


def _predict(model, obs, device, bs=1024):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(obs), bs):
            out.append(torch.sigmoid(model(torch.as_tensor(obs[i:i + bs], dtype=torch.float32, device=device))).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


class PooledCNN:
    """One observation-only CNN on all population labels; fine-tuned on population + context per budget."""
    name = "pooled"

    def __init__(self, cfg, device):
        self.cfg, self.device = cfg, device

    def fit_population(self, gds, rm_series, verbose=0):
        cfg = self.cfg
        self.rm_series = rm_series
        self.tr_u, self.tr_i, self.tr_y = gds.tr_u, gds.tr_i, gds.tr_y
        self.model = obs_only_model(cfg, rm_series.shape[2]).to(self.device)
        tr = DataLoader(RMEdgeDataset(gds.tr_u, gds.tr_i, gds.tr_y, rm_series), batch_size=cfg.rm_batch_size,
                        shuffle=True, collate_fn=rm_collate)
        va = DataLoader(RMEdgeDataset(gds.va_u, gds.va_i, gds.va_y, rm_series), batch_size=cfg.rm_batch_size,
                        shuffle=False, collate_fn=rm_collate)
        self.val_auc, _ = CoPLRMTrainer(self.model, _rm_cfg(cfg, self.device), log_dir=None).train(
            tr, va, None, gds.tr_y, verbose=verbose)
        self.temperature = 1.0
        if getattr(cfg, "rm_calibrate", False):               # same post-hoc calibration as the CoPL reward model
            from .fold import calibrate
            self.model, self.temperature = calibrate(self.model, va, None, self.device)
        self.pos_weight = float((1 - gds.tr_y).sum() / max(1, gds.tr_y.sum())) if cfg.use_pos_weight else None
        return self

    def predict(self, X_ctx, y_ctx, X_hold):
        """Fine-tune a copy on population + context (context up-weighted), then predict the holdout."""
        cfg = self.cfg
        if len(y_ctx) == 0 or cfg.pooled_ft_epochs <= 0:
            return _predict(self.model, X_hold, self.device)
        model = copy.deepcopy(self.model)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.pooled_ft_lr, weight_decay=cfg.rm_weight_decay)
        X_pop = self.rm_series[self.tr_i]
        y_pop = self.tr_y
        X_all = np.concatenate([X_pop, X_ctx]); y_all = np.concatenate([y_pop, y_ctx]).astype(np.float32)
        w_all = np.concatenate([np.ones(len(y_pop)), np.full(len(y_ctx), cfg.pooled_ctx_weight)]).astype(np.float32)
        pw = None if self.pos_weight is None else torch.tensor([self.pos_weight], device=self.device)
        n = len(y_all); bs = cfg.rm_batch_size
        rng = np.random.default_rng(cfg.seed + len(y_ctx))
        model.train()
        for _ in range(cfg.pooled_ft_epochs):
            perm = rng.permutation(n)
            for i in range(0, n, bs):
                b = perm[i:i + bs]
                xb = torch.as_tensor(X_all[b], dtype=torch.float32, device=self.device)
                yb = torch.as_tensor(y_all[b], device=self.device)
                wb = torch.as_tensor(w_all[b], device=self.device)
                loss = (F.binary_cross_entropy_with_logits(model(xb), yb, pos_weight=pw, reduction="none") * wb).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        return _predict(model, X_hold, self.device)


class IndepCNN:
    """Observation-only CNN trained on the held-out evaluator's context only (no transfer)."""
    name = "indep"

    def __init__(self, cfg, device):
        self.cfg, self.device = cfg, device
        self.obs_dim = None

    def fit_population(self, gds, rm_series, verbose=0):
        self.obs_dim = rm_series.shape[2]
        return self

    def predict(self, X_ctx, y_ctx, X_hold):
        cfg = self.cfg
        t = len(y_ctx)
        if t < cfg.indep_min_labels:
            return np.full(len(X_hold), np.nan)
        if len(np.unique(y_ctx)) < 2:                 # class prior with Laplace smoothing
            return np.full(len(X_hold), (y_ctx.sum() + 1.0) / (t + 2.0))
        torch.manual_seed(cfg.seed + t)
        model = obs_only_model(cfg, self.obs_dim).to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.rm_lr, weight_decay=cfg.rm_weight_decay)
        xb = torch.as_tensor(X_ctx, dtype=torch.float32, device=self.device)
        yb = torch.as_tensor(y_ctx, dtype=torch.float32, device=self.device)
        pw = torch.tensor([(1 - y_ctx).sum() / max(1, y_ctx.sum())], device=self.device) if cfg.use_pos_weight else None
        model.train()
        for _ in range(cfg.indep_epochs):
            loss = F.binary_cross_entropy_with_logits(model(xb), yb, pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
        return _predict(model, X_hold, self.device)


class KNNVote:
    """Label vote of the k nearest population items in the encoder's latent space (no GCF, no RM).

    Context-independent by construction: it isolates what the similarity graph alone predicts.
    """
    name = "knn_vote"

    def __init__(self, cfg, device=None):
        self.cfg = cfg

    def fit_population(self, gds, rm_series=None, verbose=0):
        self.builder = gds.sim_builder
        self.Z_pop = gds.Z_train
        y = np.zeros(gds.n_items, dtype=np.int64)
        for uid, (ids, yy) in gds.per_user_items.items():
            y[ids] = yy
        self.y_pop = y
        self.metric = E.latent_metric(self.builder)
        self.gamma = None if self.metric == "cosine" else getattr(self.builder, "gamma", None)
        return self

    def predict_graph(self, Xg_hold):
        Zq = self.builder.transform_test(Xg_hold)
        return E.heldout_vote(self.Z_pop, self.y_pop, Zq, self.cfg.knn_vote_k, metric=self.metric,
                              alpha=self.cfg.knn_vote_alpha, gamma=self.gamma)
