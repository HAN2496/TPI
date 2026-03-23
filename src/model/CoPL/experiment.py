from dataclasses import asdict
import json
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from ..experiment import BaseExperiment
from .gcf import CoPLGCF
from .gcf_gcn import CoPLGCF_PyG
from .rm import RewardModel, CNNRewardModel, MoLECNNRewardModel, RMEdgeDataset, rm_collate
from .dataset import CoPLGraphDataset
from .similarity import build_similarity
from .trainer import CoPLGCFTrainer, CoPLRMTrainer
from .visualization import (plot_test_item_bridge, plot_item_embeddings,
                            plot_user_embeddings, plot_rm_distributions, plot_wu_evolution)
from ...evaluation import evaluate_predictions, save_metrics_txt, compute_sequential_aurocs, plot_sequential_auroc, plot_training_curves


class CoPLExperiment(BaseExperiment):

    def run(self, out_dir: Path, eval_only: bool = False) -> dict:
        self._out_dir = out_dir
        self._eval_only = eval_only
        self.build()
        if eval_only:
            self.load(out_dir)
            train_metrics = {}
        else:
            with open(out_dir / "cfg.json", "w", encoding="utf-8") as f:
                json.dump(asdict(self.cfg), f, ensure_ascii=False, indent=2)
            train_metrics = self.train(out_dir)
            self.save(out_dir)
        eval_metrics = self.evaluate(out_dir)
        summary = self.make_summary(train_metrics, eval_metrics)
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return eval_metrics

    def _get_load_path(self, component: str):
        """컴포넌트를 로드할 Path 반환. None이면 학습."""
        if self._eval_only:
            return self._out_dir
        load_ts = getattr(self.cfg, f"load_{component}", None)
        if load_ts is None:
            return None
        return Path("artifacts") / "copl" / load_ts

    def build(self):
        self.device = torch.device(self.cfg.device)
        self._log("=" * 60)
        self._log("[CoPL] Building graph dataset...")
        vae_path = self._get_load_path("vae")
        if vae_path is not None:
            self._log(f"  [VAE] Loading from {vae_path}...")
            sim_builder = build_similarity(self.cfg.similarity_method)
            sim_builder.load(vae_path / "vae.pt", device=self.device)
            self.dataset = CoPLGraphDataset(self.cfg, sim_builder=sim_builder)
        else:
            self.dataset = CoPLGraphDataset(self.cfg)
            self.dataset.sim_builder.save(self._out_dir / "vae.pt")
        self.dataset.to(self.device)
        self.gcf = self._build_gcf()
        self.rm = self._build_rm()

    def _build_gcf(self):
        ds, cfg = self.dataset, self.cfg
        common = dict(
            n_u=ds.n_users, n_i=ds.n_items, d=cfg.gcf_emb_dim,
            pos_adj_norm=ds.Apos_norm, neg_adj_norm=ds.Aneg_norm,
            dropout=cfg.gcf_dropout, l=cfg.gcf_layers,
            item_item_adj_norm=ds.Aii_norm, item_item_weight=cfg.item_item_weight,
        )
        if cfg.gcf_model == "gcf_gcn":
            return CoPLGCF_PyG(**common).to(self.device)

        Z = torch.tensor(self.dataset.Z_train, dtype=torch.float32)  # (n_items, vae_latent_dim)
        if Z.shape[1] != cfg.gcf_emb_dim:
            proj = torch.nn.Linear(Z.shape[1], cfg.gcf_emb_dim, bias=False)
            torch.nn.init.xavier_uniform_(proj.weight)
            with torch.no_grad():
                item_feat_init = proj(Z)
        else:
            item_feat_init = Z

        return CoPLGCF(**common, loss_type=cfg.gcf_loss_type,
                       loss_kwargs=cfg.gcf_loss_kwargs,
                       item_feat_init=item_feat_init,
                       m_i_type=cfg.gcf_m_i_type).to(self.device)

    def _build_rm(self):
        cfg = self.cfg
        common = dict(obs_dim=self.dataset.obs_dim, user_dim=cfg.gcf_emb_dim)
        if cfg.rm_model == "mlp":
            return RewardModel(**common, hidden=cfg.rm_mlp_hidden).to(self.device)
        elif cfg.rm_model == "cnn":
            return CNNRewardModel(**common, hidden=cfg.rm_hidden, mlp_hidden=cfg.rm_mlp_hidden,
                                  kernel_size=cfg.rm_kernel_size, layers=cfg.rm_layers).to(self.device)
        elif cfg.rm_model == "mole_cnn":
            return MoLECNNRewardModel(**common, hidden=cfg.rm_hidden, mlp_hidden=cfg.rm_mlp_hidden,
                                      kernel_size=cfg.rm_kernel_size, layers=cfg.rm_layers,
                                      num_experts=cfg.rm_num_experts, rank=cfg.rm_mole_rank,
                                      tau=cfg.rm_mole_tau).to(self.device)
        raise ValueError(f"Unknown rm_model: {cfg.rm_model}")

    def train(self, out_dir: Path) -> dict:
        cfg, ds = self.cfg, self.dataset

        gcf_path = self._get_load_path("gcf")
        if gcf_path is None:
            self._log("\n[1] Training GCF...")
            gcf_trainer = CoPLGCFTrainer(self.gcf, {
                'device': cfg.device, 'gcf_lr': cfg.gcf_lr,
                'gcf_weight_decay': cfg.gcf_weight_decay, 'gcf_lambda_reg': cfg.gcf_lambda_reg,
                'gcf_epochs': cfg.gcf_epochs, 'use_pos_weight': cfg.use_pos_weight,
            }, log_dir=out_dir)
            gcf_best_auc, _, E_u, E_i, gcf_metrics = gcf_trainer.train(
                ds.tr_u, ds.tr_i, ds.tr_y, ds.va_u, ds.va_i, ds.va_y, verbose=cfg.verbose)
            self.E_u = E_u.to(self.device)
            self.E_i = E_i
            self._log(f"  GCF Best Val AUC: {gcf_best_auc:.4f}")
        else:
            self._log(f"\n[1] Loading GCF from {gcf_path}...")
            self.gcf.load_state_dict(torch.load(gcf_path / "best_gcf.pt", map_location=self.device, weights_only=True))
            self.gcf.eval()
            with torch.no_grad():
                E_u, E_i = self.gcf.encode_graph(test=True)
            self.E_u = E_u.to(self.device)
            self.E_i = E_i
            gcf_best_auc = float("nan")
            gcf_metrics = {}

        rm_path = self._get_load_path("rm")
        if rm_path is None:
            self._log("\n[2] Training Reward Model...")
            train_loader = DataLoader(
                RMEdgeDataset(ds.tr_u, ds.tr_i, ds.tr_y, ds.item_series),
                batch_size=cfg.rm_batch_size, shuffle=True, collate_fn=rm_collate)
            val_loader = DataLoader(
                RMEdgeDataset(ds.va_u, ds.va_i, ds.va_y, ds.item_series),
                batch_size=cfg.rm_batch_size, shuffle=False, collate_fn=rm_collate)
            rm_trainer = CoPLRMTrainer(self.rm, {
                'device': cfg.device, 'rm_lr': cfg.rm_lr,
                'rm_weight_decay': cfg.rm_weight_decay, 'rm_lambda_reg': cfg.rm_lambda_reg,
                'rm_epochs': cfg.rm_epochs, 'use_pos_weight': cfg.use_pos_weight,
            }, log_dir=out_dir)
            rm_best_auc, rm_metrics = rm_trainer.train(train_loader, val_loader, self.E_u, ds.tr_y, verbose=cfg.verbose)
            self._log(f"  RM Best Val AUC: {rm_best_auc:.4f}")
        else:
            self._log(f"\n[2] Loading Reward Model from {rm_path}...")
            self.rm.load_state_dict(torch.load(rm_path / "best_rm.pt", map_location=self.device, weights_only=True))
            rm_best_auc = float("nan")
            rm_metrics = {}

        combined = {}
        for k, v in gcf_metrics.items():
            prefix, name = k.split("/", 1)
            combined[f"{prefix}/gcf_{name}"] = v
        for k, v in rm_metrics.items():
            prefix, name = k.split("/", 1)
            combined[f"{prefix}/rm_{name}"] = v
        if combined:
            plot_training_curves(combined, out_dir / "plots" / "training_curves.png", title="CoPL Training")

        return {"gcf_val_auroc": gcf_best_auc, "rm_val_auroc": rm_best_auc}

    def evaluate(self, out_dir: Path) -> dict:
        cfg, ds = self.cfg, self.dataset
        self._log(f"\n[3] Test-time evaluation on {cfg.test_driver}...")
        X_test, y_test = ds.load_test_driver(cfg.test_driver)

        self.rm.eval()
        all_metrics = {}
        neigh_idx_last, neigh_w_last = None, None
        wu_history, ctx_sizes_wu, e_u_final = [], [], None

        split_idx = len(X_test) // 2
        if split_idx >= 1 and len(X_test) - split_idx >= 1:
            holdout_X = X_test[split_idx:]
            holdout_y = y_test[split_idx:]
            
            if len(np.unique(holdout_y)) < 2:
                self._log(f"  [Warning] {cfg.test_driver} test set lacks both classes. Skipping seq AUROC.")
            else:
                ctx_sizes, test_aurocs = [], []
                final_probs = None
                holdout_obs = torch.tensor(holdout_X, dtype=torch.float32).to(self.device)

                target_pcts = [0.1, 0.2, 0.3, 0.4, 0.5]
                snapshot_steps = {max(1, int(len(X_test) * p)): int(p * 100) for p in target_pcts}
                snapshot_steps = {k: v for k, v in snapshot_steps.items() if k <= split_idx}

                for t in range(1, split_idx + 1):
                    _, neigh_idx_ctx, neigh_w_ctx = ds.attach_test_items(
                        X_test[:t], self.E_i.cpu(), topk=cfg.adapt_topk, device=self.device)
                    e_u_t, w_u = ds.adapt_test_user(
                        y_test[:t], neigh_idx_ctx, neigh_w_ctx, self.E_u, device=self.device)

                    wu_history.append(w_u)
                    ctx_sizes_wu.append(t)
                    e_u_final = e_u_t

                    e_u_t_expanded = e_u_t.unsqueeze(0).expand(len(holdout_X), -1)
                    with torch.no_grad():
                        probs = torch.sigmoid(self.rm(e_u_t_expanded, holdout_obs)).cpu().numpy()

                    auroc = roc_auc_score(holdout_y, probs)
                    ctx_sizes.append(t)
                    test_aurocs.append(auroc)
                    final_probs = probs
                    neigh_idx_last, neigh_w_last = neigh_idx_ctx, neigh_w_ctx

                    if t in snapshot_steps:
                        pct = snapshot_steps[t]
                        snap_dir = out_dir / "plots" / "snapshots" / f"context_{pct}pct"
                        snap_dir.mkdir(parents=True, exist_ok=True)

                        evaluate_predictions(holdout_y, probs, snap_dir, "metrics",
                                             title=f"CoPL - {cfg.test_driver} (Context {pct}%)")

                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.bar(cfg.train_driver_names, w_u, color="mediumseagreen")
                        ax.set_title(f"User Attention (w_u) - Context {pct}%")
                        ax.set_ylabel("Weight")
                        ax.set_ylim(0, 1.0)
                        ax.grid(axis='y', alpha=0.3)
                        fig.tight_layout()
                        fig.savefig(snap_dir / "user_attention.png", dpi=100)
                        plt.close(fig)

                        plot_test_item_bridge(neigh_idx_ctx, neigh_w_ctx, ds.item_owner_uid, ds.train_drivers,
                                              snap_dir / "item_bridge.png")

                if final_probs is not None:
                    plot_sequential_auroc(ctx_sizes, test_aurocs, out_dir / "plots", cfg.test_driver)
                    m = evaluate_predictions(holdout_y, final_probs, out_dir / "plots", cfg.test_driver,
                                              title=f"CoPL - {cfg.test_driver} (test holdout)")
                    all_metrics[f"test/{cfg.test_driver}"] = m
                    self._log(f"  Test AUROC: {m['auroc']:.4f}  AUPRC: {m['auprc']:.4f}  Brier: {m['brier']:.4f}")

        plots_dir = out_dir / "plots"
        if neigh_idx_last is not None:
            plot_test_item_bridge(neigh_idx_last, neigh_w_last, ds.item_owner_uid, ds.train_drivers,
                                  plots_dir / "bridge.png")

        self._log("\n[4] Evaluating training drivers...")
        for uid, uname in enumerate(cfg.train_driver_names):
            u_tr = ds.tr_u == uid
            u_va = ds.va_u == uid
            u_iids = np.concatenate([ds.tr_i[u_tr], ds.va_i[u_va]])
            u_labels = np.concatenate([ds.tr_y[u_tr], ds.va_y[u_va]])
            if len(u_iids) == 0:
                continue
            obs_u = torch.tensor(ds.item_series[u_iids], dtype=torch.float32).to(self.device)
            u_emb = self.E_u[uid].unsqueeze(0).expand(len(u_iids), -1)
            with torch.no_grad():
                u_probs = torch.sigmoid(self.rm(u_emb, obs_u)).cpu().numpy()
            m = evaluate_predictions(u_labels, u_probs, plots_dir / "train", uname,
                                      title=f"CoPL - {uname} (train)")
            all_metrics[f"train/{uname}"] = m
            self._log(f"  {uname}: AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}  Brier={m['brier']:.4f}")

        item_labels = np.zeros(ds.n_items, dtype=np.int64)
        for uid, (item_ids, y) in ds.per_user_items.items():
            item_labels[item_ids] = y

        ds.sim_builder.visualize(plots_dir / "similarity", ds.item_series, ds.item_owner_uid,
                                 cfg.train_driver_names, list(cfg.features), item_labels)
        plot_item_embeddings(self.E_i, ds.item_owner_uid,
                             cfg.train_driver_names, plots_dir / "embeddings")
        plot_user_embeddings(self.E_u, cfg.train_driver_names,
                             e_u_final, cfg.test_driver, plots_dir / "embeddings")
        plot_rm_distributions(self.rm, self.E_u, ds.item_series, ds.item_owner_uid,
                              item_labels, cfg.train_driver_names, self.device, plots_dir / "reward_model")
        plot_wu_evolution(wu_history, ctx_sizes_wu, cfg.train_driver_names,
                          plots_dir / "wu_evolution.png")

        save_metrics_txt(all_metrics, out_dir / "metrics.txt")
        return all_metrics

    def save(self, out_dir: Path) -> None:
        pass  # CoPLGCFTrainer, CoPLRMTrainer가 best_gcf.pt, best_rm.pt 저장

    def load(self, out_dir: Path) -> None:
        self.gcf.load_state_dict(torch.load(out_dir / "best_gcf.pt", map_location=self.device, weights_only=True))
        self.gcf.eval()
        with torch.no_grad():
            self.E_u, self.E_i = self.gcf.encode_graph(test=True)
        self.E_u = self.E_u.to(self.device)
        self.rm.load_state_dict(torch.load(out_dir / "best_rm.pt", map_location=self.device, weights_only=True))

    def make_summary(self, train_metrics: dict, eval_metrics: dict) -> dict:
        s = super().make_summary(train_metrics, eval_metrics)
        s["test_driver"] = self.cfg.test_driver
        s["similarity_method"] = self.cfg.similarity_method
        s["aii_meta"] = self.dataset.Aii_meta
        return s
