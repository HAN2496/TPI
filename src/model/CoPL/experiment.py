from dataclasses import asdict
import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from ..experiment import BaseExperiment
from .gcf import CoPLGCF
from .rm import (RewardModel, CNNRewardModel, MoLECNNRewardModel, PreferenceTransformerRewardModel,
                 RMEdgeDataset, rm_collate)
from .dataset import CoPLGraphDataset
from .similarity import build_similarity
from .trainer import CoPLGCFTrainer, CoPLRMTrainer
from .visualization import (plot_test_item_bridge, plot_item_embeddings,
                            plot_user_embeddings, plot_rm_distributions, plot_wu_evolution,
                            plot_attention_bars)
from .ablation import Ablation
from ...evaluation import (evaluate_predictions, save_metrics_txt,
                            plot_sequential_auroc, plot_training_curves)


class CoPLExperiment(BaseExperiment):

    def run(self, out_dir, eval_only = False) -> dict:
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
        sim_ckpt_name = f"{self.cfg.similarity_method}.pt"
        if vae_path is not None:
            self._log(f"  [SIM:{self.cfg.similarity_method}] Loading from {vae_path}...")
            sim_builder = build_similarity(self.cfg.similarity_method)
            sim_builder.load(vae_path / sim_ckpt_name, device=self.device)
            self.dataset = CoPLGraphDataset(self.cfg, sim_builder=sim_builder)
        else:
            self.dataset = CoPLGraphDataset(self.cfg)
            self.dataset.sim_builder.save(self._out_dir / sim_ckpt_name)
        self.dataset.to(self.device)
        self.gcf = self._build_gcf()
        self.rm = self._build_rm()
        self.ablation = Ablation(self) if self.cfg.ablation else None
        if self.ablation:
            self.ablation.build()

    def _build_gcf(self):
        ds, cfg = self.dataset, self.cfg
        common = dict(
            n_u=ds.n_users, n_i=ds.n_items, d=cfg.gcf_emb_dim,
            pos_adj_norm=ds.Apos_norm, neg_adj_norm=ds.Aneg_norm,
            dropout=cfg.gcf_dropout, l=cfg.gcf_layers,
            item_item_adj_norm=ds.Aii_norm, item_item_weight=cfg.item_item_weight,
        )
        if cfg.gcf_model == "gcf_gcn":
            from .gcf_gcn import CoPLGCF_PyG
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
        elif cfg.rm_model == "preference_transformer":
            return PreferenceTransformerRewardModel(**common, hidden=cfg.rm_hidden, num_heads=cfg.rm_num_heads, num_layers=cfg.rm_layers, max_len=cfg.rm_max_len).to(self.device)
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

        train_loader = DataLoader(
            RMEdgeDataset(ds.tr_u, ds.tr_i, ds.tr_y, ds.item_series),
            batch_size=cfg.rm_batch_size, shuffle=True, collate_fn=rm_collate)
        val_loader = DataLoader(
            RMEdgeDataset(ds.va_u, ds.va_i, ds.va_y, ds.item_series),
            batch_size=cfg.rm_batch_size, shuffle=False, collate_fn=rm_collate)

        rm_path = self._get_load_path("rm")
        if rm_path is None:
            self._log("\n[2] Training Reward Model...")
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

        pooled_best_auc, pooled_metrics = float("nan"), {}
        if self.ablation is not None:
            pooled_best_auc, pooled_metrics = self.ablation.fit(out_dir, train_loader, val_loader)

        combined = {}
        for k, v in gcf_metrics.items():
            prefix, name = k.split("/", 1)
            combined[f"{prefix}/gcf_{name}"] = v
        for k, v in rm_metrics.items():
            prefix, name = k.split("/", 1)
            combined[f"{prefix}/rm_{name}"] = v
        for k, v in pooled_metrics.items():
            prefix, name = k.split("/", 1)
            combined[f"{prefix}/pooled_{name}"] = v
        if combined:
            plot_training_curves(combined, out_dir / "plots" / "training_curves.png", title="CoPL Training")

        return {"gcf_val_auroc": gcf_best_auc, "rm_val_auroc": rm_best_auc,
                "pooled_val_auroc": pooled_best_auc}

    def _evaluate_one_driver(self, tname, ds):
        cfg = self.cfg
        self._log(f"\n[3] Test-time evaluation on {tname}...")
        X_test, y_test = ds.load_test_driver(tname)
        split_idx = len(X_test) // 2
        if split_idx < 1 or len(X_test) - split_idx < 1:
            self._log(f"  [Skip] {tname}: too few samples")
            return None
        holdout_X = X_test[split_idx:]
        holdout_y = y_test[split_idx:]
        if len(np.unique(holdout_y)) < 2:
            self._log(f"  [Skip] {tname} test set lacks both classes.")
            return None

        d = {"name": tname, "X_test": X_test, "y_test": y_test,
             "holdout_X": holdout_X, "holdout_y": holdout_y,
             "snapshots": {}, "wu_history": [], "ctx_sizes_wu": [],
             "neigh_idx_last": None, "neigh_w_last": None, "e_u_final": None}

        holdout_obs = torch.tensor(holdout_X, dtype=torch.float32).to(self.device)

        target_pcts = [0.1, 0.2, 0.3, 0.4, 0.5]
        snapshot_steps = {max(1, int(len(X_test) * p)): int(p * 100) for p in target_pcts}
        snapshot_steps = {k: v for k, v in snapshot_steps.items() if k <= split_idx}

        ctx_sizes, test_aurocs = [], []
        best = {"auroc": -1.0}
        for t in range(1, split_idx + 1):
            _, neigh_idx_ctx, neigh_w_ctx = ds.attach_test_items(
                X_test[:t], self.E_i.cpu(), topk=cfg.adapt_topk, device=self.device)
            e_u_t, w_u = ds.adapt_test_user(
                y_test[:t], neigh_idx_ctx, neigh_w_ctx, self.E_u, device=self.device)
            d["wu_history"].append(w_u)
            d["ctx_sizes_wu"].append(t)
            d["e_u_final"] = e_u_t
            d["neigh_idx_last"] = neigh_idx_ctx
            d["neigh_w_last"] = neigh_w_ctx

            e_u_t_expanded = e_u_t.unsqueeze(0).expand(len(holdout_X), -1)
            with torch.no_grad():
                probs = torch.sigmoid(self.rm(e_u_t_expanded, holdout_obs)).cpu().numpy()
            auroc = roc_auc_score(holdout_y, probs)
            ctx_sizes.append(t)
            test_aurocs.append(auroc)
            if auroc > best["auroc"]:
                best = dict(probs=probs, e_u=e_u_t, w_u=w_u,
                            neigh_idx=neigh_idx_ctx, neigh_w=neigh_w_ctx,
                            ctx=t, auroc=auroc)

            if t in snapshot_steps:
                pct = snapshot_steps[t]
                self._log(f"  [Context {pct:>3}%] AUROC={auroc:.4f}")
                d["snapshots"][pct] = dict(probs=probs, w_u=w_u,
                                            neigh_idx=neigh_idx_ctx, neigh_w=neigh_w_ctx, ctx=t)

        d["ctx_sizes"] = ctx_sizes
        d["test_aurocs"] = test_aurocs
        d["best"] = best
        d["best_pct"] = round(best["ctx"] / len(X_test) * 100)
        return d

    def evaluate(self, out_dir):
        cfg, ds = self.cfg, self.dataset
        self.rm.eval()
        all_metrics = {}
        plots_dir = out_dir / "plots"

        drivers = [d for d in (self._evaluate_one_driver(t, ds) for t in cfg.test_driver_names) if d]
        if not drivers:
            save_metrics_txt(all_metrics, out_dir / "metrics.txt")
            return all_metrics

        names = [d["name"] for d in drivers]
        ys = [d["holdout_y"] for d in drivers]

        for pct in sorted({pct for d in drivers for pct in d["snapshots"]}):
            ds_with = [d for d in drivers if pct in d["snapshots"]]
            wn = [d["name"] for d in ds_with]
            snap_dir = plots_dir / "snapshots" / f"context_{pct}pct"
            evaluate_predictions(
                [d["holdout_y"] for d in ds_with],
                [d["snapshots"][pct]["probs"] for d in ds_with],
                snap_dir, wn, save_name="metrics", title=f"CoPL (Context {pct}%)")
            plot_attention_bars(
                [d["snapshots"][pct]["w_u"] for d in ds_with],
                cfg.train_driver_names, wn,
                snap_dir / "user_attention.png", suffix=f" (Context {pct}%)")
            plot_test_item_bridge(
                [d["snapshots"][pct]["neigh_idx"] for d in ds_with],
                [d["snapshots"][pct]["neigh_w"] for d in ds_with],
                ds.item_owner_uid, ds.train_drivers, wn,
                snap_dir / "item_bridge.png")

        plot_sequential_auroc(
            [d["ctx_sizes"] for d in drivers],
            [d["test_aurocs"] for d in drivers],
            plots_dir, names, save_name="seq_auroc")

        ms = evaluate_predictions(
            ys, [d["best"]["probs"] for d in drivers],
            plots_dir, names, save_name="metrics_best",
            title="CoPL (best ctx per driver)")
        for d, m in zip(drivers, ms):
            all_metrics[f"test/{d['name']}"] = m
            self._log(f"  Test [{d['name']}] (best ctx={d['best_pct']}%) "
                      f"AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}  Brier={m['brier']:.4f}")

        snap_dir = plots_dir / "snapshots" / "context_best"
        evaluate_predictions(ys, [d["best"]["probs"] for d in drivers],
                              snap_dir, names, save_name="metrics", title="CoPL (best ctx)")
        plot_attention_bars([d["best"]["w_u"] for d in drivers],
                             cfg.train_driver_names, names,
                             snap_dir / "user_attention.png", suffix=" (best ctx)")
        plot_test_item_bridge(
            [d["best"]["neigh_idx"] for d in drivers],
            [d["best"]["neigh_w"] for d in drivers],
            ds.item_owner_uid, ds.train_drivers, names,
            snap_dir / "item_bridge.png")

        if self.ablation is not None:
            self.ablation.run(drivers, plots_dir, all_metrics)

        plot_test_item_bridge(
            [d["neigh_idx_last"] for d in drivers],
            [d["neigh_w_last"] for d in drivers],
            ds.item_owner_uid, ds.train_drivers, names,
            plots_dir / "bridge.png")

        self._log("\n[4] Evaluating training drivers...")
        train_y_list, train_probs_list, train_names = [], [], []
        for uid, uname in enumerate(cfg.train_driver_names):
            u_iids = np.concatenate([ds.tr_i[ds.tr_u == uid], ds.va_i[ds.va_u == uid]])
            u_labels = np.concatenate([ds.tr_y[ds.tr_u == uid], ds.va_y[ds.va_u == uid]])
            if len(u_iids) == 0:
                continue
            obs_u = torch.tensor(ds.item_series[u_iids], dtype=torch.float32).to(self.device)
            u_emb = self.E_u[uid].unsqueeze(0).expand(len(u_iids), -1)
            with torch.no_grad():
                u_probs = torch.sigmoid(self.rm(u_emb, obs_u)).cpu().numpy()
            train_y_list.append(u_labels)
            train_probs_list.append(u_probs)
            train_names.append(uname)
        if train_names:
            ms = evaluate_predictions(train_y_list, train_probs_list,
                                       plots_dir / "train", train_names,
                                       save_name="metrics", title="CoPL Train Drivers")
            for uname, m in zip(train_names, ms):
                all_metrics[f"train/{uname}"] = m
                self._log(f"  {uname}: AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}  Brier={m['brier']:.4f}")

        item_labels = np.zeros(ds.n_items, dtype=np.int64)
        for uid, (item_ids, y) in ds.per_user_items.items():
            item_labels[item_ids] = y

        ds.sim_builder.visualize(plots_dir / "similarity", ds.item_series, ds.item_owner_uid,
                                 cfg.train_driver_names, list(cfg.features), item_labels)

        E_i_cpu = self.E_i.detach().cpu().numpy()
        E_i_tests = []
        for d in drivers:
            if d["neigh_idx_last"] is None:
                E_i_tests.append(None)
                continue
            E_i_tests.append(torch.tensor(np.stack([
                (d["neigh_w_last"][i][:, None] * E_i_cpu[d["neigh_idx_last"][i]]).sum(axis=0)
                for i in range(d["neigh_idx_last"].shape[0])
            ])))
        plot_item_embeddings(self.E_i, ds.item_owner_uid, cfg.train_driver_names,
                             plots_dir / "embeddings",
                             E_i_tests=E_i_tests, test_driver_names=names)
        plot_user_embeddings(self.E_u, cfg.train_driver_names,
                             [d["e_u_final"] for d in drivers], names,
                             plots_dir / "embeddings")
        plot_rm_distributions(self.rm, self.E_u, ds.item_series, ds.item_owner_uid,
                              item_labels, cfg.train_driver_names, self.device,
                              plots_dir / "reward_model")
        plot_wu_evolution(
            [d["wu_history"] for d in drivers],
            [d["ctx_sizes_wu"] for d in drivers],
            cfg.train_driver_names, names,
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
        if self.ablation is not None:
            self.ablation.load(out_dir)

    def make_summary(self, train_metrics: dict, eval_metrics: dict) -> dict:
        s = super().make_summary(train_metrics, eval_metrics)
        s["test_drivers"] = list(self.cfg.test_driver_names)
        s["similarity_method"] = self.cfg.similarity_method
        s["aii_meta"] = self.dataset.Aii_meta
        return s
