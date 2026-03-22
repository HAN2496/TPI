import numpy as np
import torch
from collections import defaultdict
from pathlib import Path
from torch.utils.data import DataLoader

from ..experiment import BaseExperiment
from .vae_binary import VAEModel
from .trainer import VPLTrainerBinary, Annealer
from sklearn.model_selection import train_test_split
from .utils import PreferenceDataset, convert_to_binary_context
from ...data.splits import load_sequences
from ...evaluation import evaluate_predictions, save_metrics_txt, compute_sequential_aurocs, plot_sequential_auroc, plot_training_curves
from .visualization import collect_z_binary, plot_z_space, plot_z_dims, plot_z_evolution, plot_reward_dist


class VPLBinaryExperiment(BaseExperiment):

    def build(self):
        cfg = self.cfg
        all_drivers = sorted(set(cfg.train_driver_names + cfg.test_driver_names))
        raw_data_cache = {}
        all_obs = []

        for name in all_drivers:
            X, y = load_sequences(name, cfg.features, cfg.time_range, cfg.downsample,
                                   smooth=cfg.smooth, smooth_cutoff=cfg.smooth_cutoff, smooth_order=cfg.smooth_order)
            raw_data_cache[name] = (X, y)
            if cfg.normalize and name in cfg.train_driver_names:
                all_obs.append(X)

        norm_mean = norm_std = None
        if cfg.normalize and all_obs:
            concat = np.concatenate(all_obs, axis=0)
            norm_mean = concat.mean(axis=(0, 1))
            norm_std = concat.std(axis=(0, 1)) + 1e-6

        def norm(X):
            return (X - norm_mean) / norm_std if norm_mean is not None else X

        train_queries = defaultdict(list)
        self.train_driver_data = {}
        self.test_driver_data = {}

        for name in all_drivers:
            X_raw, y = raw_data_cache[name]
            X = norm(X_raw)
            if name in cfg.test_driver_names:
                self.test_driver_data[name] = (X, y)
                continue
            self.train_driver_data[name] = (X, y)
            perm = np.random.permutation(len(X))
            dataset_dict = convert_to_binary_context(X[perm], y[perm], name, cfg.context_size, balanced=cfg.balanced)
            if dataset_dict is None:
                continue
            for key in ['observations', 'labels', 'driver_name']:
                train_queries[key].extend(dataset_dict[key])

        self.train_dataset_dict = {k: np.stack(v) for k, v in train_queries.items() if v}

        xb = self.train_dataset_dict["observations"]
        annotation_size, size_segment, obs_dim = xb.shape[1], xb.shape[2], xb.shape[3]
        encoder_input = annotation_size * (size_segment * obs_dim + 1)
        decoder_input = obs_dim + cfg.latent_dim

        reward_scaling = size_segment if cfg.reward_scaling == "T" else cfg.reward_scaling
        self.model = VAEModel(
            encoder_input=encoder_input, decoder_input=decoder_input,
            latent_dim=cfg.latent_dim, hidden_dim=cfg.hidden_dim,
            annotation_size=annotation_size, size_segment=size_segment,
            kl_weight=cfg.kl_weight, flow_prior=cfg.flow_prior,
            annealer=Annealer(cfg.n_epochs // cfg.anneal_cycles, cfg.anneal_shape,
                              cyclical=cfg.anneal_cycles > 1),
            reward_scaling=reward_scaling,
        ).to(cfg.device)

    def train(self, out_dir: Path) -> dict:
        cfg = self.cfg
        n_total = len(self.train_dataset_dict["observations"])
        all_idx = np.arange(n_total)
        train_idx, val_idx = train_test_split(all_idx, test_size=cfg.val_size, random_state=cfg.seed)
        train_dict = {k: v[train_idx] for k, v in self.train_dataset_dict.items()}
        val_dict   = {k: v[val_idx]   for k, v in self.train_dataset_dict.items()}
        train_loader = DataLoader(PreferenceDataset(train_dict), batch_size=cfg.batch_size, shuffle=True)
        val_loader   = DataLoader(PreferenceDataset(val_dict),   batch_size=len(val_idx), shuffle=False)

        self.trainer = VPLTrainerBinary(self.model, out_dir, {
            "device": cfg.device, "lr": cfg.lr, "weight_decay": cfg.weight_decay,
            "n_epochs": cfg.n_epochs, "eval_freq": cfg.eval_freq,
            "early_stop": cfg.early_stop, "patience": cfg.patience, "min_delta": cfg.min_delta,
        })
        print("[1] Training VPL Binary...")
        metrics, _ = self.trainer.train(train_loader, val_loader, verbose=cfg.verbose)
        plot_training_curves(dict(metrics), out_dir / "plots" / "training_curves.png", title="VPL Binary")
        self.model.load_state_dict(torch.load(self.trainer.best_model_path, weights_only=True))
        return {}

    def _eval_sequential_holdout(self, X, y, driver_name, out_dir=None, is_test_driver=False):
        """Hold-out evaluation: use X[:t] to estimate z, and evaluate on fixed X[split_idx:]."""
        device = torch.device(self.cfg.device)
        split_idx = len(X) // 2

        if split_idx < 1 or len(X) - split_idx < 1:
            return None, None, None, None, []

        test_X = X[split_idx:]
        test_y = y[split_idx:]

        if len(np.unique(test_y)) < 2:
            print(f"  [Warning] {driver_name} test set lacks both classes. Skipping.")
            return None, None, None, None, []

        T = X.shape[1]
        test_obs = torch.tensor(test_X, dtype=torch.float32, device=device).unsqueeze(1)

        ctx_sizes = []
        test_aurocs = []
        final_probs = None
        z_trajectory = []

        from sklearn.metrics import roc_auc_score

        target_pcts = [0.1, 0.2, 0.3, 0.4, 0.5]
        snapshot_steps = {max(1, int(len(X) * p)): int(p * 100) for p in target_pcts}
        snapshot_steps = {k: v for k, v in snapshot_steps.items() if k <= split_idx}

        for t in range(1, split_idx + 1):
            ctx_dict = convert_to_binary_context(
                X[:t], y[:t], driver_name, self.cfg.context_size, balanced=False)
            if ctx_dict is None or len(ctx_dict["labels"]) == 0:
                continue

            loader = DataLoader(PreferenceDataset(ctx_dict), batch_size=len(ctx_dict["labels"]), shuffle=False)
            means = []
            with torch.no_grad():
                for b in loader:
                    m, _ = self.model.encode(b["observations"].to(device).float(),
                                              b["labels"].to(device).float())
                    means.append(m.cpu().numpy())

            if not means:
                continue

            z = np.mean(np.concatenate(means, axis=0), axis=0)
            z_trajectory.append(z)
            z_t = torch.tensor(z, dtype=torch.float32, device=device).view(1, 1, 1, -1).expand(len(test_X), 1, T, -1)

            with torch.no_grad():
                step_r = self.model.decode(test_obs, z_t).squeeze(1).squeeze(-1)
                probs = torch.sigmoid(step_r.mean(dim=1)).cpu().numpy()

            auroc = roc_auc_score(test_y, probs)
            ctx_sizes.append(t)
            test_aurocs.append(auroc)
            final_probs = probs

            if out_dir and is_test_driver and t in snapshot_steps:
                pct = snapshot_steps[t]
                snap_dir = out_dir / "plots" / "snapshots" / f"context_{pct}pct"
                snap_dir.mkdir(parents=True, exist_ok=True)

                evaluate_predictions(test_y, probs, snap_dir, "metrics", 
                                     title=f"VPL - {driver_name} (Context {pct}%)")

                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(range(len(z)), z, color="mediumpurple")
                ax.set_title(f"Latent Vector (z) - Context {pct}%")
                ax.set_xlabel("Latent Dimension")
                ax.set_ylabel("Value")
                ax.grid(axis='y', alpha=0.3)
                fig.tight_layout()
                fig.savefig(snap_dir / "latent_z.png", dpi=100)
                plt.close(fig)

        if final_probs is None:
            return None, None, None, None, []

        return ctx_sizes, test_aurocs, final_probs, test_y, z_trajectory

    def evaluate(self, out_dir: Path) -> dict:
        self.model.eval()
        all_metrics = {}
        device = torch.device(self.cfg.device)
        plots_dir = out_dir / "plots"
        test_z = {}

        for driver_name, (X, y) in self.test_driver_data.items():
            ctx, aurocs, p, l, z_traj = self._eval_sequential_holdout(X, y, driver_name, out_dir=out_dir, is_test_driver=True)
            if p is None:
                continue
            plot_sequential_auroc(ctx, aurocs, plots_dir, driver_name)
            m = evaluate_predictions(l, p, plots_dir, driver_name,
                                      title=f"VPL Binary - {driver_name} (test)")
            all_metrics[f"test/{driver_name}"] = m
            print(f"  {driver_name}: AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}  Brier={m['brier']:.4f}")
            if z_traj:
                test_z[driver_name] = np.stack(z_traj)
                plot_z_evolution(z_traj, ctx, driver_name, plots_dir / f"z_evolution_{driver_name}.png")

        print("\n[2] Evaluating training drivers...")
        for uname, (X, y) in self.train_driver_data.items():
            ctx, aurocs, p, l, _ = self._eval_sequential_holdout(X, y, uname, out_dir=out_dir, is_test_driver=False)
            if p is None:
                continue
            plot_sequential_auroc(ctx, aurocs, plots_dir / "train", uname)
            m = evaluate_predictions(l, p, plots_dir / "train", uname,
                                      title=f"VPL Binary - {uname} (train)")
            all_metrics[f"train/{uname}"] = m
            print(f"  {uname}: AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}  Brier={m['brier']:.4f}")

        z_by_driver = collect_z_binary(self.model, self.train_driver_data, self.cfg.context_size, device)
        plot_z_space({**z_by_driver, **test_z}, list(self.test_driver_data.keys()), plots_dir / "z_space.png")
        plot_z_dims(z_by_driver, plots_dir / "z_dims.png")
        plot_reward_dist(self.model, self.train_driver_data, z_by_driver, device, plots_dir / "reward_model")

        save_metrics_txt(all_metrics, out_dir / "metrics.txt")
        return all_metrics
    def save(self, out_dir: Path) -> None:
        pass  # VPLTrainerBinary가 best_model.pt 저장

    def load(self, out_dir: Path) -> None:
        self.model.load_state_dict(
            torch.load(out_dir / "best_model.pt", map_location=self.cfg.device, weights_only=True))

    def make_summary(self, train_metrics: dict, eval_metrics: dict) -> dict:
        s = super().make_summary(train_metrics, eval_metrics)
        s["test_drivers"] = self.cfg.test_driver_names
        return s
