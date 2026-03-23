import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from ..experiment import BaseExperiment
from .vae import VAEModel
from .trainer import VPLTrainer, Annealer
from .utils import PreferenceDataset, create_vpl_dataset, convert_to_pairwise, compute_step_rewards
from ...evaluation import evaluate_predictions, save_metrics_txt, compute_sequential_aurocs, plot_sequential_auroc, plot_training_curves
from .visualization import collect_z_pairwise, plot_z_space, plot_z_dims, plot_reward_dist


class VPLPairwiseExperiment(BaseExperiment):

    def build(self):
        cfg = self.cfg
        self.train_dataset_dict, self.train_driver_data, self.test_driver_data, _ = create_vpl_dataset(
            cfg.train_driver_names, cfg.test_driver_names,
            cfg.features, cfg.time_range, cfg.downsample,
            tie_ratio=cfg.tie_ratio, context_size=cfg.context_size, normalize=cfg.normalize,
            smooth=cfg.smooth, smooth_cutoff=cfg.smooth_cutoff, smooth_order=cfg.smooth_order)

        n_total = len(self.train_dataset_dict["observations"])
        all_idx = np.arange(n_total)
        self.train_idx, self.val_idx = train_test_split(
            all_idx, test_size=cfg.val_size, random_state=cfg.seed)

        xb = self.train_dataset_dict["observations"]
        annotation_size, size_segment, obs_dim = xb.shape[1], xb.shape[2], xb.shape[3]
        encoder_input = annotation_size * (2 * size_segment * obs_dim + 1)
        decoder_input = size_segment * obs_dim + cfg.latent_dim

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
        train_dict = {k: v[self.train_idx] for k, v in self.train_dataset_dict.items()}
        val_dict = {k: v[self.val_idx] for k, v in self.train_dataset_dict.items()}
        train_loader = DataLoader(PreferenceDataset(train_dict), batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(PreferenceDataset(val_dict), batch_size=len(val_dict["labels"]), shuffle=False)

        self.trainer = VPLTrainer(self.model, out_dir, {
            "device": cfg.device, "lr": cfg.lr, "weight_decay": cfg.weight_decay,
            "n_epochs": cfg.n_epochs, "eval_freq": cfg.eval_freq,
            "early_stop": cfg.early_stop, "patience": cfg.patience, "min_delta": cfg.min_delta,
        })
        print("[1] Training VPL Pairwise...")
        metrics, best_acc = self.trainer.train(train_loader, val_loader, verbose=cfg.verbose)
        plot_training_curves(dict(metrics), out_dir / "training_curves.png", title="VPL Pairwise")
        return {"val_accuracy": float(best_acc)}

    def _eval_sequential(self, X, y, driver_name):
        """Sequential eval: predict episode t using context[0..t-1]."""
        device = torch.device(self.cfg.device)
        if len(np.unique(y)) < 2:
            return None, None, None
        probs, labels, ctx_sizes = [], [], []
        for t in range(1, len(X)):
            ctx_dict = convert_to_pairwise(
                X[:t], y[:t], driver_name, self.cfg.context_size, self.cfg.tie_ratio)
            if ctx_dict is None or len(ctx_dict["labels"]) == 0:
                continue
            loader = DataLoader(PreferenceDataset(ctx_dict), batch_size=len(ctx_dict["labels"]), shuffle=False)
            means = []
            with torch.no_grad():
                for b in loader:
                    m, _ = self.model.encode(b["observations"].to(device).float(),
                                              b["observations_2"].to(device).float(),
                                              b["labels"].to(device).float())
                    means.append(m.cpu().numpy())
            z = np.mean(np.concatenate(means, axis=0), axis=0)
            step_rewards = compute_step_rewards(self.model, X[t:t+1], z, device)
            probs.append(torch.sigmoid(torch.tensor(step_rewards.mean(axis=1))).item())
            labels.append(int(y[t]))
            ctx_sizes.append(t)
        return (np.array(probs), np.array(labels), ctx_sizes) if probs else (None, None, None)

    def evaluate(self, out_dir: Path) -> dict:
        self.model.eval()
        all_metrics = {}
        device = torch.device(self.cfg.device)
        viz_dir = out_dir / "plots" / "viz"

        print("[2] Evaluating test drivers...")
        for driver_name, (X, y) in self.test_driver_data.items():
            p, l, ctx = self._eval_sequential(X, y, driver_name)
            if p is None:
                continue
            seq_x, seq_y = compute_sequential_aurocs(p.tolist(), l.tolist(), ctx)
            plot_sequential_auroc(seq_x, seq_y, out_dir / "plots", driver_name)
            m = evaluate_predictions(l, p, out_dir / "plots", driver_name,
                                      title=f"VPL Pairwise - {driver_name} (test)")
            all_metrics[f"test/{driver_name}"] = m
            print(f"  {driver_name}: AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}  Brier={m['brier']:.4f}")

        z_by_driver = collect_z_pairwise(self.model, self.train_dataset_dict, device)
        plot_z_space(z_by_driver, list(self.test_driver_data.keys()), viz_dir / "z_space.png")
        plot_z_dims(z_by_driver, viz_dir / "z_dims.png")
        plot_reward_dist(self.model, self.train_driver_data, z_by_driver, device, viz_dir)

        save_metrics_txt(all_metrics, out_dir / "metrics.txt")
        return all_metrics

    def save(self, out_dir: Path) -> None:
        pass  # VPLTrainer가 best_model.pt 저장

    def load(self, out_dir: Path) -> None:
        self.model.load_state_dict(
            torch.load(out_dir / "best_model.pt", map_location=self.cfg.device, weights_only=True))

    def make_summary(self, train_metrics: dict, eval_metrics: dict) -> dict:
        s = super().make_summary(train_metrics, eval_metrics)
        s["test_drivers"] = self.cfg.test_driver_names
        return s
