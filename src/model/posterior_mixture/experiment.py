import json
import re
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import asdict
from sklearn.metrics import roc_auc_score

from ..experiment import BaseExperiment
from ..bayes_additive.features import EpisodeFeatureExtractor, SoftBasisTransformer
from ..bayes_additive.model import sigmoid
from ...data.splits import load_sequences
from ...evaluation import evaluate_predictions, save_metrics_txt, plot_sequential_auroc
from .model import UserPosteriorMixtureBayesianLogisticReward

_CHANNELS = [
    "long_acc", "lat_acc", "vertical_acc", "yaw_rate", "imu_roll_rate",
    "pitch_rate", "bounce_rate", "roll_rate", "steer_angle", "steer_speed",
    "accel_pedal", "brake_force",
]


class PosteriorMixtureExperiment(BaseExperiment):
    """User-Posterior Mixture Bayesian Reward Model.

    Train users are represented by Bayesian reward posteriors q_u(theta)=N(m_u, S_u);
    the target user's prior is a mixture over these user posteriors (plus an optional
    population component) inflated by transfer variance. Online context updates the
    posterior of each component and re-weights them via diagonal Laplace evidence.
    """

    def build(self):
        cfg = self.cfg
        self._smooth_kw = {
            "smooth": cfg.smooth,
            "smooth_cutoff": cfg.smooth_cutoff,
            "smooth_order": cfg.smooth_order,
        }
        self._log("[1] Loading train drivers...")
        self.train_data = {}
        for name in cfg.train_driver_names:
            X, y = load_sequences(name, cfg.features, cfg.time_range, cfg.downsample, **self._smooth_kw)
            self.train_data[name] = (X.astype(np.float32), y.astype(np.int64))
            self._log(f"  - {name}: n={len(y)}, pos={int(y.sum())}, neg={int((1-y).sum())}")

        self.test_data = {}
        for tname in cfg.test_driver_names:
            X, y = load_sequences(tname, cfg.features, cfg.time_range, cfg.downsample, **self._smooth_kw)
            self.test_data[tname] = (X.astype(np.float32), y.astype(np.int64))

        duration = float(cfg.time_range[1] - cfg.time_range[0])
        self.extractor = EpisodeFeatureExtractor(
            cfg.features, duration=duration, n_subwindows=cfg.n_subwindows)
        self.basis = SoftBasisTransformer(knots=cfg.knots, include_below=cfg.include_below)
        self.model = UserPosteriorMixtureBayesianLogisticReward(
            C=cfg.C,
            penalty=cfg.penalty,
            prior_var_floor=cfg.prior_var_floor,
            user_var_scale=cfg.user_var_scale,
            update_temperature=cfg.update_temperature,
            map_max_iter=cfg.map_max_iter,
            map_tol=cfg.map_tol,
            map_use_full_cov=cfg.map_use_full_cov,
            prototype_shrinkage=cfg.prototype_shrinkage,
            component_temperature=cfg.component_temperature,
            global_component_weight=cfg.global_component_weight,
            transfer_var_scale=cfg.transfer_var_scale,
            posterior_var_floor=cfg.posterior_var_floor,
            component_var_floor=cfg.component_var_floor,
            random_state=cfg.seed,
        )

    def train(self, out_dir: Path) -> dict:
        cfg = self.cfg
        self._log("[2] Building interpretable basis...")
        raw_by_user, y_by_user = [], []
        for name in cfg.train_driver_names:
            X, y = self.train_data[name]
            raw, raw_names = self.extractor.transform(X)
            raw_by_user.append(raw)
            y_by_user.append(y)

        raw_all = np.concatenate(raw_by_user, axis=0)
        self.basis.fit(raw_all, raw_names)
        Phi_by_user = [self.basis.transform(raw) for raw in raw_by_user]
        self._log(f"  Raw features={raw_all.shape[1]}, basis features={Phi_by_user[0].shape[1]}")

        self._log("[3] Fitting train-user posteriors and population prior...")
        self.model.fit_population(
            Phi_by_user, y_by_user, self.basis.feature_names,
            user_names=cfg.train_driver_names)

        Phi_all = np.concatenate(Phi_by_user, axis=0)
        y_all = np.concatenate(y_by_user, axis=0)
        self.model.reset_user()
        probs = self.model.predict(Phi_all)
        train_auroc = roc_auc_score(y_all, probs) if len(np.unique(y_all)) > 1 else float("nan")
        train_brier = float(np.mean((probs - y_all) ** 2))
        self._log(f"  Posterior-mixture train AUROC={train_auroc:.4f}")

        self._write_train_user_profiles(out_dir, Phi_by_user, y_by_user, cfg.train_driver_names)
        self._plot_train_user_predictions(out_dir / "plots" / "train", Phi_by_user, y_by_user, cfg.train_driver_names)
        self._plot_train_profile_heatmap(
            out_dir / "plots" / "train" / "user_profile_heatmap.png",
            Phi_by_user, cfg.train_driver_names)
        self._plot_posterior_similarity(out_dir / "plots" / "posterior" / "posterior_similarity.png")
        self._plot_posterior_distance(out_dir / "plots" / "posterior" / "user_posterior_kl.png")

        with open(out_dir / "cfg.json", "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

        return {
            "posterior_mixture/train_auroc": float(train_auroc),
            "posterior_mixture/train_brier": train_brier,
        }

    def evaluate(self, out_dir: Path) -> dict:
        cfg = self.cfg
        plots_dir = out_dir / "plots"
        drivers = []

        for tname in cfg.test_driver_names:
            self._log(f"[4] Online adaptation on {tname}...")
            X_test, y_test = self.test_data[tname]
            raw_test, _ = self.extractor.transform(X_test)
            Phi_test = self.basis.transform(raw_test)
            split_idx = len(Phi_test) // 2
            ctx_X, ctx_y = Phi_test[:split_idx], y_test[:split_idx]
            holdout_X, holdout_y = Phi_test[split_idx:], y_test[split_idx:]

            self.model.reset_user()
            prior_mean = self.model.mean.copy()
            prior_weights = self.model.component_weights().copy()
            prior_probs = self.model.predict(holdout_X)
            ctx_sizes, aurocs, weight_history = [], [], []
            final_probs = prior_probs
            for t in range(1, split_idx + 1):
                self.model.fit_user_map(ctx_X[:t], ctx_y[:t], warm_start=True)
                weight_history.append(self.model.component_weights().copy())
                probs = self.model.predict(holdout_X)
                final_probs = probs
                if len(np.unique(holdout_y)) > 1:
                    ctx_sizes.append(t)
                    aurocs.append(roc_auc_score(holdout_y, probs))

            posterior_mean = self.model.mean.copy()
            self._plot_top_groups(holdout_X, plots_dir / f"posterior_top_features_{tname}.png")
            self._plot_prior_posterior_groups(
                holdout_X, prior_mean, posterior_mean,
                plots_dir / "posterior" / f"prior_posterior_contributions_{tname}.png")
            self._plot_component_weights(
                prior_weights, plots_dir / "posterior" / f"component_weights_{tname}.png")
            self._plot_component_weight_evolution(
                ctx_sizes=list(range(1, len(weight_history) + 1)),
                weight_history=weight_history,
                save_path=plots_dir / "posterior" / f"component_weight_evolution_{tname}.png")
            self._plot_uncertainty_heatmap(
                plots_dir / "posterior" / f"uncertainty_heatmap_{tname}.png",
                test_name=tname, test_cov_diag=np.diag(self.model.cov))

            drivers.append(dict(
                name=tname, holdout_X=holdout_X, holdout_y=holdout_y,
                final_probs=final_probs, prior_probs=prior_probs,
                ctx_sizes=ctx_sizes, aurocs=aurocs,
                Phi_full=Phi_test,
            ))

        names = [d["name"] for d in drivers]
        ys = [d["holdout_y"] for d in drivers]
        ms_online = evaluate_predictions(
            ys, [d["final_probs"] for d in drivers], plots_dir, names,
            save_name="metrics_online", title="User-Posterior Mixture (online)")
        ms_prior = evaluate_predictions(
            ys, [d["prior_probs"] for d in drivers], plots_dir, names,
            save_name="metrics_prior", title="User-Posterior Mixture (prior)")
        plot_sequential_auroc(
            [d["ctx_sizes"] for d in drivers], [d["aurocs"] for d in drivers],
            plots_dir, names)

        all_metrics = {}
        for d, m, mp in zip(drivers, ms_online, ms_prior):
            all_metrics[f"test/{d['name']}"] = m
            all_metrics[f"test/{d['name']}_prior"] = mp
            self._write_preference_summary(
                out_dir / f"preference_summary_{d['name']}.txt", d["holdout_X"], m, mp)
            self._log(f"  [{d['name']}] Final AUROC={m['auroc']:.4f}  prior={mp['auroc']:.4f}")

        self._write_component_posterior(out_dir / "component_posterior.txt")
        save_metrics_txt(all_metrics, out_dir / "metrics.txt")

        train_Phi_by_user = []
        for name in cfg.train_driver_names:
            raw, _ = self.extractor.transform(self.train_data[name][0])
            train_Phi_by_user.append(self.basis.transform(raw))
        self._plot_item_embeddings(
            train_Phi_by_user, cfg.train_driver_names,
            [d["Phi_full"] for d in drivers], [d["name"] for d in drivers],
            plots_dir / "embeddings" / "item_emb.png",
        )
        return all_metrics

    def save(self, out_dir: Path) -> None:
        joblib.dump({
            "extractor": self.extractor,
            "basis": self.basis,
            "model": self.model.state_dict(),
        }, out_dir / "model.joblib")

    def load(self, out_dir: Path) -> None:
        obj = joblib.load(out_dir / "model.joblib")
        self.extractor = obj["extractor"]
        self.basis = obj["basis"]
        self.model = UserPosteriorMixtureBayesianLogisticReward.from_state_dict(obj["model"])

    def make_summary(self, train_metrics: dict, eval_metrics: dict) -> dict:
        s = super().make_summary(train_metrics, eval_metrics)
        s["test_drivers"] = list(self.cfg.test_driver_names)
        s["features"] = list(self.cfg.features)
        s["time_range"] = list(self.cfg.time_range)
        return s

    # ---------- helpers ----------

    def _top_groups(self, Phi, top_k=8):
        return self.basis.group_contributions(self.model.mean, Phi, self.model.cov, top_k=top_k)

    def _component_theta_for_name(self, name):
        for idx, proto_name in enumerate(self.model.prototype_names):
            if proto_name == name:
                return self.model.component_prior_means[idx]
        return self.model.prior_mean

    def _group_contribution_dict(self, theta, Phi, cov=None):
        rows = self.basis.group_contributions(theta, Phi, cov=cov, top_k=10_000)
        return {name: signed for name, signed, _, _ in rows}

    def _plot_top_groups(self, Phi, save_path):
        rows = self._top_groups(Phi, top_k=12)
        if not rows:
            return
        names = [r[0] for r in rows][::-1]
        vals = [r[1] for r in rows][::-1]
        errs = [r[3] if r[3] is not None else 0.0 for r in rows][::-1]
        colors = ["tomato" if v > 0 else "steelblue" for v in vals]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(names, vals, xerr=errs, color=colors, alpha=0.75)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Average posterior contribution to P(label=1) logit")
        ax.set_title("User-Posterior Mixture: Target Reward Components")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

    def _plot_train_user_predictions(self, plots_dir, Phi_by_user, y_by_user, user_names):
        plots_dir = Path(plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)
        ys, probs_list = [], []
        for name, Phi, y in zip(user_names, Phi_by_user, y_by_user):
            theta = self._component_theta_for_name(name)
            ys.append(y)
            probs_list.append(sigmoid(np.asarray(Phi) @ theta))
        evaluate_predictions(ys, probs_list, plots_dir, user_names,
                             save_name="metrics", title="User-Posterior Mixture (train)")

    def _plot_train_profile_heatmap(self, save_path, Phi_by_user, user_names, top_k=20):
        contribs = {}
        for name, Phi in zip(user_names, Phi_by_user):
            theta = self._component_theta_for_name(name)
            contribs[name] = self._group_contribution_dict(theta, Phi)

        groups = sorted({g for d in contribs.values() for g in d})
        if not groups:
            return
        score = {
            g: max(abs(contribs[name].get(g, 0.0)) for name in user_names)
            for g in groups
        }
        groups = sorted(groups, key=lambda g: score[g], reverse=True)[:top_k]
        mat = np.asarray([[contribs[name].get(g, 0.0) for g in groups] for name in user_names])
        vmax = max(float(np.max(np.abs(mat))), 1e-6)

        fig_w = max(10, 0.45 * len(groups))
        fig_h = max(4, 0.55 * len(user_names))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Average logit contribution")
        ax.set_xticks(np.arange(len(groups)))
        ax.set_yticks(np.arange(len(user_names)))
        ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(user_names, fontsize=9)
        ax.set_title("Train User Reward Posteriors")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center", fontsize=6)
        fig.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
        plt.close(fig)

    def _plot_posterior_similarity(self, save_path):
        names = list(self.model.prototype_names)
        if len(names) <= 1:
            return
        theta = self.model.component_prior_means.copy()
        centered = theta - self.model.prior_mean[None, :]
        norm = np.linalg.norm(centered, axis=1, keepdims=True)
        emb = centered / np.maximum(norm, 1e-8)
        sim = emb @ emb.T

        fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.9), max(5, len(names) * 0.8)))
        im = ax.imshow(sim, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Cosine similarity")
        ax.set_xticks(np.arange(len(names)))
        ax.set_yticks(np.arange(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_title("Posterior Component Mean Similarity")
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
        plt.close(fig)

    def _plot_posterior_distance(self, save_path):
        dist, names = self.model.posterior_distance_matrix()
        if dist is None:
            return
        fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.9), max(5, len(names) * 0.8)))
        im = ax.imshow(dist, cmap="viridis", aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Symmetric KL")
        ax.set_xticks(np.arange(len(names)))
        ax.set_yticks(np.arange(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_title("Train User Posterior Symmetric-KL Distance")
        vmax = float(dist.max())
        for i in range(len(names)):
            for j in range(len(names)):
                color = "white" if dist[i, j] < 0.6 * vmax else "black"
                ax.text(j, i, f"{dist[i, j]:.2f}", ha="center", va="center", fontsize=7, color=color)
        fig.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
        plt.close(fig)

    def _plot_prior_posterior_groups(self, Phi, prior_mean, posterior_mean, save_path, top_k=16):
        prior = self._group_contribution_dict(prior_mean, Phi)
        post = self._group_contribution_dict(posterior_mean, Phi)
        groups = sorted(
            set(prior) | set(post),
            key=lambda g: max(abs(prior.get(g, 0.0)), abs(post.get(g, 0.0))),
            reverse=True,
        )
        groups = groups[:top_k]
        if not groups:
            return
        y = np.arange(len(groups))
        prior_vals = np.asarray([prior.get(g, 0.0) for g in groups])
        post_vals = np.asarray([post.get(g, 0.0) for g in groups])

        fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(groups))))
        ax.barh(y + 0.18, prior_vals, height=0.34, color="gray", alpha=0.65, label="Prior")
        ax.barh(y - 0.18, post_vals, height=0.34, color="mediumseagreen", alpha=0.8, label="Posterior")
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(groups, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Average logit contribution")
        ax.set_title("Target Prior vs Posterior Reward Contributions")
        ax.legend()
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
        plt.close(fig)

    def _plot_component_weights(self, prior_weights, save_path):
        names = list(self.model.prototype_names)
        post_weights = self.model.component_weights()
        if prior_weights is None:
            prior_weights = np.full_like(post_weights, 1.0 / len(post_weights))
        x = np.arange(len(names))
        width = 0.38

        fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.1), 4.5))
        ax.bar(x - width / 2, prior_weights, width=width, color="gray", alpha=0.65, label="Prior")
        ax.bar(x + width / 2, post_weights, width=width, color="mediumseagreen", alpha=0.85, label="Posterior")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_ylim(0, max(1.0, float(max(np.max(prior_weights), np.max(post_weights))) * 1.15))
        ax.set_ylabel("Mixture weight")
        ax.set_title("Posterior Component Weights: Prior vs Posterior")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
        plt.close(fig)

    def _plot_component_weight_evolution(self, ctx_sizes, weight_history, save_path):
        if not weight_history:
            return
        weights = np.stack(weight_history, axis=0)
        names = list(self.model.prototype_names)
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(names), 1)))

        fig, ax = plt.subplots(figsize=(10, 4.5))
        for idx, name in enumerate(names):
            ax.plot(ctx_sizes, weights[:, idx], lw=1.6, color=colors[idx], label=name)
        ax.set_xlabel("Context size")
        ax.set_ylabel("Posterior component weight")
        ax.set_title("Posterior Component Weight Evolution")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
        plt.close(fig)

    def _write_preference_summary(self, path, Phi, final_metrics, prior_metrics):
        rows = self._top_groups(Phi, top_k=15)
        lines = [
            "User-Posterior Mixture Preference Summary",
            "=" * 52,
            f"final AUROC : {final_metrics['auroc']:.4f}",
            f"prior AUROC : {prior_metrics['auroc']:.4f}",
            "",
            "Top posterior reward components",
        ]
        for name, signed, abs_val, uncertainty in rows:
            direction = "raises label=1 probability" if signed > 0 else "lowers label=1 probability"
            unc = f"{uncertainty:.4f}" if uncertainty is not None else "n/a"
            lines.append(f"- {name:<35} {signed:+.4f}  abs={abs_val:.4f}  unc={unc}  ({direction})")

        lines += [
            "",
            "Interpretation note:",
            "  Train users are represented by Bayesian reward posteriors q_u=N(m_u,S_u).",
            "  The target prior is a mixture over these user posteriors (plus an optional",
            "  population component) inflated by transfer variance. Listed contributions",
            "  are mixture-posterior means with uncertainties from mixture variance.",
        ]
        Path(path).write_text("\n".join(lines), encoding="utf-8")

    def _write_train_user_profiles(self, out_dir, Phi_by_user, y_by_user, user_names):
        model = self.model
        phi_by_name = {name: Phi for name, Phi in zip(user_names, Phi_by_user)}
        y_by_name = {name: y for name, y in zip(user_names, y_by_user)}
        Phi_all = np.concatenate(Phi_by_user, axis=0)
        y_all = np.concatenate(y_by_user, axis=0)

        cov_by_name = {
            name: np.diag(np.maximum(cd, model.posterior_var_floor))
            for name, cd in zip(model.user_posterior_names, model.user_posterior_cov_diags)
        }

        profiles = []
        lines = [
            "Train User Reward Posteriors",
            "=" * 52,
            "",
        ]
        for name, theta in zip(model.prototype_names, model.component_prior_means):
            Phi = phi_by_name[name] if name in phi_by_name else Phi_all
            y = y_by_name[name] if name in y_by_name else y_all
            probs = sigmoid(Phi @ theta)
            auroc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else float("nan")
            cov = cov_by_name[name] if name in cov_by_name else None
            rows = self.basis.group_contributions(theta, Phi, cov=cov, top_k=10)

            profiles.append({
                "name": name,
                "train_auroc": float(auroc),
                "top_components": [
                    {"group": r[0], "signed": r[1], "abs": r[2], "uncertainty": r[3]}
                    for r in rows
                ],
            })

            lines.append(f"[{name}] train AUROC={auroc:.4f}")
            for group, signed, abs_val, unc in rows:
                direction = "positive" if signed > 0 else "negative"
                if unc is None:
                    lines.append(f"- {group:<35} {signed:+.4f}  abs={abs_val:.4f}  ({direction})")
                else:
                    lines.append(f"- {group:<35} {signed:+.4f}  abs={abs_val:.4f}  unc={unc:.4f}  ({direction})")
            lines.append("")

        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "train_user_profiles.txt").write_text("\n".join(lines), encoding="utf-8")
        with open(out_dir / "train_user_profiles.json", "w", encoding="utf-8") as f:
            json.dump(profiles, f, ensure_ascii=False, indent=2)

    def _write_component_posterior(self, path):
        rows = self.model.component_summary()
        if not rows:
            return
        lines = [
            "Posterior Component Summary",
            "=" * 52,
            "",
        ]
        rows = sorted(rows, key=lambda r: r["weight"], reverse=True)
        for row in rows:
            lines.append(f"- {row['name']:<20} weight={row['weight']:.4f}  fit_score={row['fit_score']:.4f}")
        Path(path).write_text("\n".join(lines), encoding="utf-8")

    # ---------- channel helpers ----------

    @staticmethod
    def _channel_of(group):
        if group == "bias":
            return None
        name = re.sub(r"^w\d+_", "", group)
        for ch in _CHANNELS:
            if name.startswith(ch + "_") or name == ch:
                return ch
        return "interactions"

    def _present_channels(self):
        if getattr(self, "_cached_channels", None) is not None:
            return self._cached_channels
        channels = []
        for g in self.basis.groups:
            ch = self._channel_of(g)
            if ch is None:
                continue
            if ch not in channels:
                channels.append(ch)
        self._cached_channels = channels
        return channels

    def _channel_aggregate_std(self, S_diag, channels):
        by_ch = {ch: [] for ch in channels}
        for j, g in enumerate(self.basis.groups):
            ch = self._channel_of(g)
            if ch is None or ch not in by_ch:
                continue
            by_ch[ch].append(np.sqrt(max(float(S_diag[j]), 0.0)))
        return np.array([np.mean(by_ch[ch]) if by_ch[ch] else 0.0 for ch in channels])

    # ---------- new visualizations ----------

    def _plot_uncertainty_heatmap(self, save_path, test_name, test_cov_diag):
        model = self.model
        train_names = list(model.user_posterior_names)
        train_cov_diags = model.user_posterior_cov_diags
        if train_cov_diags.shape[0] == 0:
            return
        channels = self._present_channels()
        rows = [self._channel_aggregate_std(cd, channels) for cd in train_cov_diags]
        row_labels = list(train_names)
        rows.append(self._channel_aggregate_std(np.asarray(test_cov_diag), channels))
        row_labels.append(f"[test] {test_name}")
        mat = np.asarray(rows)
        vmin = float(mat.min()) if mat.size else 0.0
        vmax = float(mat.max()) if mat.size else 1.0

        fig, ax = plt.subplots(figsize=(max(8, len(channels) * 0.95), max(4, len(row_labels) * 0.6)))
        im = ax.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"Avg $\sqrt{S}$ per channel")
        ax.set_xticks(np.arange(len(channels)))
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_xticklabels(channels, rotation=30, ha="right")
        ax.set_yticklabels(row_labels)
        ax.axhline(len(train_names) - 0.5, color="black", lw=2.0)
        threshold = vmin + 0.6 * (vmax - vmin)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                color = "white" if mat[i, j] > threshold else "black"
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=8, color=color)
        ax.set_title("Posterior Uncertainty by Feature Channel")
        fig.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
        plt.close(fig)

    def _plot_item_embeddings(self, train_Phi_by_user, train_names,
                              test_Phi_list, test_names, save_path):
        """Item-as-reward-vector embedding: r_i = (phi(x_i) @ m_u)_u over train users.

        Plays the role of CoPL's E_i visualization: items inherit a user-aware
        representation through per-user MAP reward scores, so items rated similarly
        by similar drivers cluster together.
        """
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE

        name_to_row = {n: i for i, n in enumerate(self.model.user_posterior_names)}
        M_ord = self.model.user_posterior_means[[name_to_row[n] for n in train_names]]

        train_embs = [np.asarray(Phi) @ M_ord.T for Phi in train_Phi_by_user]
        test_embs = [np.asarray(Phi) @ M_ord.T for Phi in test_Phi_list]
        E_train = np.concatenate(train_embs, axis=0)
        owner = np.concatenate([np.full(r.shape[0], i) for i, r in enumerate(train_embs)])
        n_train = E_train.shape[0]
        E_all = np.concatenate([E_train] + test_embs, axis=0)

        n = E_all.shape[0]
        perp = min(30, max(5, (n - 1) // 3))
        z_tsne = TSNE(n_components=2, random_state=42, perplexity=perp).fit_transform(E_all)
        z_pca = PCA(n_components=2).fit_transform(E_all)

        colors = [plt.cm.tab10(i) for i in range(len(train_names))]
        markers = ["*", "X", "P", "D"]
        test_colors = ["black", "red", "green", "purple"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, proj, title in [(axes[0], z_tsne, "t-SNE"), (axes[1], z_pca, "PCA")]:
            for uid, name in enumerate(train_names):
                m = owner == uid
                ax.scatter(proj[:n_train][m, 0], proj[:n_train][m, 1],
                           s=18, alpha=0.75, color=colors[uid], label=name)
            offset = n_train
            for ti, r in enumerate(test_embs):
                k = r.shape[0]
                ax.scatter(proj[offset:offset + k, 0], proj[offset:offset + k, 1],
                           s=40, alpha=0.9, color=test_colors[ti % len(test_colors)],
                           marker=markers[ti % len(markers)],
                           label=test_names[ti], zorder=5)
                offset += k
            ax.set_title(f"Item Embeddings (per-user reward) [{title}]")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
        plt.close(fig)

