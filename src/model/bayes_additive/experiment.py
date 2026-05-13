import json
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import asdict
from sklearn.metrics import roc_auc_score

from ..experiment import BaseExperiment
from ...data.splits import load_sequences
from ...evaluation import evaluate_predictions, save_metrics_txt, plot_sequential_auroc
from .features import EpisodeFeatureExtractor, SoftBasisTransformer
from .model import OnlineBayesianLogisticReward, PrototypeBayesianLogisticReward, sigmoid


class BayesAdditiveExperiment(BaseExperiment):
    """Interpretable population prior with online user-specific reward posterior."""

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
        model_cls = PrototypeBayesianLogisticReward if cfg.prior_mode == "prototype" else OnlineBayesianLogisticReward
        model_kwargs = dict(
            C=cfg.C,
            penalty=cfg.penalty,
            prior_var_floor=cfg.prior_var_floor,
            user_var_scale=cfg.user_var_scale,
            update_temperature=cfg.update_temperature,
            map_max_iter=cfg.map_max_iter,
            map_tol=cfg.map_tol,
            map_use_full_cov=cfg.map_use_full_cov,
            random_state=cfg.seed,
        )
        if model_cls is PrototypeBayesianLogisticReward:
            model_kwargs.update(
                prototype_shrinkage=cfg.prototype_shrinkage,
                component_var_scale=cfg.component_var_scale,
                component_temperature=cfg.component_temperature,
                global_component_weight=cfg.global_component_weight,
            )
        self.model = model_cls(**model_kwargs)

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

        self._log("[3] Fitting population prior...")
        stats = self.model.fit_population(
            Phi_by_user, y_by_user, self.basis.feature_names,
            user_names=cfg.train_driver_names)

        Phi_all = np.concatenate(Phi_by_user, axis=0)
        y_all = np.concatenate(y_by_user, axis=0)
        self.model.reset_user()
        probs = self.model.predict(Phi_all)
        train_auroc = roc_auc_score(y_all, probs) if len(np.unique(y_all)) > 1 else float("nan")
        self._log(f"  Population train AUROC={train_auroc:.4f}")
        self._write_train_user_profiles(out_dir, Phi_by_user, y_by_user, cfg.train_driver_names)
        self._plot_train_user_predictions(out_dir / "plots" / "train", Phi_by_user, y_by_user, cfg.train_driver_names)
        self._plot_train_profile_heatmap(
            out_dir / "plots" / "train" / "user_profile_heatmap.png",
            Phi_by_user, cfg.train_driver_names)
        self._plot_prototype_similarity(out_dir / "plots" / "prototypes" / "prototype_similarity.png")

        with open(out_dir / "cfg.json", "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

        return {
            "population/train_auroc": float(train_auroc),
            "population/train_brier": stats["population_brier"],
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
            prior_weights = self.model.component_weights() if hasattr(self.model, "component_weights") else None
            prior_probs = self.model.predict(holdout_X)
            ctx_sizes, aurocs, weight_history = [], [], []
            final_probs = prior_probs
            for t in range(1, split_idx + 1):
                self.model.fit_user_map(ctx_X[:t], ctx_y[:t], warm_start=True)
                if hasattr(self.model, "component_weights"):
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
                plots_dir / "prototypes" / f"prior_posterior_contributions_{tname}.png")
            self._plot_prototype_weights(
                prior_weights, plots_dir / "prototypes" / f"prototype_weights_{tname}.png")
            self._plot_prototype_weight_evolution(
                ctx_sizes=list(range(1, len(weight_history) + 1)),
                weight_history=weight_history,
                save_path=plots_dir / "prototypes" / f"prototype_weight_evolution_{tname}.png")

            drivers.append(dict(
                name=tname, holdout_X=holdout_X, holdout_y=holdout_y,
                final_probs=final_probs, prior_probs=prior_probs,
                ctx_sizes=ctx_sizes, aurocs=aurocs,
            ))

        names = [d["name"] for d in drivers]
        ys = [d["holdout_y"] for d in drivers]
        ms_online = evaluate_predictions(
            ys, [d["final_probs"] for d in drivers], plots_dir, names,
            save_name="metrics_online", title="BayesAdditive (online posterior)")
        ms_prior = evaluate_predictions(
            ys, [d["prior_probs"] for d in drivers], plots_dir, names,
            save_name="metrics_prior", title="BayesAdditive (population prior)")
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

        self._write_component_posterior(out_dir / "prototype_posterior.txt")
        save_metrics_txt(all_metrics, out_dir / "metrics.txt")
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
        state = obj["model"]
        if state.get("model_class") == "PrototypeBayesianLogisticReward":
            self.model = PrototypeBayesianLogisticReward.from_state_dict(state)
        else:
            self.model = OnlineBayesianLogisticReward.from_state_dict(state)

    def make_summary(self, train_metrics: dict, eval_metrics: dict) -> dict:
        s = super().make_summary(train_metrics, eval_metrics)
        s["test_drivers"] = list(self.cfg.test_driver_names)
        s["features"] = list(self.cfg.features)
        s["time_range"] = list(self.cfg.time_range)
        return s

    def _top_groups(self, Phi, top_k=8):
        return self.basis.group_contributions(self.model.mean, Phi, self.model.cov, top_k=top_k)

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
        ax.set_title("User-Specific Reward Components")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

    def _component_theta_for_name(self, name):
        if not hasattr(self.model, "prototype_names"):
            return self.model.prior_mean
        for idx, proto_name in enumerate(self.model.prototype_names):
            if proto_name == name:
                return self.model.component_prior_means[idx]
        return self.model.prior_mean

    def _plot_train_user_predictions(self, plots_dir, Phi_by_user, y_by_user, user_names):
        plots_dir = Path(plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)
        ys, probs_list = [], []
        for name, Phi, y in zip(user_names, Phi_by_user, y_by_user):
            theta = self._component_theta_for_name(name)
            ys.append(y)
            probs_list.append(sigmoid(np.asarray(Phi) @ theta))
        evaluate_predictions(ys, probs_list, plots_dir, user_names,
                             save_name="metrics", title="BayesAdditive Prototype (train)")

    def _group_contribution_dict(self, theta, Phi):
        rows = self.basis.group_contributions(theta, Phi, cov=None, top_k=10_000)
        return {name: signed for name, signed, _, _ in rows}

    def _plot_train_profile_heatmap(self, save_path, Phi_by_user, user_names, top_k=20):
        if not hasattr(self.model, "component_prior_means"):
            return

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
        ax.set_title("Historical User Preference Profiles")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center", fontsize=6)
        fig.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
        plt.close(fig)

    def _plot_prototype_similarity(self, save_path):
        if not hasattr(self.model, "component_prior_means"):
            return
        names = list(self.model.prototype_names)
        theta = self.model.component_prior_means.copy()
        if len(names) <= 1:
            return
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
        ax.set_title("Prototype Preference Similarity")
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
        plt.close(fig)

    def _plot_prior_posterior_groups(self, Phi, prior_mean, posterior_mean, save_path, top_k=16):
        prior = self._group_contribution_dict(prior_mean, Phi)
        post = self._group_contribution_dict(posterior_mean, Phi)
        groups = sorted(set(prior) | set(post), key=lambda g: max(abs(prior.get(g, 0.0)), abs(post.get(g, 0.0))), reverse=True)
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
        ax.set_title("Target User Prior vs Posterior Preference Contributions")
        ax.legend()
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
        plt.close(fig)

    def _plot_prototype_weights(self, prior_weights, save_path):
        if not hasattr(self.model, "component_weights"):
            return
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
        ax.set_title("Prototype Assignment: Prior vs Posterior")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
        plt.close(fig)

    def _plot_prototype_weight_evolution(self, ctx_sizes, weight_history, save_path):
        if not weight_history or not hasattr(self.model, "prototype_names"):
            return
        weights = np.stack(weight_history, axis=0)
        names = list(self.model.prototype_names)
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(names), 1)))

        fig, ax = plt.subplots(figsize=(10, 4.5))
        for idx, name in enumerate(names):
            ax.plot(ctx_sizes, weights[:, idx], lw=1.6, color=colors[idx], label=name)
        ax.set_xlabel("Context size")
        ax.set_ylabel("Posterior prototype weight")
        ax.set_title("Prototype Posterior Evolution")
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
            "Bayesian Sparse Additive Preference Summary",
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
            "  This is not a hard rule list. Each row is a smooth additive component",
            "  whose coefficient posterior was initialized from other drivers and",
            "  updated online using this user's context feedback.",
        ]
        Path(path).write_text("\n".join(lines), encoding="utf-8")

    def _write_train_user_profiles(self, out_dir, Phi_by_user, y_by_user, user_names):
        if not hasattr(self.model, "component_prior_means"):
            return

        phi_by_name = {name: Phi for name, Phi in zip(user_names, Phi_by_user)}
        y_by_name = {name: y for name, y in zip(user_names, y_by_user)}
        Phi_all = np.concatenate(Phi_by_user, axis=0)
        y_all = np.concatenate(y_by_user, axis=0)

        profiles = []
        lines = [
            "Historical User Preference Profiles",
            "=" * 52,
            "",
        ]
        for name, theta in zip(self.model.prototype_names, self.model.component_prior_means):
            Phi = phi_by_name.get(name, Phi_all)
            y = y_by_name.get(name, y_all)
            probs = sigmoid(Phi @ theta)
            auroc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else float("nan")
            rows = self.basis.group_contributions(theta, Phi, cov=None, top_k=10)
            rec = {
                "name": name,
                "train_auroc": float(auroc),
                "top_components": [
                    {"group": r[0], "signed": r[1], "abs": r[2]} for r in rows
                ],
            }
            profiles.append(rec)

            lines += [
                f"[{name}] train AUROC={auroc:.4f}",
            ]
            for group, signed, abs_val, _ in rows:
                direction = "positive" if signed > 0 else "negative"
                lines.append(f"- {group:<35} {signed:+.4f}  abs={abs_val:.4f}  ({direction})")
            lines.append("")

        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "train_user_profiles.txt").write_text("\n".join(lines), encoding="utf-8")
        with open(out_dir / "train_user_profiles.json", "w", encoding="utf-8") as f:
            json.dump(profiles, f, ensure_ascii=False, indent=2)

    def _write_component_posterior(self, path):
        if not hasattr(self.model, "component_summary"):
            return
        rows = self.model.component_summary()
        if not rows:
            return
        lines = [
            "Prototype Posterior",
            "=" * 52,
            "",
        ]
        rows = sorted(rows, key=lambda r: r["weight"], reverse=True)
        for row in rows:
            lines.append(f"- {row['name']:<20} weight={row['weight']:.4f}  fit_score={row['fit_score']:.4f}")
        Path(path).write_text("\n".join(lines), encoding="utf-8")
