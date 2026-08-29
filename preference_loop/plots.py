import matplotlib.pyplot as plt
import numpy as np

from preference_loop.data import sigmoid
from reward.fully_bayesian import viz


ROLE_COLOR = {"train": "#4c78a8", "test": "#e45756"}
ROLE_MARKER = {"train": "o", "test": "s"}
OPTIMIZER_STYLE = {
    "grid": {"color": "#f2a541", "ls": "-.", "label": "Grid"},
    "cmaes": {"color": "#e45756", "ls": "--", "label": "CMA-ES"},
}


def save_figure(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def user_axes(n_users, height=2.8, sharex=False, sharey=False):
    ncols = 3
    nrows = int(np.ceil(n_users / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, height * nrows),
        squeeze=False, sharex=sharex, sharey=sharey,
    )
    flat = axes.ravel()
    for ax in flat[n_users:]:
        ax.remove()
    return fig, flat[:n_users]


def plot_feedback_balance(cfg, data, path):
    names = data["user_names"]
    expected = np.asarray([data["feedback"][name]["expected_good"] for name in names])
    observed = np.asarray([data["feedback"][name]["observed_good"] for name in names])
    x = np.arange(len(names))
    colors = [ROLE_COLOR[data["roles"][name]] for name in names]
    fig, ax = plt.subplots(figsize=(max(10.0, 0.72 * len(names)), 4.5))
    lo, hi = cfg.feedback_ratio_range
    ax.axhspan(lo, hi, color="0.5", alpha=0.08, label=f"accepted {lo:.1f}-{hi:.1f}")
    ax.bar(x, observed, color=colors, alpha=0.82, label="observed good")
    ax.scatter(x, expected, color="black", marker="D", s=22, zorder=3, label="expected good")
    ax.axhline(0.5, color="0.35", ls=":", lw=1.0)
    ax.set_xticks(x, [name.replace("user_", "u") for name in names], rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Good feedback ratio")
    ax.set_title("Independent random-scenario feedback per user")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8)
    save_figure(fig, path)


def plot_user_sigmoid(env, data, path):
    names = data["user_names"]
    fig, axes = user_axes(len(names), height=3.0)
    weight_indices = np.arange(env.parameter_dim) != env.bias_index
    for i, (ax, name) in enumerate(zip(axes, names)):
        theta = data["theta_true"][i]
        feedback = data["feedback"][name]
        rewards = feedback["rewards"]
        probabilities = feedback["probabilities"]
        labels = feedback["labels"]
        pad = max(0.05 * np.ptp(rewards), 1.0)
        reward_grid = np.linspace(rewards.min() - pad, rewards.max() + pad, 400)
        centered = reward_grid + theta[env.bias_index]

        ax.scatter(
            rewards[labels == 0], probabilities[labels == 0],
            color="#d62728", alpha=0.18, s=7, label="bad (0)",
        )
        ax.scatter(
            rewards[labels == 1], probabilities[labels == 1],
            color="#2ca02c", alpha=0.18, s=7, label="good (1)",
        )
        ax.plot(
            reward_grid, sigmoid(0.5 * centered),
            color="#1f77b4", ls="--", lw=1.0, label="0.5x parameter scale",
        )
        ax.plot(reward_grid, sigmoid(centered), color="black", lw=1.8, label="current parameter scale")
        ax.plot(
            reward_grid, sigmoid(2.0 * centered),
            color="#ff7f0e", ls=":", lw=1.0, label="2x parameter scale",
        )
        ax.axvline(-theta[env.bias_index], color="0.5", ls=":", lw=0.8)
        ax.set_title(
            f"[{data['roles'][name]}] {name}  "
            f"||w||={np.linalg.norm(theta[weight_indices]):.2f}\n"
            f"good={labels.mean():.2f}",
            fontsize=8,
        )
        ax.set_xlabel("episode reward R")
        ax.set_ylabel("p(good)")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.2)
        if i == 0:
            ax.legend(fontsize=6.5)
    fig.suptitle("Per-user reward-to-feedback sigmoid", y=1.002)
    save_figure(fig, path)


def plot_controller_recovery(cfg, env, data, result, directory):
    if env.controller_dim != 1:
        raise ValueError("Controller landscape plot requires one parameter")
    groups = [(method,) for method in result["optimizer_names"]]
    if len(result["optimizer_names"]) > 1:
        groups.append(tuple(result["optimizer_names"]))
    for methods in groups:
        tag = methods[0] if len(methods) == 1 else "combined"
        labels = " & ".join(OPTIMIZER_STYLE[m]["label"] for m in methods)
        for role in ("train", "test"):
            names = [name for name in data["user_names"] if data["roles"][name] == role]
            fig, axes = user_axes(len(names), height=2.9, sharex=True)
            for ax, name in zip(axes, names):
                evaluation = result["users"][name]
                grid = evaluation["evaluation"]["candidates"][:, 0]
                candidate = evaluation["candidate"]
                if cfg.visualize_oracle:
                    monte_carlo = candidate["true_mean"]
                    std = candidate["true_std"]
                    ax.fill_between(
                        grid, monte_carlo - std, monte_carlo + std,
                        color="0.5", alpha=0.12, label="true random-scenario +/-1 sd",
                    )
                    ax.plot(grid, monte_carlo, "o-", color="black", ms=2.7, lw=1.4, label="true grid mean")
                ax.fill_between(
                    grid, candidate["estimate_low"], candidate["estimate_high"],
                    color="#4c78a8", alpha=0.20, label="posterior 90% CI",
                )
                ax.plot(
                    grid, candidate["estimate_mean"], "o-",
                    color="#4c78a8", ms=2.7, lw=1.2, label="posterior mean",
                )
                true_best = np.asarray(evaluation["true_best_controller"])[0]
                ax.axvline(true_best, color="#2ca02c", ls=":", lw=1.1, label="true best")
                for method in methods:
                    style = OPTIMIZER_STYLE[method]
                    optimization = evaluation["controllers"][method]
                    if cfg.visualize_oracle:
                        ax.axvline(
                            optimization["oracle_parameters"][0], color="0.35",
                            ls="--", lw=1.1, label="oracle controller",
                        )
                    ax.axvline(
                        optimization["parameters"][0], color=style["color"], ls=style["ls"],
                        lw=1.1, label=f"{style['label']} controller",
                    )
                estimates = ", ".join(
                    f"{OPTIMIZER_STYLE[m]['label']}="
                    f"{evaluation['controllers'][m]['parameters'][0]:.1f}"
                    for m in methods
                )
                regrets = ", ".join(
                    f"{m}={evaluation['controllers'][m]['regret']:.3f}" for m in methods
                )
                ax.set_title(
                    f"{name}: true={true_best:.1f}, {estimates}\nregret: {regrets}",
                    fontsize=8,
                )
                ax.set_xlabel(env.controller_names[0])
                ax.set_ylabel("feedback logit")
                ax.set_xlim(grid[0], grid[-1])
                ax.grid(alpha=0.2)
            axes[0].legend(fontsize=6.2)
            fig.suptitle(
                f"{labels} controllers for {role} users on random "
                f"scenarios ({cfg.n_evaluation_scenarios} per candidate)",
                y=1.002,
            )
            save_figure(fig, directory / f"controller_recovery_{tag}_{role}.png")


def plot_parameter_recovery(env, data, result, path):
    names = data["user_names"]
    theta_true = data["theta_true"]
    theta_hat = result["theta_hat"]
    fig, axes = plt.subplots(
        1, env.parameter_dim, figsize=(4.2 * env.parameter_dim, 3.8), squeeze=False,
    )
    for j, (ax, label) in enumerate(zip(axes.ravel(), env.parameter_labels)):
        lo = float(min(theta_true[:, j].min(), theta_hat[:, j].min()))
        hi = float(max(theta_true[:, j].max(), theta_hat[:, j].max()))
        pad = max(0.08 * (hi - lo), 0.5)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", lw=1.0)
        for role in ("train", "test"):
            index = np.asarray([data["roles"][name] == role for name in names])
            ax.scatter(
                theta_true[index, j], theta_hat[index, j],
                color=ROLE_COLOR[role], marker=ROLE_MARKER[role], s=36, label=role,
            )
        for name, x, y in zip(names, theta_true[:, j], theta_hat[:, j]):
            ax.annotate(name[-2:], (x, y), xytext=(3, 2), textcoords="offset points", fontsize=6)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("true parameter")
        ax.set_ylabel("posterior mean")
        ax.set_title(label)
        ax.grid(alpha=0.2)
    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle("Train and personalized test-user parameter recovery", y=1.02)
    save_figure(fig, path)


def plot_population_recovery(env, data, population, path):
    train_theta = data["theta_true"][:len(data["train_names"])]
    cohort_mean = train_theta.mean(axis=0)
    cohort_cov = np.cov(train_theta, rowvar=False)
    fig, axes = plt.subplots(1, 4, figsize=(17.0, 3.8))
    x = np.arange(env.parameter_dim)
    width = 0.25
    axes[0].bar(x - width, env.population_mean, width, label="target MVN", color="black", alpha=0.75)
    axes[0].bar(x, cohort_mean, width, label="train cohort", color="#f2a541", alpha=0.85)
    axes[0].bar(x + width, population.mu_bar, width, label="posterior", color="#4c78a8", alpha=0.8)
    axes[0].set_xticks(x, env.parameter_labels, rotation=35, ha="right")
    axes[0].set_title("Population mean")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.2)

    vmax = float(max(
        np.abs(env.population_cov).max(),
        np.abs(cohort_cov).max(),
        np.abs(population.Sigma_bar).max(),
    ))
    matrices = (env.population_cov, cohort_cov, population.Sigma_bar)
    titles = ("Target MVN covariance", "Train-cohort covariance", "Posterior covariance")
    for ax, matrix, title in zip(axes[1:], matrices, titles):
        image = ax.imshow(matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_xticks(x, env.parameter_labels, rotation=40, ha="right", fontsize=7)
        ax.set_yticks(x, env.parameter_labels, fontsize=7)
        ax.set_title(title)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Population recovery from train users only", y=1.02)
    save_figure(fig, path)


def plot_preferred_controller_recovery(cfg, env, data, result, path):
    names = data["user_names"]
    methods = result["optimizer_names"]
    xlabel = "oracle" if cfg.visualize_oracle else "true preferred"
    fig, axes = plt.subplots(
        len(methods), env.controller_dim,
        figsize=(5.2 * env.controller_dim, 4.5 * len(methods)), squeeze=False,
    )
    for method_index, method in enumerate(methods):
        oracle = np.stack([
            result["users"][name]["controllers"][method]["oracle_parameters"]
            if cfg.visualize_oracle
            else result["users"][name]["true_best_controller"]
            for name in names
        ])
        estimated = np.stack([
            result["users"][name]["controllers"][method]["parameters"] for name in names
        ])
        for j, label in enumerate(env.controller_names):
            ax = axes[method_index, j]
            lo = float(min(oracle[:, j].min(), estimated[:, j].min()))
            hi = float(max(oracle[:, j].max(), estimated[:, j].max()))
            pad = 0.05 * max(hi - lo, 1.0)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", lw=1.0)
            for role in ("train", "test"):
                index = np.asarray([data["roles"][name] == role for name in names])
                ax.scatter(
                    oracle[index, j], estimated[index, j],
                    color=ROLE_COLOR[role], marker=ROLE_MARKER[role], s=44, label=role,
                )
            for name, x, y in zip(names, oracle[:, j], estimated[:, j]):
                ax.annotate(name[-2:], (x, y), xytext=(4, 2), textcoords="offset points", fontsize=7)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_xlabel(f"{xlabel} {label}")
            ax.set_ylabel(f"{OPTIMIZER_STYLE[method]['label']} {label}")
            ax.set_title("Personalized controller recovery")
            ax.grid(alpha=0.2)
    axes.ravel()[0].legend(fontsize=8)
    save_figure(fig, path)


def plot_posterior_quality(data, result, directory):
    for role in ("train", "test"):
        names = [n for n in data["user_names"] if data["roles"][n] == role]
        viz.evaluate_predictions(
            [data["feedback"][name]["labels"] for name in names],
            [result["users"][name]["probability_feedback"] for name in names],
            directory, names,
            save_name=f"posterior_quality_feedback_{role}",
            title=f"Posterior predictive on feedback episodes ({role} users)",
        )
        viz.evaluate_predictions(
            [result["users"][name]["evaluation"]["labels"] for name in names],
            [result["users"][name]["probability_evaluation"] for name in names],
            directory, names,
            save_name=f"posterior_quality_evaluation_{role}",
            title=f"Posterior predictive on held-out evaluation episodes ({role} users)",
        )


def plot_results(cfg, env, run, data, population, result):
    plot_feedback_balance(cfg, data, run.plots / "feedback_balance.png")
    plot_posterior_quality(data, result, run.plots)
    plot_user_sigmoid(env, data, run.plots / "user_sigmoid.png")
    plot_controller_recovery(cfg, env, data, result, run.plots)
    plot_parameter_recovery(env, data, result, run.plots / "parameter_recovery.png")
    plot_population_recovery(env, data, population, run.plots / "population_recovery.png")
    plot_preferred_controller_recovery(
        cfg, env, data, result, run.plots / "preferred_controller_recovery.png",
    )
