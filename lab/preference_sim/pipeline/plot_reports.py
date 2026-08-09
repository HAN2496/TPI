from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .reward_features import FEATURE_NAMES


SCATTER_PAIRS = (
    ("speed_mean", "energy_mean"),
    ("clearance_mean", "cadence"),
    ("airtime_frac", "bounce_rms"),
)


def plot_run_reports(run_dir: Path) -> Path:
    episodes = pd.read_csv(run_dir / "tables" / "episodes.csv")
    users = pd.read_csv(run_dir / "tables" / "users.csv")
    feature_columns = [f"feature_{name}" for name in FEATURE_NAMES]
    episodes[feature_columns].corr().to_csv(run_dir / "reports" / "feature_correlation.csv")

    profiles = list(dict.fromkeys(episodes["profile"].astype(str)))
    colors = plt.get_cmap("tab20")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)
    for profile_index, profile in enumerate(profiles):
        subset = episodes[episodes["profile"] == profile]
        color = colors(profile_index % 20)
        for ax, (x_name, y_name) in zip(axes.flat, SCATTER_PAIRS):
            ax.scatter(
                subset[f"feature_{x_name}"], subset[f"feature_{y_name}"],
                s=22, alpha=0.7, color=color, label=profile,
            )
    for ax, (x_name, y_name) in zip(axes.flat, SCATTER_PAIRS):
        ax.set(xlabel=x_name, ylabel=y_name, title=f"{x_name} vs. {y_name}")
    axes[0, 0].legend(frameon=False, fontsize=8)

    archetype_rates = users.groupby("archetype", sort=False)["positive_rate_all"].agg(["mean", "std"])
    axes[1, 1].bar(
        archetype_rates.index,
        archetype_rates["mean"],
        yerr=archetype_rates["std"].fillna(0.0),
        color=[colors(i % 20) for i in range(len(archetype_rates))],
        alpha=0.8,
        capsize=3,
    )
    axes[1, 1].set(
        xlabel="synthetic-user archetype",
        ylabel="positive-feedback rate",
        ylim=(0.0, 1.0),
        title="Feedback balance",
    )
    axes[1, 1].tick_params(axis="x", rotation=25)
    for ax in axes.flat:
        ax.grid(alpha=0.2)

    output = run_dir / "reports" / "trajectory_tradeoffs.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output
