import numpy as np
import matplotlib.pyplot as plt

from models.fully_bayesian.features import FNS

C = {"physics": "#eb6834", "kalman": "#eda100", "fir": "#1baf7a", "unet": "#2a78d6"}
INK, MUTED = "#0b0b0b", "#52514e"
BOUNCE_STATS = ["impulse_abs", "abs_peak_deriv", "p2p_deriv", "wrms_z_deriv"]

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5})

corr = lambda a, b: np.corrcoef(a, b)[0, 1]


def waveform_r(pred, y):
    return np.array([[corr(pred[i, j], y[i, j]) for j in range(3)] for i in range(len(y))])


def plot_overlay(y, preds, R, ids, path, show, channels, fs):
    main = show[-1]
    t = np.arange(y.shape[2]) / fs
    order = np.argsort(R[main].mean(1))
    picks = {"worst": order[0], "median": order[len(order) // 2], "best": order[-1]}
    fig, axes = plt.subplots(3, 3, figsize=(14, 8), sharex=True)
    for row, (tag, i) in enumerate(picks.items()):
        for col in range(3):
            ax = axes[row, col]
            ax.plot(t, y[i, col], color=INK, lw=1.0, label="true")
            for m in show:
                ax.plot(t, preds[m][i, col], color=C[m], lw=1.3 if m == main else 0.8, label=m)
            ax.text(0.02, 0.95, f"{main} r={R[main][i, col]:.3f}", transform=ax.transAxes,
                    va="top", fontsize=8, color=MUTED)
            if row == 0:
                ax.set_title(channels[col])
            if col == 0:
                ax.set_ylabel(f"{tag}\n{ids[i]}", fontsize=8)
            if row == 2:
                ax.set_xlabel("time [s]")
    axes[0, -1].legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.suptitle("6D reconstruction on held-out drivers (standardized)", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_r_distribution(R, path, show, channels):
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for j, ax in enumerate(axes):
        for k, m in enumerate(show):
            vals = R[m][:, j]
            ax.scatter(rng.uniform(k - 0.15, k + 0.15, len(vals)), vals, s=8, color=C[m], alpha=0.35, lw=0)
            med = np.median(vals)
            ax.plot([k - 0.25, k + 0.25], [med, med], color=INK, lw=1.5)
            ax.text(k + 0.28, med, f"{med:.3f}", va="center", fontsize=8, color=INK)
        ax.set_title(channels[j])
        ax.set_xticks(range(len(show)), show)
        ax.set_xlim(-0.5, len(show) - 0.2)
    axes[0].set_ylabel("waveform r (per episode)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_stats_scatter(y, preds, path, show, fs):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    for stat, ax in zip(BOUNCE_STATS, axes):
        tv = FNS[stat](y[:, 0], fs)
        ax.plot([tv.min(), tv.max()], [tv.min(), tv.max()], color=MUTED, lw=0.8, ls="--")
        for m in show:
            pv = FNS[stat](preds[m][:, 0], fs)
            ax.scatter(tv, pv, s=12, color=C[m], alpha=0.5, lw=0, label=f"{m} r={corr(tv, pv):.3f}")
        ax.set_title(stat)
        ax.set_xlabel("true")
        ax.legend(fontsize=7, loc="upper left")
    axes[0].set_ylabel("recon")
    fig.suptitle("bounce stats: true vs reconstructed (test episodes)", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
