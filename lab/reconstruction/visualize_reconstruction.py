from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs" / "lab" / "reconstruction"

from models.fully_bayesian.features import FNS

C = {"physics": "#eb6834", "kalman": "#eda100", "kalman_nc": "#e87ba4", "fir": "#1baf7a",
     "unet": "#2a78d6", "waveunet": "#7b4ab5"}
# SHOW = ["physics", "kalman", "fir", "unet", "waveunet"]
SHOW = ["unet", "waveunet"]
INK, MUTED = "#0b0b0b", "#52514e"
Y_CH = ["Bounce_rate_6D", "Roll_rate_6D", "Pitch_rate_6D"]
BOUNCE_STATS = ["impulse_abs", "abs_peak_deriv", "p2p_deriv", "wrms_z_deriv"]
FS = 100.0

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5})

data = np.load(OUT / "reconstruction_predictions.npz")
y, ids = data["y"], data["ids"]
preds = {m: data[m] for m in SHOW}
t = np.arange(y.shape[2]) / FS

corr = lambda a, b: np.corrcoef(a, b)[0, 1]
R = {m: np.array([[corr(pr[i, j], y[i, j]) for j in range(3)] for i in range(len(y))])
     for m, pr in preds.items()}


def plot_overlay():
    order = np.argsort(R[SHOW[-1]].mean(1))
    picks = {"worst": order[0], "median": order[len(order) // 2], "best": order[-1]}
    fig, axes = plt.subplots(3, 3, figsize=(14, 8), sharex=True)
    for row, (tag, i) in enumerate(picks.items()):
        for col in range(3):
            ax = axes[row, col]
            ax.plot(t, y[i, col], color=INK, lw=1.0, label="true")
            for m in SHOW:
                ax.plot(t, preds[m][i, col], color=C[m], lw=1.3 if m == SHOW[-1] else 0.8, label=m)
            ax.text(0.02, 0.95, f"{SHOW[-1]} r={R[SHOW[-1]][i, col]:.3f}", transform=ax.transAxes,
                    va="top", fontsize=8, color=MUTED)
            if row == 0:
                ax.set_title(Y_CH[col])
            if col == 0:
                ax.set_ylabel(f"{tag}\n{ids[i]}", fontsize=8)
            if row == 2:
                ax.set_xlabel("time [s]")
    axes[0, -1].legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.suptitle("6D reconstruction on held-out drivers (standardized)", y=0.995)
    fig.tight_layout()
    fig.savefig(OUT / "overlay.png", dpi=150, bbox_inches="tight")


def plot_r_dist():
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for j, ax in enumerate(axes):
        for k, m in enumerate(SHOW):
            vals = R[m][:, j]
            ax.scatter(rng.uniform(k - 0.15, k + 0.15, len(vals)), vals, s=8, color=C[m], alpha=0.35, lw=0)
            med = np.median(vals)
            ax.plot([k - 0.25, k + 0.25], [med, med], color=INK, lw=1.5)
            ax.text(k + 0.28, med, f"{med:.3f}", va="center", fontsize=8, color=INK)
        ax.set_title(Y_CH[j])
        ax.set_xticks(range(len(SHOW)), SHOW)
        ax.set_xlim(-0.5, len(SHOW) - 0.2)
    axes[0].set_ylabel("waveform r (per episode)")
    fig.tight_layout()
    fig.savefig(OUT / "waveform_correlation_distribution.png", dpi=150, bbox_inches="tight")


def plot_stats_scatter():
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    for stat, ax in zip(BOUNCE_STATS, axes):
        tv = FNS[stat](y[:, 0], FS)
        ax.plot([tv.min(), tv.max()], [tv.min(), tv.max()], color=MUTED, lw=0.8, ls="--")
        for m in SHOW:
            pv = FNS[stat](preds[m][:, 0], FS)
            ax.scatter(tv, pv, s=12, color=C[m], alpha=0.5, lw=0, label=f"{m} r={corr(tv, pv):.3f}")
        ax.set_title(stat)
        ax.set_xlabel("true")
        ax.legend(fontsize=7, loc="upper left")
    axes[0].set_ylabel("recon")
    fig.suptitle("bounce stats: true vs reconstructed (test episodes)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "stats_scatter.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    plot_overlay()
    plot_r_dist()
    plot_stats_scatter()
    print(f"saved: overlay.png, waveform_correlation_distribution.png, stats_scatter.png -> {OUT}")
