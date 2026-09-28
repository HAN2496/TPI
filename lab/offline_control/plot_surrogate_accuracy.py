"""Held-out accuracy of the quadratic OLS and GP trajectory-feature surrogates.

No gain is optimized here. Models are the split-0 checkpoints of an existing
benchmark run, so the 200 audit episodes never entered surrogate fitting.
Run from repository root: python -m lab.offline_control.plot_surrogate_accuracy
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lab.offline_control.run_benchmark import simulate_gain
from preference_loop.optimization import SCENARIO_COVARIATES

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "font.size": 12, "axes.labelsize": 12, "axes.titlesize": 13, "xtick.labelsize": 11,
                     "ytick.labelsize": 11, "legend.fontsize": 11, "axes.unicode_minus": False})

STYLE = {"poly": ("2nd-order polynomial (OLS)", "#DC7E35", "--"), "gp": ("Gaussian process", "#3178B3", "-")}
FEATURES = ("pitch-rate feature", "longitudinal-acceleration feature")


def main(args):
    benchmark, output = Path(args.benchmark), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    source = Path(json.loads((benchmark / "manifest.json").read_text(encoding="utf-8"))["source"])
    audit_users = json.loads((benchmark / "split_0.json").read_text(encoding="utf-8"))["audit_users"]
    models = {kind: joblib.load(benchmark / f"{kind}_0.joblib") for kind in STYLE}
    with np.load(source / "preference_data.npz") as log:
        users = np.flatnonzero(np.isin(log["user_names"], audit_users))
        x = np.column_stack([log["feedback_controller"][users, :, 0].reshape(-1),
                             *[log[f"feedback_metadata_{key}"][users].reshape(-1) for key in SCENARIO_COVARIATES]])
        z = log["feedback_Z"][users, :, 1:].reshape(-1, 2)
        # Predeclared scenarios: first logged episode of the first three audit users.
        seeds = log["feedback_metadata_env_seed"][users[:3], 0]
    bounds = json.loads((source / "cfg.json").read_text(encoding="utf-8"))["controller_bounds"][0]
    scenarios = x[::10][:3, 1:]

    fig, axes = plt.subplots(2, 2, figsize=(6.6, 6.4), constrained_layout=True)
    for j, feature in enumerate(FEATURES):
        limit = [0, np.quantile(z[:, j], .995) * 1.05]
        for ax, (kind, model) in zip(axes[j], models.items()):
            label, color, _ = STYLE[kind]
            prediction = model.predict(x)[:, j]
            rmse = np.sqrt(np.mean((prediction - z[:, j]) ** 2))
            ax.plot(limit, limit, color="0.55", lw=1)
            ax.scatter(z[:, j], prediction, s=14, color=color, alpha=.6, edgecolors="none")
            ax.text(.04, .96, f"{feature[0].upper()}{feature[1:]}\nRMSE {rmse:.4f}", transform=ax.transAxes, va="top")
            ax.set(xlim=limit, ylim=limit, xlabel="Simulated", ylabel="Predicted")
            ax.grid(alpha=.2)
    for ax, (label, _, _) in zip(axes[0], STYLE.values()):
        ax.set_title(label)
    fig.suptitle(f"Held-out episodes ({len(z)})")
    fig.savefig(output / "surrogate_parity_held_out.png", dpi=200)
    plt.close(fig)
    if args.parity_only:  # 발표용 parity 그림만 다시 그릴 때: 아래 gain response 시뮬레이션 생략
        return

    gains = np.linspace(*bounds, args.gain_points)
    with ProcessPoolExecutor(args.workers) as executor:
        simulated = np.stack(list(executor.map(simulate_gain, gains, [seeds] * len(gains), [bounds] * len(gains))))
    np.savez_compressed(output / "surrogate_gain_response.npz", gains=gains, seeds=seeds, scenarios=scenarios, simulated=simulated)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, constrained_layout=True)
    for s, scenario in enumerate(scenarios):
        inputs = np.column_stack([gains, np.tile(scenario, (len(gains), 1))])
        for j, feature in enumerate(FEATURES):
            ax = axes[j, s]
            ax.plot(gains, simulated[:, s, j], color="#222222", lw=2.4, label="Simulator")
            for kind, model in models.items():
                label, color, ls = STYLE[kind]
                ax.plot(gains, model.predict(inputs)[:, j], color=color, ls=ls, lw=2, label=label)
            ax.set_ylabel(feature)
            ax.grid(alpha=.2)
        v0, _, half_width, height = scenario
        axes[0, s].set_title(f"Scenario {s + 1}: v0={v0 * 3.6:.0f} km/h, L={half_width:.2f} m, H={height:.3f} m", fontsize=10)
        axes[1, s].set_xlabel("P gain kp")
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("Gain response in held-out scenarios")
    fig.savefig(output / "surrogate_gain_response.png", dpi=200)
    plt.close(fig)
    print(f"Saved figures to {output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="outputs/offline_control/20260916_quadratic_ols")
    parser.add_argument("--output", default="outputs/offline_control/20260917_meeting_figures")
    parser.add_argument("--gain-points", type=int, default=271)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--parity-only", action="store_true", help="held-out parity 그림만 그리고 시뮬레이션은 생략")
    main(parser.parse_args())
