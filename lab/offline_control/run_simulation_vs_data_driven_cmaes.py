"""Simulation-based vs data-driven CMA-ES for personalized fixed gains.

Simulation-based: the saved online run, where CMA-ES scores each candidate by
rolling it out on 40 fixed simulator scenarios. Data-driven: the same CMA-ES
settings and seeds, scoring each candidate by a GP fitted to the 1,000 logged
train-user episodes, averaged over those logged scenarios. Both use the saved
posterior mean of each test user. Selected gains are evaluated on the fresh
128-scenario simulator bank with true preferences only after selection.
Run from repository root: python -m lab.offline_control.run_simulation_vs_data_driven_cmaes
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
import time
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from envs.vmc.env import ErideEnv
from envs.vmc.preference import make_env
from envs.vmc.rollout import make_controller
from lab.offline_control.run_benchmark import simulate_gain
from preference_loop.data import derive_seed
from preference_loop.offline_models import FeatureRegressor
from preference_loop.optimization import SCENARIO_COVARIATES, optimize_cmaes

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "font.size": 12, "axes.labelsize": 12, "axes.titlesize": 13, "xtick.labelsize": 11,
                     "ytick.labelsize": 11, "legend.fontsize": 11, "axes.unicode_minus": False})

STYLE = {"simulation": ("Simulation-based CMA-ES", "#3178B3", "--"), "data": ("Data-driven CMA-ES (GP)", "#D62728", "-.")}


def main(args):
    source, output = Path(args.source), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = SimpleNamespace(**json.loads((source / "cfg.json").read_text(encoding="utf-8")))
    bounds = cfg.controller_bounds[0]
    if not args.replot:  # --replot 이면 CMA-ES·시뮬레이션 없이 저장된 cmaes_users.csv 로 아래 그림만 다시 그림
        env = make_env(cfg)
        with np.load(source / "preference_data.npz") as log:
            names, roles = log["user_names"], log["user_roles"]
            train, test = np.flatnonzero(roles == "train"), np.flatnonzero(roles == "test")
            x = np.column_stack([log["feedback_controller"][train, :, 0].reshape(-1),
                                 *[log[f"feedback_metadata_{key}"][train].reshape(-1) for key in SCENARIO_COVARIATES]])
            z = log["feedback_Z"][train, :, 1:].reshape(-1, 2)
            theta_true = log["theta_true"][test]
        with np.load(source / "personalized_posteriors.npz") as posterior:
            samples = posterior["theta_samples"][test]
        with np.load(source / "cmaes_optimization.npz") as saved:
            simulation_gain, saved_history = saved["final_controller"][test, 0], saved["controller"][test, :, 0]
        theta_hat = samples.mean(axis=1)

        tick = time.perf_counter()
        gp = FeatureRegressor("gp", seed=cfg.seed).fit(x, z)
        fit_seconds = time.perf_counter() - tick
        features = ("offline", (gp.model, x[:, 1:], gp.lower, gp.span))
        data_gain, data_history = [], []
        tick = time.perf_counter()
        for u, index in enumerate(test):
            result = optimize_cmaes(cfg, env, theta_hat[u, 1:], derive_seed(cfg.cma_seed, int(index), 0), features, None)
            data_gain.append(result["parameters"][0])
            data_history.append(result["controller"][:, 0])
        optimize_seconds = time.perf_counter() - tick
        data_gain, data_history = np.asarray(data_gain), np.stack(data_history)
        # CMA-ES samples its first generation before seeing any score, so matching it
        # confirms the saved simulation run used the same settings and seeds.
        np.testing.assert_allclose(data_history[:, :cfg.cma_population_size], saved_history[:, :cfg.cma_population_size])
        print(f"GP fitted in {fit_seconds:.1f}s, data-driven CMA-ES for {len(test)} users in {optimize_seconds:.1f}s; first generation matches saved run", flush=True)

        with np.load(args.simulator_bank) as bank:
            gains, seeds, bank_features = bank["gains"], bank["seeds"], bank["features"]
        selected = {"simulation": simulation_gain, "data": data_gain}
        with ProcessPoolExecutor(args.workers) as executor:
            outcomes = {key: np.stack(list(executor.map(simulate_gain, gain, [seeds] * len(gain), [bounds] * len(gain))))
                        for key, gain in selected.items()}
        true_curve = bank_features.mean(axis=1) @ theta_true[:, 1:].T
        hat_curve = bank_features.mean(axis=1) @ theta_hat[:, 1:].T
        best = np.argmax(true_curve, axis=0)
        rows = []
        for u, index in enumerate(test):
            row = {"user": names[index], "true_best_gain": gains[best[u]]}
            for key, gain in selected.items():
                mean = outcomes[key][u].mean(axis=0)
                row[f"{key}_gain"] = gain[u]
                row[f"{key}_gain_error"] = abs(gain[u] - gains[best[u]])
                row[f"{key}_true_regret"] = true_curve[best[u], u] - mean @ theta_true[u, 1:]
                row[f"{key}_posterior_regret"] = hat_curve[:, u].max() - mean @ theta_hat[u, 1:]
            rows.append(row)
        with (output / "cmaes_users.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        rollouts = {"simulation": cfg.cma_population_size * cfg.cma_generations * cfg.n_optimization_scenarios, "data": 0}
        summary = [{"method": STYLE[key][0], "simulator_rollouts_per_user": rollouts[key],
                    **{metric: float(np.mean([row[f"{key}_{metric}"] for row in rows])) for metric in ("gain_error", "true_regret", "posterior_regret")}}
                   for key in STYLE]
        summary[1]["gp_fit_seconds"], summary[1]["optimization_seconds_all_users"] = fit_seconds, optimize_seconds
        (output / "cmaes_summary.json").write_text(json.dumps({"source": str(source.resolve()), "gp_kernel": str(gp.model.kernel_), "summary": summary}, indent=2), encoding="utf-8")
        for row in summary:
            print(row, flush=True)

        # Both curves are drawn on the 128 evaluation scenarios, so their gap is surrogate
        # error only. Scenario covariates follow from the seed at reset, without a rollout.
        scenario_env = ErideEnv(make_controller({"controller": "p", "kp": bounds[0]}), mode="pure", record_inner=True)
        bank_scenarios = []
        for seed in seeds:
            scenario_env.reset(seed=int(seed))
            bank_scenarios.append([scenario_env.initial_state[4], *scenario_env.bump.bump_specs[0]])
        curves = {"simulation": bank_features.mean(axis=1), "data": gp.mean_features(gains, np.asarray(bank_scenarios))}
        print(f"Max |GP - simulator| mean feature on evaluation scenarios: {np.abs(curves['data'] - curves['simulation']).max(axis=0)}", flush=True)
        fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
        # First five test users are fixed in advance; no favorable examples are selected.
        for u, ax in enumerate(axes.ravel()[:5]):
            logit = {key: samples[u, :, :1].T + feature @ samples[u, :, 1:].T for key, feature in curves.items()}
            ax.fill_between(gains, *np.quantile(logit["simulation"], [.05, .95], axis=1), color=STYLE["simulation"][1], alpha=.15, lw=0)
            ax.plot(gains, logit["simulation"].mean(axis=1), color=STYLE["simulation"][1], lw=2.4)
            ax.plot(gains, logit["data"].mean(axis=1), color=STYLE["data"][1], ls="--", lw=2)
            ax.axvline(gains[best[u]], color="#2CA02C", ls=":", lw=2)
            for key, gain in selected.items():
                ax.axvline(gain[u], color=STYLE[key][1], ls=STYLE[key][2], lw=1.8)
            ax.set(title=f"Test user {u + 1}", xlabel="P gain kp", ylabel="Feedback logit", xlim=bounds)
            ax.grid(alpha=.2)
        legend = axes.ravel()[5]
        legend.axis("off")
        handles = [plt.Rectangle((0, 0), 1, 1, color=STYLE["simulation"][1], alpha=.3), plt.Line2D([], [], color=STYLE["simulation"][1], lw=2.4),
                   plt.Line2D([], [], color=STYLE["data"][1], ls="--", lw=2), plt.Line2D([], [], color="#2CA02C", ls=":", lw=2),
                   *[plt.Line2D([], [], color=color, ls=ls, lw=1.8) for _, color, ls in STYLE.values()]]
        labels = ["Simulator posterior 90% CI", "Simulator posterior mean", "GP posterior mean",
                  "True best", *[label for label, _, _ in STYLE.values()]]
        legend.legend(handles, labels, loc="center", fontsize=11, frameon=False)
        fig.savefig(output / "cmaes_users.png", dpi=200)
        plt.close(fig)

    with (output / "cmaes_users.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    column = lambda name: np.array([float(row[name]) for row in rows])
    true_best, picked = column("true_best_gain"), {key: column(f"{key}_gain") for key in STYLE}
    fig, ax = plt.subplots(figsize=(5.4, 5.2), constrained_layout=True)
    ax.plot(bounds, bounds, color="0.55", lw=1)
    # 두 방법의 선택 gain 이 거의 같아 겹치므로 simulation-based 는 큰 빈 원을 위에, data-driven 은 작은 채운 사각형을 아래에
    for key, marker, size, order in (("simulation", "o", 130, 3), ("data", "s", 26, 2)):
        label, color, _ = STYLE[key]
        ax.scatter(true_best, picked[key], s=size, marker=marker, facecolors="none" if key == "simulation" else color,
                   edgecolors=color, linewidths=1.4, zorder=order, label=f"{label}: |error| {column(f'{key}_gain_error').mean():.1f}")
    ax.set(xlim=bounds, ylim=bounds, xlabel="True best kp", ylabel="Selected kp",
           title=f"{len(rows)} test users")
    ax.legend(loc="upper right")
    ax.grid(alpha=.2)
    fig.savefig(output / "cmaes_selected_vs_true_best.png", dpi=200)
    plt.close(fig)
    print(f"Saved to {output.resolve()}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="outputs/preference_loop/20260814_223618")
    parser.add_argument("--simulator-bank", default="outputs/offline_control/20260915_comparison/simulator_bank.npz")
    parser.add_argument("--output", default="outputs/offline_control/20260917_meeting_figures")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--replot", action="store_true", help="저장된 cmaes_users.csv 로 selected vs true best 그림만 다시 그림")
    main(parser.parse_args())
