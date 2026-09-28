"""Reproduce the GP / COMs / continuous-action audit using saved posteriors.

Run from repository root: python -m lab.offline_control.run_benchmark
The existing online/offline entry point and presentation files are untouched.
"""
from __future__ import annotations

import os

# Set before importing numerical libraries, including in Windows workers.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from preference_loop.offline_models import (
    ConservativeRewardModels, FeatureRegressor, bounded_kernel_weights,
    continuous_values,
)
from preference_loop.optimization import SCENARIO_COVARIATES


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def simulate_gain(gain, seeds, bounds):
    from envs.vmc.preference import make_env
    env = make_env(SimpleNamespace(controller_bounds=[bounds], preferred_controller_bounds=[bounds]))
    return np.stack([env.design_row(env.rollout([gain], int(seed)))[1:] for seed in seeds])


def source_fingerprint(source):
    return {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in (
        "cfg.json", "preference_data.npz", "personalized_posteriors.npz",
    )}


def run(args):
    started = time.perf_counter()
    source, output = Path(args.source), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((source / "cfg.json").read_text(encoding="utf-8"))
    with np.load(source / "preference_data.npz") as log:
        names, roles = log["user_names"], log["user_roles"]
        train_users = np.flatnonzero(roles == "train")
        test_users = np.flatnonzero(roles == "test")[:args.users]
        gains_logged = log["feedback_controller"][train_users, :, 0]
        outcomes = log["feedback_Z"][train_users, :, 1:]
        contexts = np.stack([log[f"feedback_metadata_{key}"][train_users] for key in SCENARIO_COVARIATES], axis=-1)
        true_weights = log["theta_true"][test_users, 1:]
        test_names = names[test_users]
        # A few exact replay checks ensure the saved log matches today's simulator.
        replay = [(float(log["feedback_controller"][i, 0, 0]), int(log["feedback_metadata_env_seed"][i, 0]), log["feedback_Z"][i, 0, 1:]) for i in train_users[:3]]
    with np.load(source / "personalized_posteriors.npz") as posterior:
        if not np.array_equal(posterior["user_names"], names):
            raise ValueError("Posterior and episode user names differ")
        weights = posterior["theta_samples"][test_users].mean(axis=1)[:, 1:]
    bounds = cfg["controller_bounds"][0]
    for gain, seed, expected in replay:
        np.testing.assert_allclose(simulate_gain(gain, [seed], bounds)[0], expected, rtol=1e-8, atol=1e-10)
    print("Saved-log simulator replay: 3/3 passed", flush=True)
    gains = np.linspace(*bounds, args.gain_points)
    eval_seeds = np.random.SeedSequence(2026091501).generate_state(args.eval_scenarios)
    manifest = {
        "source": str(source.resolve()), "source_sha256": source_fingerprint(source),
        "arguments": vars(args), "test_users": test_names.tolist(),
        "logging_policy": "kp uniform on [30,300], independent of scenario",
        "preference_weights": "saved Bayesian posterior means; bias excluded",
        "data_origin": "VMC simulation; synthetic preference feedback",
        "split": "80% train-user groups for model fitting, 20% for independent OPE; model validation inside fitting split",
        "coms": "Eq.3 fixed-alpha contextual scalar-reward adaptation; alpha=0.1 after per-user reward standardization; matched MLP alpha=0",
        "optimizer": "identical bounded gain grid for all models; one fixed gain per user",
        "simulation": "fresh common scenarios, reserved for final diagnostics; no tuning or selection on these outcomes",
        "dr": "GP and polynomial DM + boundary-corrected Gaussian-kernel IPW residual, independent audit log; finite-bandwidth smoothing bias remains",
        "audit_scope": "audit episodes excluded from surrogate training and gain selection, but the saved hierarchical population posterior was originally fit using all 100 train users; weights are held fixed for this optimizer comparison, with no claim of unconditional OPE inference",
        "code_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__), Path("preference_loop/offline_models.py"))},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    cache = Path(args.simulator_bank) if args.simulator_bank else output / "simulator_bank.npz"
    if cache.exists() or args.simulator_bank:
        # Defer loading outcomes until all offline policies have been selected.
        executor, futures = None, None
    else:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        futures = [executor.submit(simulate_gain, gain, eval_seeds, bounds) for gain in gains]
        print(f"Final simulator bank queued: {len(gains)} gains x {len(eval_seeds)} scenarios", flush=True)

    repeats, model_info, curves = [], [], {}
    for repeat in range(args.repeats):
        rng = np.random.default_rng(510 + repeat)
        order = rng.permutation(len(train_users))
        fit_groups, audit_groups = order[:int(.8 * len(order))], order[int(.8 * len(order)):]
        valid_count = max(1, int(.2 * len(fit_groups)))
        valid_groups, tuning_groups = fit_groups[:valid_count], fit_groups[valid_count:]

        def arrays(groups):
            x = np.column_stack([gains_logged[groups].reshape(-1), contexts[groups].reshape(-1, contexts.shape[-1])])
            z = outcomes[groups].reshape(-1, outcomes.shape[-1])
            return x, z

        x, z = arrays(fit_groups)
        ax, az = arrays(audit_groups)
        tx, tz = arrays(tuning_groups)
        vx, vz = arrays(valid_groups)
        assert not set(fit_groups) & set(audit_groups)
        split = {"repeat": repeat, "fit_users": names[train_users[fit_groups]].tolist(), "audit_users": names[train_users[audit_groups]].tolist()}
        (output / f"split_{repeat}.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
        result = {"repeat": repeat, "models": {}, "audits": {}}
        for kind in ("poly", "gp"):
            tick = time.perf_counter()
            model = FeatureRegressor(kind, seed=510 + repeat).fit(x, z)
            fit_seconds = time.perf_counter() - tick
            tick = time.perf_counter()
            mean = model.mean_features(gains, x[:, 1:])
            values = mean @ weights.T
            select_seconds = time.perf_counter() - tick
            audit_mean = model.mean_features(gains, ax[:, 1:])
            predicted_audit = model.predict(ax)
            result["models"][kind] = {"values": values, "fit_seconds": fit_seconds, "select_seconds": select_seconds}
            info = {"repeat": repeat, "method": kind, "fit_seconds": fit_seconds, "selection_seconds": select_seconds, "best_step": None,
                    "audit_reward_rmse": float(np.sqrt(np.mean(((predicted_audit - az) @ weights.T) ** 2))),
                    "audit_pitch_rmse": float(np.sqrt(np.mean((predicted_audit[:, 0] - az[:, 0]) ** 2))),
                    "audit_long_rmse": float(np.sqrt(np.mean((predicted_audit[:, 1] - az[:, 1]) ** 2)))}
            model_info.append(info)
            joblib.dump(model, output / f"{kind}_{repeat}.joblib")
            for bandwidth in (10., 20., 40.):
                kw = bounded_kernel_weights(gains, ax[:, 0], bounds, bandwidth, 1 / (bounds[1] - bounds[0]))
                estimates = continuous_values(audit_mean, az, predicted_audit, kw)
                result["audits"][(kind, bandwidth)] = {key: value if key == "ess" else value @ weights.T for key, value in estimates.items()}
            if kind == "gp":
                (output / f"gp_kernel_{repeat}.txt").write_text(str(model.model.kernel_), encoding="utf-8")
            print(f"Repeat {repeat + 1}: {kind} fitted in {fit_seconds:.1f}s, audit feature RMSE {info['audit_pitch_rmse']:.4f}/{info['audit_long_rmse']:.4f}", flush=True)
        for kind, alpha in (("mlp", 0.), ("coms", .1)):
            tick = time.perf_counter()
            tuning = ConservativeRewardModels(alpha=alpha, seed=610 + repeat, steps=args.steps).fit(
                tx, tz @ weights.T, bounds, validation=(vx, vz @ weights.T))
            best_step = tuning.best_step
            history = tuning.history
            del tuning
            model = ConservativeRewardModels(alpha=alpha, seed=610 + repeat, steps=best_step).fit(x, z @ weights.T, bounds)
            fit_seconds = time.perf_counter() - tick
            tick = time.perf_counter()
            values = model.mean_rewards(gains, x[:, 1:])
            select_seconds = time.perf_counter() - tick
            predicted_audit = model.predict(ax)
            result["models"][kind] = {"values": values, "fit_seconds": fit_seconds, "select_seconds": select_seconds}
            model_info.append({"repeat": repeat, "method": kind, "fit_seconds": fit_seconds, "selection_seconds": select_seconds,
                               "best_step": best_step, "audit_reward_rmse": float(np.sqrt(np.mean((predicted_audit - az @ weights.T) ** 2))),
                               "audit_pitch_rmse": None, "audit_long_rmse": None})
            import torch
            torch.save({"state_dict": model.net.state_dict(), "lower": model.lower, "span": model.span,
                        "reward_mean": model.ymean, "reward_scale": model.yscale, "weights": weights,
                        "alpha": alpha, "seed": 610 + repeat, "steps": best_step, "hidden": model.hidden,
                        "validation_history": history}, output / f"{kind}_{repeat}.pt")
            print(f"Repeat {repeat + 1}: {kind} fitted in {fit_seconds:.1f}s, selected step {best_step}", flush=True)
            del model
        repeats.append(result)
        joblib.dump(result, output / f"offline_result_{repeat}.joblib")
        write_csv(output / "model_fit.csv", model_info)

    if futures is not None:
        sim_features = np.stack([future.result() for future in futures])
        executor.shutdown()
        np.savez_compressed(cache, gains=gains, seeds=eval_seeds, features=sim_features)
    else:
        with np.load(cache) as bank:
            np.testing.assert_array_equal(bank["gains"], gains)
            np.testing.assert_array_equal(bank["seeds"], eval_seeds)
            sim_features = bank["features"]
    mean_sim = sim_features.mean(axis=1)
    physical_true = mean_sim @ true_weights.T
    physical_hat = mean_sim @ weights.T
    true_best = np.argmax(physical_true, axis=0)
    posterior_best = np.argmax(physical_hat, axis=0)
    users, audit_rows, curve_rows = [], [], []
    for result in repeats:
        repeat = result["repeat"]
        for method, model in result["models"].items():
            values = model["values"]
            curves[f"{method}_{repeat}"] = values
            selected = np.argmax(values, axis=0)
            for u, name in enumerate(test_names):
                k = selected[u]
                users.append({"repeat": repeat, "user": name, "method": method, "gain": gains[k],
                              "oracle_gain": gains[true_best[u]], "posterior_sim_gain": gains[posterior_best[u]],
                              "true_reward": physical_true[k, u],
                              "true_regret": physical_true[true_best[u], u] - physical_true[k, u],
                              "posterior_objective_regret": physical_hat[posterior_best[u], u] - physical_hat[k, u],
                              "gain_error": abs(gains[k] - gains[true_best[u]]),
                              "curve_rmse": np.sqrt(np.mean((values[:, u] - physical_hat[:, u]) ** 2)),
                              "selection_prediction_bias": values[k, u] - physical_hat[k, u]})
                for (nuisance, bandwidth), audit in result["audits"].items():
                    for estimator in ("dm", "dr", "ipw"):
                        predicted = audit[estimator][k, u]
                        audit_rows.append({"repeat": repeat, "user": name, "policy_model": method,
                                           "nuisance": nuisance, "bandwidth": bandwidth, "estimator": estimator,
                                           "estimate": predicted, "simulator_value": physical_hat[k, u],
                                           "error": predicted - physical_hat[k, u], "ess": audit["ess"][k]})
            curve_rows.append({"repeat": repeat, "method": method,
                               "rmse": float(np.sqrt(np.mean((values - physical_hat) ** 2)))})
        for u, name in enumerate(test_names):
            k = posterior_best[u]
            users.append({"repeat": repeat, "user": name, "method": "sim_posterior_reference", "gain": gains[k],
                          "oracle_gain": gains[true_best[u]], "posterior_sim_gain": gains[k],
                          "true_reward": physical_true[k, u],
                          "true_regret": physical_true[true_best[u], u] - physical_true[k, u],
                          "posterior_objective_regret": 0., "gain_error": abs(gains[k] - gains[true_best[u]]),
                          "curve_rmse": 0., "selection_prediction_bias": 0.})
    write_csv(output / "users.csv", users)
    write_csv(output / "continuous_evaluation.csv", audit_rows)
    write_csv(output / "curve_errors.csv", curve_rows)
    np.savez_compressed(output / "gain_curves.npz", gains=gains, test_users=test_names,
                        posterior_weights=weights, true_weights=true_weights,
                        simulator_true=physical_true, simulator_posterior=physical_hat, **curves)
    summarize(output, users, audit_rows, model_info, args.repeats)
    manifest["elapsed_seconds"] = time.perf_counter() - started
    manifest["simulator_replay_passed"] = len(replay)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Complete in {manifest['elapsed_seconds']:.1f}s: {output.resolve()}", flush=True)


def summarize(output, users, audit_rows, model_info, repeats):
    import pandas as pd
    frame, audit, timing = pd.DataFrame(users), pd.DataFrame(audit_rows), pd.DataFrame(model_info)
    columns = ["true_reward", "true_regret", "posterior_objective_regret", "gain_error", "curve_rmse", "selection_prediction_bias"]
    per_repeat = frame.groupby(["method", "repeat"])[columns].mean().reset_index()
    per_repeat.to_csv(output / "repeat_summary.csv", index=False, encoding="utf-8-sig")
    summary = []
    for method, group in per_repeat.groupby("method", sort=False):
        row = {"method": method}
        for col in columns:
            row[col] = float(group[col].mean())
            row[col + "_split_sd"] = float(group[col].std(ddof=1)) if repeats > 1 else 0.
        for col in ("fit_seconds", "selection_seconds", "audit_reward_rmse", "audit_pitch_rmse", "audit_long_rmse"):
            row[col] = float(timing[timing.method == method][col].mean()) if method != "sim_posterior_reference" else None
        summary.append(row)
    write_csv(output / "summary.csv", summary)
    audit["absolute_error"] = abs(audit.error)
    audit["squared_error"] = audit.error ** 2
    audit_summary = audit.groupby(["nuisance", "bandwidth", "estimator"], sort=True).agg(
        mae=("absolute_error", "mean"), mse=("squared_error", "mean"), bias=("error", "mean"),
        ess_mean=("ess", "mean"), ess_min=("ess", "min")).reset_index()
    audit_summary["rmse"] = np.sqrt(audit_summary.pop("mse"))
    audit_summary.to_csv(output / "continuous_summary.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(summary)[["method", "true_reward", "true_regret", "posterior_objective_regret", "curve_rmse"]].to_string(index=False), flush=True)
    print(audit_summary[(audit_summary.nuisance == "gp") & (audit_summary.bandwidth == 20)].to_string(index=False), flush=True)
    plot(output, frame, summary)


def plot(output, frame, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams["axes.unicode_minus"] = False
    order = ("poly", "gp", "mlp", "coms")
    palette = {"poly": "#DC7E35", "gp": "#3178B3", "mlp": "#9E77B4", "coms": "#329A72"}
    selected = [next(row for row in summary if row["method"] == method) for method in order]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    axes[0].bar([row["method"] for row in selected], [row["true_regret"] for row in selected],
                yerr=[row["true_regret_split_sd"] for row in selected], capsize=4,
                color=[palette[row["method"]] for row in selected])
    axes[0].set(ylabel="True-reward regret (lower is better)", title="50 users; mean +/- SD over log splits")
    with np.load(output / "gain_curves.npz") as data:
        # First user is fixed in advance; no selection of a favorable example.
        axes[1].plot(data["gains"], data["simulator_posterior"][:, 0], "k", lw=2, label="simulator")
        for method in order:
            axes[1].plot(data["gains"], data[f"{method}_0"][:, 0], label=method, color=palette[method])
    axes[1].set(xlabel="Fixed gain kp", ylabel="Posterior-mean reward", title="First held-out user, first split")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
        ax.yaxis.set_major_locator(MaxNLocator(6))
    fig.savefig(output / "comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="outputs/preference_loop/20260814_223618")
    parser.add_argument("--output", default="outputs/offline_control/20260915_comparison")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--users", type=int, default=50)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--simulator-bank", default=None, help="Reuse an existing final-evaluation bank with identical gains and scenario seeds")
    parser.add_argument("--gain-points", type=int, default=271)
    parser.add_argument("--eval-scenarios", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parsed = parser.parse_args()
    with threadpool_limits(limits=1):
        run(parsed)
