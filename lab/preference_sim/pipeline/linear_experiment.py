"""Competence-constrained policy bank with a strictly linear preference model."""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from .common import ROOT, write_json
from .linear_env import FEATURE_NAMES, SIGNAL_NAMES, make_env, rollout, transform_policy, weight_vector


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def evaluate(policy, config, weights, seeds, horizon=None):
    horizon = int(horizon or config["environment"]["horizon"])
    env = make_env(config, weights, horizon=horizon)
    try:
        episodes = [rollout(policy, env, seed, horizon) for seed in seeds]
    finally:
        env.close()
    means = np.asarray([e["signals"].mean(axis=0) for e in episodes])
    phi = np.asarray([e["phi"] for e in episodes]).mean(axis=0)
    return {
        "n_episodes": len(episodes), "horizon": horizon,
        "completion_rate": float(np.mean([e["length"] == horizon and not e["terminated"] for e in episodes])),
        "fall_rate": float(np.mean([e["terminated"] for e in episodes])),
        "mean_length_fraction": float(np.mean([e["length"] / horizon for e in episodes])),
        "phi": dict(zip(FEATURE_NAMES, map(float, phi))),
        **{f"phi_{name}": float(value) for name, value in zip(FEATURE_NAMES, phi)},
        **dict(zip(SIGNAL_NAMES, map(float, means.mean(axis=0)))),
    }


def competent(metrics, config, profile_name=None):
    gate = config["competence"]
    passed = bool(
        metrics["completion_rate"] >= gate["min_completion_rate"]
        and metrics["mean_length_fraction"] >= gate["min_mean_length_fraction"]
        and metrics["speed"] >= gate["min_speed"]
    )
    profile = config.get("profiles", {}).get(profile_name, {})
    if profile.get("bilateral", False):
        passed = passed and all(
            metrics[f"phi_{feature}"] >= gate[f"min_{feature}"]
            for feature in ("alternation", "role_exchange", "stance_balance", "push_balance")
        )
    for metric, bounds in profile.get("gates", {}).items():
        value = metrics[metric]
        passed = passed and value >= bounds.get("min", -np.inf) and value <= bounds.get("max", np.inf)
    return bool(passed)


def model_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_policy(config, path, profile_name):
    policy = PPO.load(path, device="cpu")
    transform = config["profiles"][profile_name].get("policy_transform")
    return transform_policy(policy, transform)


def train_profile(config, run_dir, name, index):
    torch.set_num_threads(1)
    run_dir = Path(run_dir)
    training = config["training"]
    weights = config["profiles"][name]["weights"]
    directory = run_dir / "policies" / name
    directory.mkdir(parents=True, exist_ok=True)
    n_envs = int(training.get("n_envs", 8))
    vec = DummyVecEnv([lambda: make_env(config, weights) for _ in range(n_envs)])
    seed = int(config["seed"]) + index * 1000
    policy = PPO.load(
        model_path(config["profiles"][name].get("pretrained_model", training["pretrained_model"])),
        env=vec, device="cpu", learning_rate=float(training["learning_rate"]),
        n_steps=int(training["n_steps"]), batch_size=int(training["batch_size"]),
        n_epochs=int(training["n_epochs"]), ent_coef=float(training.get("ent_coef", 0.0)),
        target_kl=float(training.get("target_kl", 0.025)),
        gamma=float(training.get("gamma", 0.99)),
    )
    interpolation_source = config["profiles"][name].get("interpolate_with")
    if interpolation_source:
        alpha = float(config["profiles"][name].get("interpolation_alpha", 0.5))
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"interpolation_alpha for {name} must be in [0, 1]")
        other = PPO.load(model_path(interpolation_source), device="cpu")
        base_state = policy.policy.state_dict()
        other_state = other.policy.state_dict()
        if base_state.keys() != other_state.keys():
            raise ValueError(f"Cannot interpolate incompatible policies for {name}")
        policy.policy.load_state_dict({
            key: torch.lerp(value, other_state[key], alpha) for key, value in base_state.items()
        })
    if training.get("initial_action_std") is not None:
        with torch.no_grad():
            policy.policy.log_std.fill_(float(np.log(training["initial_action_std"])))
    policy.set_random_seed(seed)
    seeds = range(int(training["evaluation_seed"]), int(training["evaluation_seed"]) + int(training["evaluation_episodes"]))
    rows, steps = [], 0
    start = time.time()
    try:
        for target in config["profiles"][name].get("checkpoint_steps", training["checkpoint_steps"]):
            remaining = int(target) - steps
            if remaining <= 0 and not (int(target) == 0 and not rows):
                continue
            if remaining > 0:
                policy.learn(total_timesteps=remaining, reset_num_timesteps=(steps == 0))
                steps = int(policy.num_timesteps)
            path = directory / f"step_{steps:09d}.zip"
            policy.save(path)
            evaluated_policy = transform_policy(policy, config["profiles"][name].get("policy_transform"))
            metrics = evaluate(evaluated_policy, config, weights, seeds)
            row = {
                "profile": name, "step": steps, "seed": seed, "weights": weights,
                "feature_version": config["environment"].get("feature_version", 1),
                "model_path": str(path.relative_to(run_dir)),
                "screen": metrics, "screen_pass": competent(metrics, config, name),
            }
            rows.append(row)
            write_json(directory / "index.json", rows)
            print(f"[{name}] step={steps} complete={metrics['completion_rate']:.2f} "
                  f"speed={metrics['speed']:.2f} height={metrics['height']:.2f} "
                  f"angle={metrics['torso_abs_angle']:.3f} knee={metrics['knee_abs_angle']:.3f} "
                  f"elapsed={time.time()-start:.0f}s", flush=True)
    finally:
        vec.close()
    return rows


def train(config, run_dir):
    with ProcessPoolExecutor(max_workers=int(config["training"].get("workers", 4))) as pool:
        futures = [pool.submit(train_profile, config, str(run_dir), name, i)
                   for i, name in enumerate(config["profiles"])]
        # Workers checkpoint independently, so completed work survives a failure.
        rows = [row for future in futures for row in future.result()]
    write_json(run_dir / "policies" / "candidates.json", rows)


def select(config, run_dir):
    rows = [row for name in config["profiles"]
            for row in read_json(run_dir / "policies" / name / "index.json")]
    selection = config["selection"]
    seeds = range(selection["seed"], selection["seed"] + selection["episodes"])
    selected, failures = [], []
    for name in config["profiles"]:
        candidates = [r for r in rows if r["profile"] == name and r["screen_pass"]]
        for row in candidates:
            policy = load_policy(config, run_dir / row["model_path"], name)
            row["validation"] = evaluate(policy, config, row["weights"], seeds)
            row["validation_pass"] = competent(row["validation"], config, name)
            print(f"[select] {name} step={row['step']} complete={row['validation']['completion_rate']:.2f}", flush=True)
        passing = [r for r in candidates if r["validation_pass"]]
        if not passing:
            failures.append(name)
            continue
        feature = config["profiles"][name]["target_feature"]
        mode = config["profiles"][name].get("target_mode", "max")
        direction = -1.0 if mode == "min" else 1.0
        best = max(passing, key=lambda r: direction * r["validation"]["phi"][feature])
        selected.append(best)
    write_json(run_dir / "policies" / "selected.json", selected)
    write_json(run_dir / "reports" / "selection.json", {"candidates": rows, "missing_profiles": failures})
    if failures:
        print(f"[select] NO PASSING POLICY: {', '.join(failures)}", flush=True)
    return selected


def _episode_row(episode, profile, seed, split, noise, eligible):
    return {
        "profile": profile, "seed": seed, "split": split, "noise": noise,
        "candidate": eligible, "length": episode["length"], "terminated": episode["terminated"],
        **dict(zip(SIGNAL_NAMES, map(float, episode["signals"].mean(axis=0)))),
        **{f"phi_{n}": float(v) for n, v in zip(FEATURE_NAMES, episode["phi"])},
    }


def collect(config, run_dir):
    selected = read_json(run_dir / "policies" / "selected.json")
    if not selected:
        raise RuntimeError("No competent policies selected")
    cfg = config["collection"]
    horizon = int(config["environment"]["horizon"])
    rows, phis, signals, masks, step_phis = [], [], [], [], []
    # Calibration fits the coordinate transform and user thresholds; context and
    # test trajectories use entirely different reset seeds. No test fitting.
    for checkpoint in selected:
        policy = load_policy(config, run_dir / checkpoint["model_path"], checkpoint["profile"])
        env = make_env(config, checkpoint["weights"])
        try:
            for split in ("calibration", "context", "test"):
                for i in range(cfg[f"{split}_episodes"]):
                    seed = cfg[f"{split}_seed"] + i
                    e = rollout(policy, env, seed, horizon)
                    rows.append(_episode_row(e, checkpoint["profile"], seed, split, 0, True))
                    phis.append(e["phi"])
                    padded = np.zeros((horizon, len(SIGNAL_NAMES)), dtype=np.float32)
                    padded[:e["length"]] = e["signals"]
                    signals.append(padded)
                    padded_phi = np.zeros((horizon, len(FEATURE_NAMES)), dtype=np.float32)
                    padded_phi[:e["length"]] = e["step_phi"]
                    step_phis.append(padded_phi)
                    masks.append(np.arange(horizon) < e["length"])
            print(f"[collect] {checkpoint['profile']} complete", flush=True)
        finally:
            env.close()
    # Failure examples identify survival/progress weights, but cannot be selected
    # as personalized controllers. The same diagnostic recipe is split by seed.
    policy = load_policy(config, run_dir / selected[0]["model_path"], selected[0]["profile"])
    env = make_env(config, {})
    try:
        for split in ("calibration", "context", "test"):
            for i in range(cfg["diagnostic_episodes"]):
                seed = cfg[f"{split}_seed"] + 10000 + i
                noise = 0.0 if i % 3 == 0 else (0.35 if i % 3 == 1 else 0.7)
                e = rollout(None if i % 3 == 0 else policy, env, seed, horizon, noise=noise)
                rows.append(_episode_row(e, "diagnostic", seed, split, noise, False))
                phis.append(e["phi"])
                padded = np.zeros((horizon, len(SIGNAL_NAMES)), dtype=np.float32)
                padded[:e["length"]] = e["signals"]
                signals.append(padded)
                padded_phi = np.zeros((horizon, len(FEATURE_NAMES)), dtype=np.float32)
                padded_phi[:e["length"]] = e["step_phi"]
                step_phis.append(padded_phi)
                masks.append(np.arange(horizon) < e["length"])
    finally:
        env.close()
    table = pd.DataFrame(rows)
    table.to_csv(run_dir / "tables" / "episodes.csv", index=False)
    np.savez_compressed(run_dir / "rollouts" / "bank.npz", phi=np.asarray(phis),
                        X=np.asarray(signals), step_phi=np.asarray(step_phis), valid=np.asarray(masks),
                        feature_version=np.asarray(config["environment"].get("feature_version", 1)),
                        feature_names=np.asarray(FEATURE_NAMES), signal_names=np.asarray(SIGNAL_NAMES))
    return report_bank(config, run_dir, table)


def report_bank(config, run_dir, table):
    test = table[(table.split == "test") & table.candidate]
    summaries = {}
    for name, group in test.groupby("profile", sort=False):
        metrics = {
            "n_episodes": len(group),
            "completion_rate": float(((group.length == config["environment"]["horizon"]) & ~group.terminated).mean()),
            "fall_rate": float(group.terminated.mean()),
            "mean_length_fraction": float(group.length.mean() / config["environment"]["horizon"]),
            **{key: float(group[key].mean()) for key in SIGNAL_NAMES},
            **{f"phi_{key}": float(group[f"phi_{key}"].mean()) for key in FEATURE_NAMES},
        }
        metrics["competence_pass"] = competent(metrics, config, name)
        summaries[name] = metrics
    effects = []
    for name, profile in config["profiles"].items():
        if name in summaries:
            for expectation in profile.get("effects", []):
                reference = expectation.get("reference", profile.get("reference", "baseline"))
                if reference not in summaries:
                    continue
                delta = summaries[name][expectation["metric"]] - summaries[reference][expectation["metric"]]
                directed = delta if expectation["direction"] == "higher" else -delta
                detail = {key: value for key, value in expectation.items() if key != "reference"}
                effects.append({"profile": name, "reference": reference, **detail, "difference": delta,
                                "passed": bool(directed >= expectation["minimum"])})
    phi_cols = [f"phi_{n}" for n in FEATURE_NAMES]
    centered = table[phi_cols].to_numpy() - table[phi_cols].to_numpy().mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    missing = sorted(set(config["profiles"]) - set(summaries))
    report = {
        "profiles": summaries, "effects": effects, "missing_profiles": missing,
        "quality_pass": not missing and all(m["competence_pass"] for m in summaries.values()) and all(e["passed"] for e in effects),
        "feature_rank": int(np.linalg.matrix_rank(centered)),
        "feature_singular_values": singular.tolist(),
        "diagnostic_episodes": int((~table.candidate).sum()),
        "note": "All held-out episodes counted, including falls. Selection used validation seeds only.",
    }
    write_json(run_dir / "reports" / "gait_quality.json", report)
    pd.DataFrame(summaries).T.to_csv(run_dir / "reports" / "gait_metrics.csv")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for ax, metric in zip(axes.flat, ("speed", "height", "torso_abs_angle", "knee_abs_angle", "airborne", "action_delta")):
        for i, (name, group) in enumerate(test.groupby("profile", sort=False)):
            ax.errorbar(i, group[metric].mean(), yerr=group[metric].std(), fmt="o", capsize=4)
        ax.set_xticks(range(len(summaries)), summaries, rotation=25)
        ax.set_title(metric + " (mean +/- SD)")
        ax.grid(alpha=0.2)
    fig.savefig(run_dir / "reports" / "gait_comparison.png", dpi=160)
    plt.close(fig)
    return report


def users(config, run_dir):
    cfg = config["users"]
    episodes = pd.read_csv(run_dir / "tables" / "episodes.csv")
    with np.load(run_dir / "rollouts" / "bank.npz", allow_pickle=False) as bank:
        phi = bank["phi"]
    calibration = (episodes.split == "calibration").to_numpy()
    feasible_cal = calibration & episodes.candidate.to_numpy()
    center = phi[calibration].mean(axis=0)
    scale = phi[calibration].std(axis=0)
    scale[scale < 1e-6] = 1.0
    z = (phi - center) / scale
    Z = np.column_stack([np.ones(len(phi)), z])
    rng = np.random.default_rng(cfg["seed"])
    n_train, n_test = int(cfg["n_train"]), int(cfg["n_test"])
    mean, sd = weight_vector(cfg["mean"]), weight_vector(cfg["sd"])
    # Exactly one Gaussian population: no archetype mixture or norm projection.
    normal_weights = rng.normal(mean, sd, size=(n_train + n_test, len(FEATURE_NAMES)))
    probes = cfg.get("probe_users", {})
    probe_names = list(probes)
    probe_weights = np.asarray(
        [weight_vector(probes[name]["weights"]) for name in probe_names], dtype=float,
    ).reshape(len(probe_names), len(FEATURE_NAMES))
    weights = np.concatenate([normal_weights, probe_weights], axis=0)
    # A linear calibration reference + independent Gaussian threshold preserves
    # joint Gaussianity of the effective logistic parameters, including bias.
    reference = phi[feasible_cal].mean(axis=0)
    normal_thresholds = normal_weights @ reference + rng.normal(
        0, cfg["threshold_sd"], len(normal_weights),
    )
    probe_thresholds = np.asarray([
        probe_weights[i] @ reference + float(probes[name].get("threshold_offset", 0.0))
        for i, name in enumerate(probe_names)
    ])
    thresholds = np.concatenate([normal_thresholds, probe_thresholds])
    beta = float(cfg["beta"])
    theta = np.column_stack([beta * (weights @ center - thresholds), beta * weights * scale])
    logits = theta @ Z.T
    expected = beta * (weights @ phi.T - thresholds[:, None])
    np.testing.assert_allclose(logits, expected, atol=1e-10, rtol=1e-10)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
    labels = (rng.random(probabilities.shape) < probabilities).astype(np.int8)
    context_indices = np.flatnonzero((episodes.split == "context").to_numpy())
    limit = int(cfg["labels_per_user"])
    if limit > len(context_indices):
        raise ValueError(f"Requested {limit} labels from only {len(context_indices)} context trajectories")
    context = np.asarray([rng.choice(context_indices, size=limit, replace=False) for _ in weights])
    identifiers = np.asarray(
        [f"train_{i:05d}" for i in range(n_train)]
        + [f"test_{i:05d}" for i in range(n_train, n_train + n_test)]
        + [f"probe_{name}" for name in probe_names]
    )
    splits = np.asarray(["train"] * n_train + ["test"] * n_test + ["probe"] * len(probe_names))
    table = pd.DataFrame({"user_id": identifiers, "split": splits, "threshold": thresholds})
    for j, name in enumerate(FEATURE_NAMES):
        table[f"w_{name}"] = weights[:, j]
    table.to_csv(run_dir / "tables" / "users.csv", index=False)
    profile_names = list(dict.fromkeys(episodes.loc[episodes.candidate, "profile"]))
    profile_phi = np.asarray([phi[((episodes.split == "calibration") & (episodes.profile == name)).to_numpy()].mean(axis=0) for name in profile_names])
    winners = (weights @ profile_phi.T).argmax(axis=1)
    # Calibration labels are deliberately not exported as available observations.
    labels[:, calibration] = -1
    np.savez_compressed(
        run_dir / "exports" / "linear_preferences.npz", Z=Z, phi=phi,
        theta_true=theta, weights_true=weights, thresholds=thresholds,
        beta=np.asarray(beta), feature_center=center, feature_scale=scale,
        labels=labels, context_indices=context, user_ids=identifiers, user_split=splits,
        episode_split=episodes.split.to_numpy(dtype=str), episode_profile=episodes.profile.to_numpy(dtype=str),
        candidate=episodes.candidate.to_numpy(), feature_names=np.asarray(["bias", *FEATURE_NAMES]),
        feature_version=np.asarray(config["environment"].get("feature_version", 1)),
    )
    positive = np.take_along_axis(labels, context, axis=1).mean(axis=1)
    normal_count = n_train + n_test
    report = {
        "n_train_users": n_train, "n_test_users": n_test, "labels_per_user": limit,
        "n_probe_users": len(probe_names),
        "population": "single Gaussian in raw weights, joint Gaussian including calibrated bias",
        "probe_population_membership": False,
        "reward": "r_u = w_u @ phi; logit_u = beta * (r_u - threshold_u) = Z @ theta_u",
        "hidden_base_reward": False, "logit_identity_max_error": float(np.max(np.abs(logits - expected))),
        "context_positive_rate_mean": float(positive.mean()),
        "single_class_contexts": int(np.count_nonzero((positive == 0) | (positive == 1))),
        "calibration_oracle_profile_counts": {
            n: int(np.count_nonzero(winners[:normal_count] == i)) for i, n in enumerate(profile_names)
        },
        "probe_oracle_profiles": {
            name: profile_names[int(winners[normal_count + i])] for i, name in enumerate(probe_names)
        },
        "negative_competence_weights": int(
            np.count_nonzero((normal_weights[:, :2] <= 0).any(axis=1))
        ),
        "note": "Named probes are outside the fitted Gaussian population and test intentional edge preferences.",
    }
    write_json(run_dir / "reports" / "users.json", report)
    return report


def validate(config, run_dir):
    from reward.fully_bayesian.model import Population
    from sklearn.metrics import log_loss, roc_auc_score
    cfg = config["validation"]
    with np.load(run_dir / "exports" / "linear_preferences.npz", allow_pickle=False) as data:
        Z, labels, context = data["Z"], data["labels"], data["context_indices"]
        ids, splits = data["user_ids"], data["user_split"]
        features, truth = data["feature_names"].tolist(), data["theta_true"]
        heldout = data["episode_split"] == "test"
        optimization = data["episode_split"] == "calibration"
        candidate = data["candidate"]
        profiles = data["episode_profile"]
    train_indices = np.flatnonzero(splits == "train")[:cfg["n_train"]]
    test_indices = np.flatnonzero(splits == "test")[:cfg["n_test"]]
    probe_indices = np.flatnonzero(splits == "probe")
    model_cfg = SimpleNamespace(
        n_burnin=cfg["n_burnin"], n_samples=cfg["n_samples"], thin=1,
        niw_kappa0=0.1, niw_nu0=None, niw_lambda0_scale=1.0, eps_var=None,
        spike_slab=False, spike_slab_unit="feature", spike_slab_a=1., spike_slab_b=1.,
        seed=config["seed"], newuser_n_iters=cfg["newuser_n_iters"],
    )
    population = Population(model_cfg)
    usable = [i for i in train_indices if np.unique(labels[i, context[i]]).size == 2]
    population.fit([Z[context[i]] for i in usable], [labels[i, context[i]] for i in usable],
                   features, ids[usable].tolist(), features)
    profile_names = list(dict.fromkeys(profiles[candidate]))
    profile_Z_opt = np.asarray([Z[optimization & candidate & (profiles == n)].mean(axis=0) for n in profile_names])
    profile_Z_eval = np.asarray([Z[heldout & candidate & (profiles == n)].mean(axis=0) for n in profile_names])
    rows = []
    for i in test_indices:
        user = population.new_user(str(ids[i]))
        user.fit(Z[context[i]], labels[i, context[i]], rng=np.random.default_rng(config["seed"] + int(i)))
        probability = user.predict(Z[heldout])[0]
        theta = user.theta.mean(axis=0)
        oracle_scores = profile_Z_eval @ truth[i]
        best = int(np.argmax(profile_Z_opt @ theta))
        oracle = int(np.argmax(profile_Z_opt @ truth[i]))
        row = {"user_id": str(ids[i]), "log_loss": float(log_loss(labels[i, heldout], probability, labels=[0, 1])),
               "profile_correct": best == oracle, "regret_logit_units": float(oracle_scores[oracle] - oracle_scores[best])}
        if np.unique(labels[i, heldout]).size == 2:
            row["auroc"] = float(roc_auc_score(labels[i, heldout], probability))
        # Report competent-policy-only discrimination separately: failures must
        # not inflate evidence of personalized gait preference recovery.
        mask = heldout & candidate
        y = labels[i, mask]
        if np.unique(y).size == 2:
            row["competent_only_auroc"] = float(roc_auc_score(y, user.predict(Z[mask])[0]))
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(run_dir / "reports" / "bayesian_users.csv", index=False)
    probe_rows = []
    for i in probe_indices:
        y_context = labels[i, context[i]]
        if np.unique(y_context).size < 2:
            probe_rows.append({"user_id": str(ids[i]), "status": "single_class_context"})
            continue
        user = population.new_user(str(ids[i]))
        user.fit(Z[context[i]], y_context, rng=np.random.default_rng(config["seed"] + int(i)))
        inferred = int(np.argmax(profile_Z_opt @ user.theta.mean(axis=0)))
        oracle = int(np.argmax(profile_Z_opt @ truth[i]))
        probe_rows.append({
            "user_id": str(ids[i]), "status": "fit",
            "predicted_profile": profile_names[inferred], "oracle_profile": profile_names[oracle],
            "profile_correct": inferred == oracle,
        })
    report = {"validation_train_users": len(usable), "validation_test_users": len(rows),
              "generated_users": len(ids), "scope": "subset integration check, not a convergence study",
              "policy_selection_scenarios": "calibration", "policy_evaluation_scenarios": "test",
              "regret_note": "Both choices use calibration dynamics; held-out regret can be negative due to scenario sampling.",
              "metrics": table.select_dtypes(include=["number", "bool"]).mean().to_dict(),
              "probe_users": probe_rows}
    write_json(run_dir / "reports" / "bayesian_validation.json", report)
    return report


def videos(config, run_dir):
    import cv2
    from .record_policy_videos import write_video
    selected = read_json(run_dir / "policies" / "selected.json")
    directory = run_dir / "reports" / "videos"
    directory.mkdir(parents=True, exist_ok=True)
    # Same first held-out seed for every policy; no cherry-picking successes.
    seed = config["collection"]["test_seed"]
    horizon = config["environment"]["horizon"]
    comparison = []
    for row in selected:
        policy = load_policy(config, run_dir / row["model_path"], row["profile"])
        env = make_env(config, row["weights"], render_mode="rgb_array")
        try:
            e = rollout(policy, env, seed, horizon, video=True)
            fps = round(1 / env.unwrapped.dt)
            write_video(directory / f"{row['profile']}.mp4", e["frames"], fps)
            comparison.append((row["profile"], e["frames"]))
        finally:
            env.close()
    cols, panel = 3, 360
    height = panel * ((len(comparison) + cols - 1) // cols)
    writer = cv2.VideoWriter(str(directory / "gaits_comparison.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (cols * panel, height))
    if not writer.isOpened():
        raise RuntimeError("Could not open comparison video writer")
    try:
        for t in range(max(len(frames) for _, frames in comparison)):
            canvas = np.full((height, cols * panel, 3), 245, dtype=np.uint8)
            for i, (name, frames) in enumerate(comparison):
                frame = cv2.cvtColor(frames[min(t, len(frames)-1)], cv2.COLOR_RGB2BGR)
                frame = cv2.resize(frame, (panel, panel))
                cv2.putText(frame, name, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 2)
                row, col = divmod(i, cols)
                canvas[row*panel:(row+1)*panel, col*panel:(col+1)*panel] = frame
            writer.write(canvas)
            if t in (125, 300, 600):
                cv2.imwrite(str(directory / f"comparison_step_{t}.png"), canvas)
    finally:
        writer.release()
    return {"seed": seed, "horizon": horizon, "profiles": [name for name, _ in comparison]}
