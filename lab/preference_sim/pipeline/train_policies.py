from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

from .common import ROOT, write_json
from .reward_features import SIGNAL_NAMES, STYLE_NAMES, make_walker2d_env


def _new_ppo(config: dict, env, seed: int) -> PPO:
    training = config["training"]
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=float(training.get("learning_rate", 3e-4)),
        n_steps=int(training.get("n_steps", 512)),
        batch_size=int(training.get("batch_size", 64)),
        n_epochs=int(training.get("n_epochs", 5)),
        gamma=float(training.get("gamma", 0.99)),
        gae_lambda=float(training.get("gae_lambda", 0.95)),
        ent_coef=float(training.get("ent_coef", 0.0)),
        policy_kwargs={"net_arch": list(training.get("net_arch", [128, 128]))},
        seed=seed,
        device=str(training.get("device", "cpu")),
        verbose=int(training.get("verbose", 0)),
    )


def sweep_profiles(styles_cfg: dict) -> dict[str, dict[str, float]]:
    """Expand archetypes plus pairwise interpolations into a policy sweep."""
    archetypes = {name: dict(weights) for name, weights in styles_cfg["archetypes"].items()}
    profiles = dict(archetypes)
    for left, right in styles_cfg.get("pairs", []):
        for alpha in styles_cfg.get("pair_alphas", [0.5]):
            blend = {
                name: alpha * archetypes[left].get(name, 0.0)
                + (1.0 - alpha) * archetypes[right].get(name, 0.0)
                for name in set(archetypes[left]) | set(archetypes[right])
            }
            profiles[f"{left}{round(alpha * 100):02d}_{right}{round((1 - alpha) * 100):02d}"] = blend
    return profiles


def _model_zip_path(path: Path) -> Path:
    return path if path.suffix == ".zip" else path.with_suffix(".zip")


def _evaluate_policy(
    model: PPO,
    config: dict,
    style_weights: dict[str, float],
    reward_mode: str,
    n_episodes: int,
    seed: int,
) -> dict:
    env = make_walker2d_env(config, style_weights, reward_mode=reward_mode)
    horizon = int(config["environment"].get("horizon", 500))
    lengths: list[int] = []
    speeds: list[float] = []
    energies: list[float] = []
    action_deltas: list[float] = []
    returns: list[float] = []
    terminations: list[bool] = []
    style_totals = np.zeros(len(STYLE_NAMES), dtype=np.float64)
    n_valid_steps = 0
    for episode_index in range(n_episodes):
        observation, _ = env.reset(seed=seed + episode_index)
        episode_return = 0.0
        terminated = truncated = False
        length = 0
        for _ in range(horizon):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            signal = np.asarray(info["preference_signals"], dtype=np.float64)
            signal_index = {name: index for index, name in enumerate(SIGNAL_NAMES)}
            speeds.append(float(signal[signal_index["forward_speed"]]))
            energies.append(float(signal[signal_index["control_energy"]]))
            action_deltas.append(float(signal[signal_index["action_delta"]]))
            style_totals += np.asarray(info["preference_components"], dtype=np.float64)
            n_valid_steps += 1
            episode_return += float(reward)
            length += 1
            if terminated or truncated:
                break
        lengths.append(length)
        returns.append(episode_return)
        terminations.append(bool(terminated))
    env.close()
    style_means = style_totals / max(n_valid_steps, 1)
    return {
        "n_episodes": n_episodes,
        "completion_rate": float(np.mean(np.asarray(lengths) >= horizon)),
        "fall_rate": float(np.mean(terminations)),
        "mean_length": float(np.mean(lengths)),
        "mean_length_fraction": float(np.mean(lengths) / horizon),
        "mean_forward_speed": float(np.mean(speeds)),
        "mean_control_energy": float(np.mean(energies)),
        "rms_action_delta": float(np.sqrt(np.mean(np.square(action_deltas)))),
        "mean_return": float(np.mean(returns)),
        "style_means": {
            name: float(value) for name, value in zip(STYLE_NAMES, style_means)
        },
    }


def _competence_pass(metrics: dict, gate: dict) -> bool:
    return bool(
        metrics["completion_rate"] >= float(gate.get("min_completion_rate", 0.9))
        and metrics["mean_length_fraction"] >= float(gate.get("min_mean_length_fraction", 0.95))
        and metrics["mean_forward_speed"] >= float(gate.get("min_forward_speed", 0.5))
    )


def train_policy_bank(config: dict, run_dir: Path) -> list[dict]:
    training = config["training"]
    seed = int(config.get("seed", 42))
    torch.set_num_threads(int(training.get("torch_num_threads", 1)))
    set_random_seed(seed)

    targets = sorted({int(step) for step in training["checkpoint_steps"]})
    if not targets or targets[0] <= 0:
        raise ValueError("training.checkpoint_steps must contain positive integers")

    pretrain_steps = int(training.get("pretrain_steps", 0))
    competence_cfg = config.get("competence", {})
    evaluation_episodes = int(competence_cfg.get("evaluation_episodes", 20))
    evaluation_seed = int(competence_cfg.get("evaluation_seed", seed + 50_000))
    base_model_path = None
    base_step = 0
    pretrain_history: list[dict] = []
    pretrained_model = training.get("pretrained_model")
    if pretrained_model:
        base_model_path = Path(pretrained_model)
        if not base_model_path.is_absolute():
            base_model_path = ROOT / base_model_path
        base_model_path = _model_zip_path(base_model_path)
        if not base_model_path.exists():
            raise FileNotFoundError(f"Pretrained model does not exist: {base_model_path}")
        evaluation_env = make_walker2d_env(config, {}, reward_mode="original")
        base_model = PPO.load(base_model_path, env=evaluation_env, device=training.get("device", "cpu"))
        base_step = int(base_model.num_timesteps)
        metrics = _evaluate_policy(
            base_model, config, {}, "original", evaluation_episodes, evaluation_seed
        )
        metrics.update({"step": base_step, "checkpoint": str(base_model_path), "passed": _competence_pass(metrics, competence_cfg)})
        pretrain_history.append(metrics)
        evaluation_env.close()
        if competence_cfg.get("required", True) and not metrics["passed"]:
            write_json(run_dir / "reports" / "pretrain_evaluation.json", {"history": pretrain_history})
            raise RuntimeError(f"Pretrained policy failed competence gate: {metrics}")
    elif pretrain_steps > 0:
        base_dir = run_dir / "policies" / "_pretrained"
        base_dir.mkdir(parents=True, exist_ok=False)
        base_env = make_walker2d_env(config, {}, reward_mode="original")
        base_model = _new_ppo(config, base_env, seed)
        pretrain_targets = sorted({
            int(value) for value in training.get("pretrain_checkpoints", [pretrain_steps])
            if int(value) > 0 and int(value) <= pretrain_steps
        })
        if not pretrain_targets or pretrain_targets[-1] != pretrain_steps:
            pretrain_targets.append(pretrain_steps)
        for target in pretrain_targets:
            remaining = target - int(base_model.num_timesteps)
            if remaining <= 0:
                continue
            base_model.learn(total_timesteps=remaining, reset_num_timesteps=False, progress_bar=False)
            base_step = int(base_model.num_timesteps)
            candidate_path = base_dir / f"step_{base_step:09d}"
            base_model.save(candidate_path)
            metrics = _evaluate_policy(
                base_model, config, {}, "original", evaluation_episodes, evaluation_seed
            )
            passed = _competence_pass(metrics, competence_cfg)
            metrics.update({"step": base_step, "checkpoint": str(candidate_path.with_suffix('.zip').relative_to(run_dir)), "passed": passed})
            pretrain_history.append(metrics)
            write_json(
                candidate_path.with_suffix(".json"),
                {
                    "kind": "shared_pretraining",
                    "requested_step": target,
                    "actual_step": base_step,
                    "reward_mode": "original",
                    "evaluation": metrics,
                },
            )
            base_model_path = candidate_path.with_suffix(".zip")
            if passed and competence_cfg.get("stop_pretraining_when_passed", True):
                break
        base_env.close()
        if competence_cfg.get("required", True) and not pretrain_history[-1]["passed"]:
            write_json(run_dir / "reports" / "pretrain_evaluation.json", {"history": pretrain_history})
            raise RuntimeError(f"Base policy failed competence gate after {base_step} steps")

    write_json(
        run_dir / "reports" / "pretrain_evaluation.json",
        {"gate": competence_cfg, "history": pretrain_history},
    )

    profiles = sweep_profiles(config["styles"])
    checkpoint_rows: list[dict] = []
    profile_evaluations: list[dict] = []
    failed_profiles: list[str] = []
    reward_mode = str(config.get("reward", {}).get("mode", "competence_style"))
    profile_seed_stride = int(training.get("profile_seed_stride", 1000))

    baseline_profile = training.get("pretrained_baseline_profile")
    if baseline_profile:
        if base_model_path is None:
            raise ValueError("training.pretrained_baseline_profile requires a pretrained model")
        baseline_dir = run_dir / "policies" / str(baseline_profile)
        baseline_dir.mkdir(parents=True, exist_ok=False)
        baseline_env = make_walker2d_env(config, {}, reward_mode=reward_mode)
        baseline_model = PPO.load(
            _model_zip_path(base_model_path), env=baseline_env, device=training.get("device", "cpu")
        )
        baseline_path = baseline_dir / "step_000000000"
        baseline_model.save(baseline_path)
        evaluation = _evaluate_policy(
            baseline_model, config, {}, reward_mode, evaluation_episodes, evaluation_seed
        )
        evaluation["passed"] = _competence_pass(evaluation, competence_cfg)
        row = {
            "profile": str(baseline_profile),
            "profile_index": -1,
            "seed": seed,
            "requested_step": 0,
            "actual_step": 0,
            "pretrained_step": base_step,
            "model_path": str(baseline_path.with_suffix(".zip").relative_to(run_dir)),
            "weights": {name: 0.0 for name in STYLE_NAMES},
            "evaluation": evaluation,
            "selected": bool(evaluation["passed"]),
        }
        write_json(baseline_path.with_suffix(".json"), row)
        checkpoint_rows.append(row)
        profile_evaluations.append(
            {"profile": str(baseline_profile), "step": 0, **evaluation}
        )
        if competence_cfg.get("require_final_profile_gate", True) and not evaluation["passed"]:
            failed_profiles.append(str(baseline_profile))
        baseline_env.close()

    for profile_index, (profile_name, weights) in enumerate(profiles.items()):
        first_row_index = len(checkpoint_rows)
        profile_dir = run_dir / "policies" / profile_name
        profile_dir.mkdir(parents=True, exist_ok=False)
        profile_seed = seed + profile_seed_stride * profile_index
        env = make_walker2d_env(config, weights, reward_mode=reward_mode)
        if base_model_path is None:
            model = _new_ppo(config, env, profile_seed)
        else:
            model = PPO.load(_model_zip_path(base_model_path), env=env, device=training.get("device", "cpu"))
            model.set_random_seed(profile_seed)

        profile_step = 0
        for target in targets:
            remaining = target - profile_step
            if remaining <= 0:
                continue
            model.learn(
                total_timesteps=remaining,
                reset_num_timesteps=(profile_step == 0),
                progress_bar=False,
            )
            actual_step = int(model.num_timesteps)
            model_path = profile_dir / f"step_{actual_step:09d}"
            model.save(model_path)
            evaluation = _evaluate_policy(
                model, config, weights, reward_mode, evaluation_episodes, evaluation_seed
            )
            evaluation["passed"] = _competence_pass(evaluation, competence_cfg)
            profile_evaluations.append(
                {"profile": profile_name, "step": actual_step, **evaluation}
            )
            row = {
                "profile": profile_name,
                "profile_index": profile_index,
                "seed": profile_seed,
                "requested_step": target,
                "actual_step": actual_step,
                "pretrained_step": base_step,
                "model_path": str(model_path.with_suffix(".zip").relative_to(run_dir)),
                "weights": {
                    name: float(weights.get(name, 0.0)) for name in STYLE_NAMES
                },
                "evaluation": evaluation,
            }
            write_json(model_path.with_suffix(".json"), row)
            checkpoint_rows.append(row)
            profile_step = actual_step
        profile_rows = checkpoint_rows[first_row_index:]
        passing_rows = [row for row in profile_rows if row["evaluation"]["passed"]]
        selection_cfg = training.get("checkpoint_selection", {})
        selection_metric = selection_cfg.get("metric")
        if passing_rows and selection_metric:
            reverse = str(selection_cfg.get("mode", "max")) == "max"
            selected_row = sorted(
                passing_rows,
                key=lambda row: float(row["evaluation"][str(selection_metric)]),
                reverse=reverse,
            )[0]
        else:
            selected_row = passing_rows[-1] if passing_rows else profile_rows[-1]
        for row in profile_rows:
            row["selected"] = row is selected_row and bool(passing_rows)
            model_json = run_dir / Path(row["model_path"]).with_suffix(".json")
            write_json(model_json, row)
        if competence_cfg.get("require_final_profile_gate", True) and not passing_rows:
            failed_profiles.append(profile_name)
        env.close()

    write_json(run_dir / "policies" / "index.json", checkpoint_rows)
    write_json(
        run_dir / "reports" / "policy_evaluation.json",
        {"gate": competence_cfg, "evaluations": profile_evaluations, "failed_profiles": failed_profiles},
    )
    if failed_profiles:
        raise RuntimeError(f"Final policy competence gate failed: {', '.join(failed_profiles)}")
    return checkpoint_rows
