from __future__ import annotations

import shutil
from pathlib import Path

from stable_baselines3 import PPO

from .common import ROOT, write_json
from .policy_io import ActionBlendPolicy
from .reward_features import STYLE_NAMES
from .train_policies import _competence_pass, _evaluate_policy


def _source_path(value: str | Path) -> Path:
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Blend source policy does not exist: {path}")
    return path


def build_action_blend_bank(config: dict, run_dir: Path) -> list[dict]:
    blend_cfg = config["policy_blend"]
    competence_cfg = config.get("competence", {})
    device = str(blend_cfg.get("device", "cpu"))
    base_source = _source_path(blend_cfg["base_model"])
    style_source = _source_path(blend_cfg["style_model"])

    source_dir = run_dir / "policies" / "_sources"
    source_dir.mkdir(parents=True, exist_ok=False)
    base_path = source_dir / "base.zip"
    style_path = source_dir / "style.zip"
    shutil.copy2(base_source, base_path)
    shutil.copy2(style_source, style_path)

    base_model = PPO.load(base_path, device=device)
    style_model = PPO.load(style_path, device=device)
    n_episodes = int(competence_cfg.get("evaluation_episodes", 20))
    eval_seed = int(competence_cfg.get("evaluation_seed", int(config.get("seed", 42)) + 50_000))
    reward_mode = str(config.get("reward", {}).get("mode", "competence_style"))
    weights = {
        name: float(blend_cfg.get("style_weights", {}).get(name, 0.0))
        for name in STYLE_NAMES
    }

    base_metrics = _evaluate_policy(
        base_model, config, {}, reward_mode, n_episodes, eval_seed
    )
    base_metrics["passed"] = _competence_pass(base_metrics, competence_cfg)
    rows = [
        {
            "kind": "ppo",
            "profile": str(blend_cfg.get("baseline_profile", "baseline")),
            "profile_index": -1,
            "seed": int(config.get("seed", 42)),
            "requested_step": 0,
            "actual_step": 0,
            "model_path": str(base_path.relative_to(run_dir)),
            "weights": {name: 0.0 for name in STYLE_NAMES},
            "evaluation": base_metrics,
            "selected": bool(base_metrics["passed"]),
        }
    ]

    minimum_effect = float(blend_cfg.get("min_speed_improvement", 0.0))
    candidate_rows = []
    for index, alpha in enumerate(blend_cfg["alphas"]):
        alpha = float(alpha)
        policy = ActionBlendPolicy(base_model, style_model, alpha)
        metrics = _evaluate_policy(
            policy, config, weights, reward_mode, n_episodes, eval_seed
        )
        metrics["speed_improvement"] = float(
            metrics["mean_forward_speed"] - base_metrics["mean_forward_speed"]
        )
        metrics["passed"] = bool(
            _competence_pass(metrics, competence_cfg)
            and metrics["speed_improvement"] >= minimum_effect
        )
        profile = f"fast_blend_{round(alpha * 1000):03d}"
        row = {
            "kind": "action_blend",
            "profile": profile,
            "profile_index": index,
            "seed": int(config.get("seed", 42)),
            "requested_step": 0,
            "actual_step": 0,
            "base_model_path": str(base_path.relative_to(run_dir)),
            "style_model_path": str(style_path.relative_to(run_dir)),
            "alpha": alpha,
            "weights": weights,
            "evaluation": metrics,
            "selected": False,
        }
        candidate_rows.append(row)
        profile_dir = run_dir / "policies" / profile
        profile_dir.mkdir(parents=True, exist_ok=False)
        write_json(profile_dir / "metadata.json", row)

    passing = [row for row in candidate_rows if row["evaluation"]["passed"]]
    if passing:
        selected = max(passing, key=lambda row: row["evaluation"]["mean_forward_speed"])
        selected["selected"] = True
        write_json(run_dir / "policies" / selected["profile"] / "metadata.json", selected)
    rows.extend(candidate_rows)
    write_json(run_dir / "policies" / "index.json", rows)
    write_json(
        run_dir / "reports" / "blend_evaluation.json",
        {
            "gate": competence_cfg,
            "minimum_speed_improvement": minimum_effect,
            "baseline": rows[0],
            "candidates": candidate_rows,
            "selected_profile": selected["profile"] if passing else None,
        },
    )
    if not base_metrics["passed"]:
        raise RuntimeError("Blend baseline failed competence gate")
    if not passing:
        raise RuntimeError("No action-blend candidate passed competence and preference gates")
    return rows
