from __future__ import annotations

import json
from pathlib import Path

from .common import write_json
from .policy_io import load_policy
from .train_policies import _competence_pass, _evaluate_policy


def _write_row_metadata(run_dir: Path, row: dict) -> None:
    if row.get("kind", "ppo") == "ppo":
        path = run_dir / Path(row["model_path"]).with_suffix(".json")
    else:
        path = run_dir / "policies" / row["profile"] / "metadata.json"
    write_json(path, row)


def select_policy_checkpoints(config: dict, run_dir: Path) -> dict:
    with (run_dir / "policies" / "index.json").open("r", encoding="utf-8") as f:
        rows = json.load(f)

    selection_cfg = config["training"]["checkpoint_selection"]
    metric = str(selection_cfg["metric"])
    mode = str(selection_cfg.get("mode", "max"))
    n_episodes = int(selection_cfg.get("validation_episodes", 50))
    seed = int(selection_cfg.get("validation_seed", int(config.get("seed", 42)) + 100_000))
    device = str(config["training"].get("device", "cpu"))
    reward_mode = str(config.get("reward", {}).get("mode", "competence_style"))
    competence_cfg = config.get("competence", {})

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["profile"]), []).append(row)

    selected_profiles = {}
    failures = []
    for profile, profile_rows in grouped.items():
        candidates = [row for row in profile_rows if row.get("evaluation", {}).get("passed", False)]
        if not candidates:
            failures.append(profile)
            continue
        validation_passing = []
        for row in candidates:
            policy = load_policy(run_dir, row, device=device)
            metrics = _evaluate_policy(
                policy, config, row.get("weights", {}), reward_mode, n_episodes, seed
            )
            metrics["passed"] = _competence_pass(metrics, competence_cfg)
            row["selection_validation"] = metrics
            row["selected"] = False
            if metrics["passed"]:
                validation_passing.append(row)
        if not validation_passing:
            failures.append(profile)
            continue
        selected = sorted(
            validation_passing,
            key=lambda row: float(row["selection_validation"][metric]),
            reverse=(mode == "max"),
        )[0]
        selected["selected"] = True
        selected_profiles[profile] = {
            "model": selected.get("model_path", selected.get("kind")),
            "metric": metric,
            "value": float(selected["selection_validation"][metric]),
        }

    for row in rows:
        _write_row_metadata(run_dir, row)
    write_json(run_dir / "policies" / "index.json", rows)
    report = {
        "validation_episodes": n_episodes,
        "validation_seed": seed,
        "metric": metric,
        "mode": mode,
        "selected_profiles": selected_profiles,
        "failed_profiles": failures,
    }
    write_json(run_dir / "reports" / "policy_selection.json", report)
    if failures:
        raise RuntimeError("No validation-passing checkpoint for: " + ", ".join(failures))
    return report
