from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path)
    if not config_path.is_absolute():
        candidate = Path.cwd() / config_path
        config_path = candidate if candidate.exists() else ROOT / config_path
    config_path = config_path.resolve()
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    return config, config_path


def config_digest(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def resolve_run_dir(config: dict[str, Any], run_id: str | None, fresh: bool) -> Path:
    data_root = Path(config.get("data_root", "data/runs"))
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    if run_id is None:
        run_id = str(config.get("name", "walker2d"))
        if fresh:
            candidate, index = run_id, 2
            while (data_root / candidate).exists():
                candidate = f"{run_id}_{index}"
                index += 1
            run_id = candidate
    run_dir = (data_root / run_id).resolve()
    if fresh and run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    for relative in ("policies", "rollouts", "tables", "exports", "reports"):
        (run_dir / relative).mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
    temporary.replace(path)


def runtime_versions() -> dict[str, str]:
    import gymnasium
    import mujoco
    import numpy
    import stable_baselines3

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "gymnasium": gymnasium.__version__,
        "mujoco": mujoco.__version__,
        "stable_baselines3": stable_baselines3.__version__,
    }
