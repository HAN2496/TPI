from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO


class ActionBlendPolicy:
    """Blend two policies at the action level without changing either model."""

    def __init__(self, base: PPO, style: PPO, alpha: float) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Blend alpha must be in [0, 1], got {alpha}")
        self.base = base
        self.style = style
        self.alpha = float(alpha)

    def predict(self, observation, deterministic: bool = True):
        base_action, _ = self.base.predict(observation, deterministic=deterministic)
        style_action, _ = self.style.predict(observation, deterministic=deterministic)
        action = (1.0 - self.alpha) * np.asarray(base_action) + self.alpha * np.asarray(style_action)
        return action.astype(np.float32), None


def _resolve_model_path(run_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else run_dir / path


def load_policy(run_dir: Path, row: dict, device: str = "cpu"):
    kind = str(row.get("kind", "ppo"))
    if kind == "ppo":
        return PPO.load(_resolve_model_path(run_dir, row["model_path"]), device=device)
    if kind == "action_blend":
        base = PPO.load(_resolve_model_path(run_dir, row["base_model_path"]), device=device)
        style = PPO.load(_resolve_model_path(run_dir, row["style_model_path"]), device=device)
        return ActionBlendPolicy(base, style, float(row["alpha"]))
    raise ValueError(f"Unknown policy kind: {kind}")
