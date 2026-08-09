from __future__ import annotations

import json
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np

from .policy_io import load_policy


def write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def write_comparison(path: Path, comparison: dict[str, list[np.ndarray]], fps: float) -> None:
    n_frames = max(len(frames) for frames in comparison.values())
    height, width = next(iter(comparison.values()))[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * len(comparison), height)
    )
    for frame_index in range(n_frames):
        panels = []
        for profile, frames in comparison.items():
            frame = frames[min(frame_index, len(frames) - 1)]
            panel = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).copy()
            cv2.putText(panel, profile, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(panel, profile, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            panels.append(panel)
        writer.write(np.hstack(panels))
    writer.release()


def record_policy_videos(
    config: dict,
    run_dir: Path,
    episodes: int = 2,
    all_checkpoints: bool = False,
    size: int = 480,
) -> Path:
    env_cfg = config["environment"]
    horizon = int(env_cfg.get("horizon", 500))
    action_noise_std = float(config["collection"].get("action_noise_std", 0.0))
    shared_profile_seeds = bool(config["collection"].get("shared_profile_seeds", False))
    base_seed = int(config.get("seed", 42)) + int(
        config["collection"].get("seed_offset", 100_000)
    )

    with (run_dir / "policies" / "index.json").open("r", encoding="utf-8") as f:
        checkpoints = json.load(f)
    if config.get("collection", {}).get("selected_only", False):
        checkpoints = [row for row in checkpoints if row.get("selected", False)]
    final_step = {row["profile"]: int(row["actual_step"]) for row in checkpoints}

    video_dir = run_dir / "reports" / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    # The reward wrapper only reshapes reward/info, so raw dynamics plus the
    # rollout seed formula reproduce the collected episodes exactly.
    env = gym.make(
        env_cfg.get("id", "Walker2d-v5"),
        render_mode="rgb_array",
        max_episode_steps=horizon,
        terminate_when_unhealthy=bool(env_cfg.get("terminate_when_unhealthy", True)),
        width=size,
        height=size,
    )
    fps = float(env.metadata["render_fps"])

    comparison: dict[str, list[np.ndarray]] = {}
    for checkpoint_index, checkpoint in enumerate(checkpoints):
        profile, step = checkpoint["profile"], int(checkpoint["actual_step"])
        if not all_checkpoints and step != final_step[profile]:
            continue
        model = load_policy(run_dir, checkpoint, device="cpu")
        for episode_index in range(episodes):
            seed_group = 0 if shared_profile_seeds else checkpoint_index
            episode_seed = base_seed + seed_group * 10_000 + episode_index
            rng = np.random.default_rng(episode_seed)
            observation, _ = env.reset(seed=episode_seed)
            frames = [np.asarray(env.render())]
            for _ in range(horizon):
                action, _ = model.predict(observation, deterministic=True)
                action = np.asarray(action, dtype=np.float32)
                if action_noise_std > 0:
                    action = action + rng.normal(0.0, action_noise_std, size=action.shape)
                action = np.clip(action, env.action_space.low, env.action_space.high)
                observation, _reward, terminated, truncated, _info = env.step(action)
                frames.append(np.asarray(env.render()))
                if terminated or truncated:
                    break
            episode_id = f"{profile}__{step:09d}__{episode_index:04d}"
            write_video(video_dir / f"{episode_id}.mp4", frames, fps)
            print(f"{episode_id}.mp4: {len(frames)} frames")
            if episode_index == 0 and step == final_step[profile]:
                comparison[profile] = frames
    env.close()

    if len(comparison) > 1:
        write_comparison(video_dir / "profiles_side_by_side.mp4", comparison, fps)
    return video_dir
