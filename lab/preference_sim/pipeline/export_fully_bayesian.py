from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import write_json
from .reward_features import FEATURE_NAMES, SIGNAL_NAMES


def _load_rollouts(run_dir: Path) -> dict[str, np.ndarray]:
    arrays: dict[str, list[np.ndarray]] = {
        "episode_ids": [], "observations": [], "actions": [], "signals": [],
        "reward_components": [], "valid": [], "episode_features": [],
    }
    for path in sorted((run_dir / "rollouts").glob("*.npz")):
        with np.load(path, allow_pickle=False) as shard:
            for key in arrays:
                arrays[key].append(shard[key])
    if not arrays["episode_ids"]:
        raise FileNotFoundError(f"No rollout shards found in {run_dir / 'rollouts'}")
    return {key: np.concatenate(parts, axis=0) for key, parts in arrays.items()}


def export_fully_bayesian(config: dict, run_dir: Path) -> Path:
    rollouts = _load_rollouts(run_dir)
    episodes = pd.read_csv(run_dir / "tables" / "episodes.csv")
    users = pd.read_csv(run_dir / "tables" / "users.csv")
    feedback = pd.read_csv(run_dir / "tables" / "feedback.csv")
    with (run_dir / "tables" / "feature_scaler.json").open("r", encoding="utf-8") as f:
        scaler = json.load(f)

    rollout_ids = [str(value) for value in rollouts["episode_ids"]]
    rollout_index = {episode_id: index for index, episode_id in enumerate(rollout_ids)}
    episode_ids = episodes["episode_id"].astype(str).tolist()
    order = np.asarray([rollout_index[episode_id] for episode_id in episode_ids], dtype=np.int64)
    for key in rollouts:
        rollouts[key] = rollouts[key][order]

    user_ids = users["user_id"].astype(str).tolist()
    user_index = {user_id: index for index, user_id in enumerate(user_ids)}
    episode_index = {episode_id: index for index, episode_id in enumerate(episode_ids)}
    labels = np.full((len(user_ids), len(episode_ids)), -1, dtype=np.int8)
    query_order = np.zeros((len(user_ids), len(episode_ids)), dtype=np.int32)
    context_mask = np.zeros((len(user_ids), len(episode_ids)), dtype=bool)
    for row in feedback.itertuples(index=False):
        u = user_index[str(row.user_id)]
        e = episode_index[str(row.episode_id)]
        labels[u, e] = int(row.label)
        query_order[u, e] = int(row.query_order)
        context_mask[u, e] = bool(row.is_context)
    if np.any(labels < 0):
        raise ValueError("Feedback matrix is incomplete")

    theta_true = users[[f"theta_{name}" for name in FEATURE_NAMES]].to_numpy(dtype=np.float32)
    feature_mean = np.asarray(scaler["mean"], dtype=np.float32)
    feature_sd = np.asarray(scaler["sd"], dtype=np.float32)
    episode_features = rollouts["episode_features"].astype(np.float32)
    episode_features_z = (episode_features - feature_mean) / feature_sd

    export_path = run_dir / "exports" / "fully_bayesian_input.npz"
    np.savez_compressed(
        export_path,
        X=rollouts["signals"].astype(np.float32),
        observations=rollouts["observations"].astype(np.float32),
        actions=rollouts["actions"].astype(np.float32),
        valid=rollouts["valid"].astype(bool),
        episode_features=episode_features,
        episode_features_z=episode_features_z.astype(np.float32),
        labels=labels,
        query_order=query_order,
        context_mask=context_mask,
        episode_ids=np.asarray(episode_ids, dtype=str),
        user_ids=np.asarray(user_ids, dtype=str),
        user_split=np.asarray(users["split"].astype(str).tolist(), dtype=str),
        theta_true=theta_true,
        signal_names=np.asarray(SIGNAL_NAMES),
        reward_feature_names=np.asarray(FEATURE_NAMES),
    )
    write_json(
        run_dir / "exports" / "fully_bayesian_input.json",
        {
            "npz": export_path.name,
            "X_shape": list(rollouts["signals"].shape),
            "labels_shape": list(labels.shape),
            "X_semantics": "shared trajectory signals [episode, time, channel]",
            "labels_semantics": "binary feedback [user, episode]",
            "feature_names": list(FEATURE_NAMES),
            "note": "Use context_mask or query_order to construct per-user context sets.",
        },
    )
    return export_path
