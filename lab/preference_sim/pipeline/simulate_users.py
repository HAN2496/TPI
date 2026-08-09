from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import write_json
from .reward_features import FEATURE_NAMES


DEFAULT_ARCHETYPES = {
    "sprinter": {"speed_mean": 2.0, "smoothness_rms": -0.3, "survival": 1.0},
    "strider": {"clearance_mean": 1.5, "cadence": -1.0, "speed_mean": 0.5, "survival": 1.0},
    "shuffler": {"clearance_mean": -1.2, "cadence": 1.5, "survival": 1.0},
    "bouncer": {"airtime_frac": 1.8, "bounce_rms": 0.8, "survival": 0.8},
    "glider": {"bounce_rms": -1.8, "smoothness_rms": -0.8, "survival": 1.0},
    "hopper": {"sync_frac": 1.8, "airtime_frac": 0.6, "survival": 0.8},
    "efficient": {"energy_mean": -1.8, "speed_mean": 0.4, "survival": 1.0},
}


def _archetype_center(weights: dict[str, float]) -> np.ndarray:
    unknown = set(weights) - set(FEATURE_NAMES)
    if unknown:
        raise KeyError(f"Unknown archetype features: {sorted(unknown)}")
    return np.asarray([float(weights.get(name, 0.0)) for name in FEATURE_NAMES])


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-value))


def _ensure_two_classes(order: np.ndarray, labels: np.ndarray, limit: int) -> np.ndarray:
    limit = min(limit, len(order))
    if limit < 2 or len(np.unique(labels[order[:limit]])) == 2:
        return order
    missing = 1 - int(labels[order[0]])
    candidates = np.flatnonzero(labels[order[limit:]] == missing)
    if len(candidates):
        swap_position = limit + int(candidates[0])
        order[limit - 1], order[swap_position] = order[swap_position], order[limit - 1]
    return order


def simulate_users_and_feedback(config: dict, run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    feedback_cfg = config["feedback"]
    episodes = pd.read_csv(run_dir / "tables" / "episodes.csv")
    feature_columns = [f"feature_{name}" for name in FEATURE_NAMES]
    features = episodes[feature_columns].to_numpy(dtype=np.float64)
    feature_mean = features.mean(axis=0)
    feature_sd = features.std(axis=0)
    feature_sd[feature_sd < 1e-8] = 1.0
    standardized = (features - feature_mean) / feature_sd

    archetypes = feedback_cfg.get("archetypes", DEFAULT_ARCHETYPES)
    archetype_names = list(archetypes)
    centers = {name: _archetype_center(archetypes[name]) for name in archetype_names}

    n_train = int(feedback_cfg.get("n_train_users", 20))
    n_test = int(feedback_cfg.get("n_test_users", 5))
    n_users = n_train + n_test
    train_limit = min(int(feedback_cfg.get("train_labels_per_user", 40)), len(episodes))
    test_limit = min(int(feedback_cfg.get("max_test_context", 40)), len(episodes))
    theta_sd = float(feedback_cfg.get("theta_sd", 0.15))
    beta = float(feedback_cfg.get("beta", 1.0))
    threshold_quantile = float(feedback_cfg.get("threshold_quantile", 0.5))
    rng = np.random.default_rng(int(config.get("seed", 42)) + 200_000)

    user_rows: list[dict] = []
    feedback_rows: list[dict] = []
    episode_ids = episodes["episode_id"].astype(str).to_numpy()
    for user_index in range(n_users):
        split = "train" if user_index < n_train else "test"
        archetype = archetype_names[user_index % len(archetype_names)]
        theta = centers[archetype] + rng.normal(0.0, theta_sd, len(FEATURE_NAMES))
        theta = theta / np.linalg.norm(theta) * np.sqrt(len(theta))
        utility_without_threshold = standardized @ theta
        threshold = float(np.quantile(utility_without_threshold, threshold_quantile))
        probability = _sigmoid(beta * (utility_without_threshold - threshold))
        labels = (rng.random(len(episodes)) < probability).astype(np.int8)
        order = rng.permutation(len(episodes))
        context_limit = train_limit if split == "train" else test_limit
        order = _ensure_two_classes(order, labels, context_limit)
        query_rank = np.empty(len(episodes), dtype=np.int32)
        query_rank[order] = np.arange(1, len(episodes) + 1)
        user_id = f"{split}_{user_index:03d}_{archetype}"

        user_row = {
            "user_id": user_id,
            "split": split,
            "archetype": archetype,
            "threshold": threshold,
            "beta": beta,
            "context_limit": context_limit,
            "positive_rate_all": float(labels.mean()),
            "positive_rate_context": float(labels[order[:context_limit]].mean()),
        }
        user_row.update(
            {f"theta_{name}": float(value) for name, value in zip(FEATURE_NAMES, theta)}
        )
        user_rows.append(user_row)

        for episode_index, episode_id in enumerate(episode_ids):
            rank = int(query_rank[episode_index])
            feedback_rows.append(
                {
                    "user_id": user_id,
                    "episode_id": episode_id,
                    "label": int(labels[episode_index]),
                    "query_order": rank,
                    "is_context": bool(rank <= context_limit),
                    "utility": float(utility_without_threshold[episode_index] - threshold),
                    "probability": float(probability[episode_index]),
                }
            )

    users = pd.DataFrame(user_rows)
    feedback = pd.DataFrame(feedback_rows)
    users.to_csv(run_dir / "tables" / "users.csv", index=False)
    feedback.to_csv(run_dir / "tables" / "feedback.csv", index=False)
    write_json(
        run_dir / "tables" / "feature_scaler.json",
        {"feature_names": list(FEATURE_NAMES), "mean": feature_mean.tolist(), "sd": feature_sd.tolist()},
    )
    write_json(
        run_dir / "reports" / "feedback_summary.json",
        {
            "n_users": n_users,
            "n_train_users": n_train,
            "n_test_users": n_test,
            "n_feedback_rows": int(len(feedback)),
            "mean_positive_rate": float(users["positive_rate_all"].mean()),
            "min_context_positive_rate": float(users["positive_rate_context"].min()),
            "max_context_positive_rate": float(users["positive_rate_context"].max()),
        },
    )
    return users, feedback
