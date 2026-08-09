from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import write_json
from .policy_io import load_policy
from .reward_features import FEATURE_NAMES, SIGNAL_NAMES, STYLE_NAMES, make_walker2d_env


def _load_checkpoint_index(run_dir: Path) -> list[dict]:
    with (run_dir / "policies" / "index.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def _padded_episode(
    observations: list[np.ndarray],
    actions: list[np.ndarray],
    signals: list[np.ndarray],
    components: list[np.ndarray],
    horizon: int,
    terminated: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    length = len(signals)
    if length == 0:
        raise ValueError("Cannot pad an empty episode")
    obs_out = np.zeros((horizon, observations[0].shape[0]), dtype=np.float32)
    action_out = np.zeros((horizon, actions[0].shape[0]), dtype=np.float32)
    signal_out = np.zeros((horizon, len(SIGNAL_NAMES)), dtype=np.float32)
    component_out = np.zeros((horizon, len(STYLE_NAMES)), dtype=np.float32)
    valid = np.zeros(horizon, dtype=bool)

    obs_out[:length] = np.asarray(observations, dtype=np.float32)
    action_out[:length] = np.asarray(actions, dtype=np.float32)
    signal_out[:length] = np.asarray(signals, dtype=np.float32)
    component_out[:length] = np.asarray(components, dtype=np.float32)
    valid[:length] = True

    if length < horizon:
        # Freeze the final pose, zero out the dynamic channels, keep the fall flag.
        obs_out[length:] = obs_out[length - 1]
        signal_out[length:] = signal_out[length - 1]
        dynamic = [SIGNAL_NAMES.index(name) for name in
                   ("forward_speed", "control_energy", "vertical_speed", "action_delta", "contact_force")]
        signal_out[length:, dynamic] = 0.0
        signal_out[length:, SIGNAL_NAMES.index("fall_indicator")] = float(terminated)
    return obs_out, action_out, signal_out, component_out, valid


def _episode_features(
    signals: np.ndarray,
    valid: np.ndarray,
    dt: float,
    contact_force_scale: float,
    horizon: int,
) -> np.ndarray:
    idx = {name: index for index, name in enumerate(SIGNAL_NAMES)}
    s = signals[valid]
    contact_right = s[:, idx["foot_contact_right"]] > 0.5
    contact_left = s[:, idx["foot_contact_left"]] > 0.5
    transitions = np.count_nonzero(np.diff(contact_right)) + np.count_nonzero(np.diff(contact_left))
    values = [
        s[:, idx["forward_speed"]].mean(),
        s[:, idx["control_energy"]].mean(),
        s[:, idx["torso_angle"]].mean(),
        s[:, idx["torso_height"]].mean(),
        np.sqrt(np.mean(s[:, idx["vertical_speed"]] ** 2)),
        np.maximum(s[:, idx["foot_height_right"]], s[:, idx["foot_height_left"]]).mean(),
        np.mean(~contact_right & ~contact_left),
        np.mean(contact_right == contact_left),
        transitions / (2.0 * len(s) * dt),
        np.sqrt(np.mean(s[:, idx["action_delta"]] ** 2)),
        np.percentile(s[:, idx["contact_force"]], 95) / contact_force_scale,
        len(s) / horizon,
    ]
    return np.asarray(values, dtype=np.float32)


def collect_policy_bank(config: dict, run_dir: Path) -> pd.DataFrame:
    collection = config["collection"]
    horizon = int(config["environment"].get("horizon", 500))
    episodes_per_checkpoint = int(collection.get("episodes_per_checkpoint", 10))
    action_noise_std = float(collection.get("action_noise_std", 0.0))
    contact_force_scale = float(config["environment"].get("contact_force_scale", 1000.0))
    base_seed = int(config.get("seed", 42)) + int(collection.get("seed_offset", 100_000))
    checkpoint_rows = _load_checkpoint_index(run_dir)
    if collection.get("selected_only", False):
        checkpoint_rows = [row for row in checkpoint_rows if row.get("selected", False)]
        if not checkpoint_rows:
            raise RuntimeError("No competence-passing policy checkpoints were selected")
    metadata_rows: list[dict] = []

    for checkpoint_index, checkpoint in enumerate(checkpoint_rows):
        profile = checkpoint["profile"]
        actual_step = int(checkpoint["actual_step"])
        weights = checkpoint["weights"]
        env = make_walker2d_env(config, weights)
        dt = float(env.unwrapped.dt)
        model = load_policy(run_dir, checkpoint, device=str(collection.get("device", "cpu")))

        batches = {
            "observations": [], "actions": [], "signals": [], "components": [],
            "valid": [], "features": [], "episode_ids": [], "seeds": [],
            "lengths": [], "terminated": [], "truncated": [],
        }
        for episode_index in range(episodes_per_checkpoint):
            seed_group = 0 if collection.get("shared_profile_seeds", False) else checkpoint_index
            episode_seed = base_seed + seed_group * 10_000 + episode_index
            rng = np.random.default_rng(episode_seed)
            observation, _ = env.reset(seed=episode_seed)
            observations: list[np.ndarray] = []
            actions: list[np.ndarray] = []
            signals: list[np.ndarray] = []
            components: list[np.ndarray] = []
            terminated = truncated = False

            for _ in range(horizon):
                action, _ = model.predict(observation, deterministic=True)
                action = np.asarray(action, dtype=np.float32)
                if action_noise_std > 0:
                    action = action + rng.normal(0.0, action_noise_std, size=action.shape)
                action = np.clip(action, env.action_space.low, env.action_space.high)
                observation, _reward, terminated, truncated, info = env.step(action)
                observations.append(np.asarray(observation, dtype=np.float32))
                actions.append(action.astype(np.float32))
                signals.append(np.asarray(info["preference_signals"], dtype=np.float32))
                components.append(np.asarray(info["preference_components"], dtype=np.float32))
                if terminated or truncated:
                    break

            episode_id = f"{profile}__{actual_step:09d}__{episode_index:04d}"
            obs_array, action_array, signal_array, component_array, valid_array = (
                _padded_episode(observations, actions, signals, components, horizon, bool(terminated))
            )
            feature_array = _episode_features(
                signal_array, valid_array, dt, contact_force_scale, horizon
            )
            batches["observations"].append(obs_array)
            batches["actions"].append(action_array)
            batches["signals"].append(signal_array)
            batches["components"].append(component_array)
            batches["valid"].append(valid_array)
            batches["features"].append(feature_array)
            batches["episode_ids"].append(episode_id)
            batches["seeds"].append(episode_seed)
            batches["lengths"].append(len(signals))
            batches["terminated"].append(bool(terminated))
            batches["truncated"].append(bool(truncated))

            row = {
                "episode_id": episode_id,
                "profile": profile,
                "policy_step": actual_step,
                "seed": episode_seed,
                "length": len(signals),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "policy_weights": json.dumps(weights, sort_keys=True),
            }
            row.update(
                {f"feature_{name}": float(value) for name, value in zip(FEATURE_NAMES, feature_array)}
            )
            realized = component_array[valid_array].mean(axis=0)
            row.update(
                {f"style_{name}": float(value) for name, value in zip(STYLE_NAMES, realized)}
            )
            metadata_rows.append(row)

        shard_path = run_dir / "rollouts" / f"{profile}__{actual_step:09d}.npz"
        np.savez_compressed(
            shard_path,
            episode_ids=np.asarray(batches["episode_ids"]),
            seeds=np.asarray(batches["seeds"], dtype=np.int64),
            lengths=np.asarray(batches["lengths"], dtype=np.int32),
            terminated=np.asarray(batches["terminated"], dtype=bool),
            truncated=np.asarray(batches["truncated"], dtype=bool),
            observations=np.asarray(batches["observations"], dtype=np.float32),
            actions=np.asarray(batches["actions"], dtype=np.float32),
            signals=np.asarray(batches["signals"], dtype=np.float32),
            reward_components=np.asarray(batches["components"], dtype=np.float32),
            valid=np.asarray(batches["valid"], dtype=bool),
            episode_features=np.asarray(batches["features"], dtype=np.float32),
            signal_names=np.asarray(SIGNAL_NAMES),
            component_names=np.asarray(STYLE_NAMES),
            feature_names=np.asarray(FEATURE_NAMES),
        )
        env.close()

    episodes = pd.DataFrame(metadata_rows)
    episodes.to_csv(run_dir / "tables" / "episodes.csv", index=False)
    quality_cfg = config.get("quality", {})
    fall_rate = float(episodes["terminated"].mean())
    mean_length_fraction = float(episodes["length"].mean() / horizon)
    profile_fall_rates = {
        str(profile): float(value)
        for profile, value in episodes.groupby("profile")["terminated"].mean().items()
    }
    profile_length_fractions = {
        str(profile): float(value / horizon)
        for profile, value in episodes.groupby("profile")["length"].mean().items()
    }
    feature_stds = {
        name: float(episodes[f"feature_{name}"].std(ddof=0)) for name in FEATURE_NAMES
    }
    quality_warnings = []
    max_fall_rate = float(quality_cfg.get("max_fall_rate", 0.95))
    min_mean_length_fraction = float(quality_cfg.get("min_mean_length_fraction", 0.0))
    if fall_rate > max_fall_rate:
        quality_warnings.append(
            f"Fall rate {fall_rate:.3f} exceeds maximum {max_fall_rate:.3f}."
        )
    for profile, value in profile_fall_rates.items():
        if value > max_fall_rate:
            quality_warnings.append(
                f"Profile {profile} fall rate {value:.3f} exceeds maximum {max_fall_rate:.3f}."
            )
    if mean_length_fraction < min_mean_length_fraction:
        quality_warnings.append(
            f"Mean length fraction {mean_length_fraction:.3f} is below {min_mean_length_fraction:.3f}."
        )
    for profile, value in profile_length_fractions.items():
        if value < min_mean_length_fraction:
            quality_warnings.append(
                f"Profile {profile} mean length fraction {value:.3f} is below {min_mean_length_fraction:.3f}."
            )
    low_variance = [name for name, value in feature_stds.items() if value < 1e-8]
    if low_variance:
        quality_warnings.append(f"Near-zero feature variance: {', '.join(low_variance)}")
    style_columns = [f"style_{name}" for name in STYLE_NAMES]
    profile_feature_means = episodes.groupby("profile", sort=False)[
        [f"feature_{name}" for name in FEATURE_NAMES]
    ].mean()
    profile_effects = []
    for expectation in quality_cfg.get("profile_effects", []):
        feature = str(expectation["feature"])
        higher = str(expectation["higher"])
        lower = str(expectation["lower"])
        minimum = float(expectation.get("min_difference", 0.0))
        column = f"feature_{feature}"
        difference = float(profile_feature_means.loc[higher, column] - profile_feature_means.loc[lower, column])
        passed = difference >= minimum
        profile_effects.append(
            {"feature": feature, "higher": higher, "lower": lower, "difference": difference, "minimum": minimum, "passed": passed}
        )
        if not passed:
            quality_warnings.append(
                f"Expected {higher} - {lower} on {feature} >= {minimum:.3f}, got {difference:.3f}."
            )

    summary = {
        "n_episodes": int(len(episodes)),
        "n_profiles": int(episodes["profile"].nunique()),
        "n_checkpoints": int(len(checkpoint_rows)),
        "fall_rate": fall_rate,
        "mean_length": float(episodes["length"].mean()),
        "mean_length_fraction": mean_length_fraction,
        "profile_fall_rates": profile_fall_rates,
        "profile_length_fractions": profile_length_fractions,
        "quality_pass": not quality_warnings,
        "quality_warnings": quality_warnings,
        "profile_effects": profile_effects,
        "feature_ranges": {
            name: {
                "min": float(episodes[f"feature_{name}"].min()),
                "max": float(episodes[f"feature_{name}"].max()),
                "std": float(episodes[f"feature_{name}"].std(ddof=0)),
            }
            for name in FEATURE_NAMES
        },
        "profile_style_means": {
            profile: {
                name: float(value)
                for name, value in zip(STYLE_NAMES, group[style_columns].mean().to_numpy())
            }
            for profile, group in episodes.groupby("profile", sort=False)
        },
    }
    write_json(run_dir / "reports" / "rollout_summary.json", summary)
    if quality_warnings and quality_cfg.get("fail_on_quality", False):
        raise RuntimeError("Rollout quality gate failed: " + " ".join(quality_warnings))
    return episodes
