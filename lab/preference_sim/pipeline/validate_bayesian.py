from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

from .common import write_json
from .reward_features import FEATURE_NAMES


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(left @ right / denominator) if denominator > 0 else float("nan")


def validate_bayesian_export(config: dict, run_dir: Path) -> dict:
    from reward.fully_bayesian.model import Population

    validation = config.get("validation", {})
    with np.load(run_dir / "exports" / "fully_bayesian_input.npz", allow_pickle=False) as data:
        features = data["episode_features_z"].astype(np.float64)
        Z = np.concatenate([np.ones((len(features), 1)), features], axis=1)
        labels = data["labels"]
        query_order = data["query_order"]
        user_ids = data["user_ids"].astype(str)
        user_split = data["user_split"].astype(str)
        theta_true = data["theta_true"].astype(np.float64)

    train_indices = np.flatnonzero(user_split == "train")
    train_limit = int(config["feedback"].get("train_labels_per_user", 40))
    Zs, ys, names = [], [], []
    for user_index in train_indices:
        mask = query_order[user_index] <= train_limit
        Zs.append(Z[mask])
        ys.append(labels[user_index, mask])
        names.append(user_ids[user_index])

    cfg = SimpleNamespace(
        n_samples=int(validation.get("n_samples", 160)),
        n_burnin=int(validation.get("n_burnin", 80)), thin=1,
        niw_kappa0=1.0, niw_nu0=None, niw_lambda0_scale=1.0,
        newuser_n_iters=int(validation.get("newuser_n_iters", 4)),
        eps_var=None, spike_slab=bool(validation.get("spike_slab", False)),
        spike_slab_unit="feature", spike_slab_a=1.0, spike_slab_b=1.0,
        seed=int(config.get("seed", 42)),
    )
    feature_names = ["bias", *FEATURE_NAMES]
    population = Population(cfg)
    population.fit(Zs, ys, feature_names, names, feature_names)

    train_metrics = []
    for user_index in train_indices:
        user_id = user_ids[user_index]
        holdout = query_order[user_index] > train_limit
        user = population.user(user_id)
        probability = user.predict(Z[holdout])[0]
        y_true = labels[user_index, holdout]
        row = {
            "user_id": user_id,
            "cosine_theta": _cosine(user.theta.mean(axis=0)[1:], theta_true[user_index]),
            "log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
        }
        if len(np.unique(y_true)) == 2:
            row["auroc"] = float(roc_auc_score(y_true, probability))
        train_metrics.append(row)

    test_metrics = []
    test_limit = int(config["feedback"].get("max_test_context", 40))
    for user_index in np.flatnonzero(user_split == "test"):
        context = query_order[user_index] <= test_limit
        holdout = query_order[user_index] > test_limit
        user = population.new_user(user_ids[user_index])
        user.fit(Z[context], labels[user_index, context])
        probability = user.predict(Z[holdout])[0]
        y_true = labels[user_index, holdout]
        row = {
            "user_id": user_ids[user_index],
            "cosine_theta": _cosine(user.theta.mean(axis=0)[1:], theta_true[user_index]),
            "log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
        }
        if len(np.unique(y_true)) == 2:
            row["auroc"] = float(roc_auc_score(y_true, probability))
        test_metrics.append(row)

    def mean_metric(rows: list[dict], key: str) -> float | None:
        values = [row[key] for row in rows if key in row and np.isfinite(row[key])]
        return float(np.mean(values)) if values else None

    report = {
        "n_train_users": len(train_metrics), "n_test_users": len(test_metrics),
        "train": {key: mean_metric(train_metrics, key) for key in ("auroc", "log_loss", "cosine_theta")},
        "test": {key: mean_metric(test_metrics, key) for key in ("auroc", "log_loss", "cosine_theta")},
        "train_users": train_metrics, "test_users": test_metrics,
    }
    write_json(run_dir / "reports" / "bayesian_validation.json", report)
    return report

