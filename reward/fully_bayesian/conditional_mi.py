"""Conditional-mutual-information ranking for sensors and individual features.

For a selectable unit A, the target quantity is

    I(Y; Z_A | Z_{-A}, U),

where U is user identity.  It is estimated as the cross-fitted log-predictive
gain of a full logistic model over a model with unit A removed.  The estimator
is model-based: with correctly specified conditional models, the expected log
score difference equals conditional mutual information.

Two units are supported:

* ``sensor``: all statistics derived from one sensor are removed together;
* ``feature``: one ``sensor__statistic`` column is removed.

This module does not compute BALD and does not use new-user uncertainty.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


_EPS = 1e-12
_VALID_UNITS = {"sensor", "feature"}


def _unit_columns(phi, unit):
    if unit not in _VALID_UNITS:
        raise ValueError(f"unit must be one of {sorted(_VALID_UNITS)}, got {unit!r}")
    feature_names = list(phi.feature_names)
    sensor_names = list(phi.groups)
    if len(feature_names) != len(sensor_names):
        raise ValueError("phi.feature_names and phi.groups must have the same length")

    bias = {j for j, (f, g) in enumerate(zip(feature_names, sensor_names))
            if f == "bias" or g == "bias"}
    selectable = [j for j in range(len(feature_names)) if j not in bias]
    labels = sensor_names if unit == "sensor" else feature_names
    units = OrderedDict()
    duplicate_count = defaultdict(int)
    for j in selectable:
        key = str(labels[j])
        if unit == "feature" and key in units:
            duplicate_count[key] += 1
            key = f"{key}#{duplicate_count[key] + 1}"
        units.setdefault(key, []).append(j)
    if not units:
        raise ValueError(f"No selectable {unit} units were found")
    return selectable, units


def _stack_data(Zs, ys):
    lengths = [len(y) for y in ys]
    Z = np.concatenate([np.asarray(z, dtype=np.float64) for z in Zs], axis=0)
    y = np.concatenate([np.asarray(v, dtype=np.int64) for v in ys])
    user = np.concatenate([np.full(n, u, dtype=np.int64) for u, n in enumerate(lengths)])
    return Z, y, user


def _within_user_folds(ys, requested_splits, seed):
    """Create the same stratified CV fold index within every user."""
    min_class = []
    for y in ys:
        counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=2)
        min_class.append(int(counts.min()))
    n_splits = min(int(requested_splits), min(min_class))
    if n_splits < 2:
        raise ValueError("Conditional MI needs at least two examples of each class per user")

    validation = [[] for _ in range(n_splits)]
    offset = 0
    for u, y in enumerate(ys):
        y = np.asarray(y, dtype=np.int64)
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + u)
        for fold, (_, test_idx) in enumerate(splitter.split(np.zeros(len(y)), y)):
            validation[fold].extend((offset + test_idx).tolist())
        offset += len(y)
    return [np.asarray(idx, dtype=np.int64) for idx in validation], n_splits


def _design_matrix(Z, user, n_users, cols, user_interactions=True):
    """Build a user-conditioned logistic design.

    With interactions, every user receives its own intercept and slopes.  With
    interactions disabled, users receive separate intercepts and share slopes.
    """
    cols = list(cols)
    one_hot = np.eye(n_users, dtype=np.float64)[user]
    if not cols:
        return one_hot
    features = Z[:, cols]
    if not user_interactions:
        return np.concatenate([one_hot, features], axis=1)

    blocks = [one_hot]
    for u in range(n_users):
        blocks.append(features * (user == u)[:, None])
    return np.concatenate(blocks, axis=1)


def _cross_fitted_lpd(X, y, validation_folds, *, C=1.0, max_iter=1000):
    probabilities = np.empty(len(y), dtype=np.float64)
    all_idx = np.arange(len(y))
    for test_idx in validation_folds:
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[test_idx] = False
        train_idx = all_idx[train_mask]
        if np.unique(y[train_idx]).size < 2:
            prior = (y[train_idx].sum() + 0.5) / (len(train_idx) + 1.0)
            probabilities[test_idx] = prior
            continue
        model = LogisticRegression(
            C=C,
            solver="lbfgs",
            fit_intercept=False,
            max_iter=max_iter,
        )
        model.fit(X[train_idx], y[train_idx])
        probabilities[test_idx] = model.predict_proba(X[test_idx])[:, 1]

    probabilities = np.clip(probabilities, _EPS, 1.0 - _EPS)
    return y * np.log(probabilities) + (1 - y) * np.log1p(-probabilities)


def _user_aggregate(values, user, n_users):
    per_user = np.array([np.mean(values[user == u]) for u in range(n_users)], dtype=np.float64)
    mean = float(per_user.mean())
    se = float(per_user.std(ddof=1) / np.sqrt(n_users)) if n_users > 1 else float("nan")
    return mean, se, per_user


def rank_units(
    Zs,
    ys,
    phi,
    unit,
    *,
    n_splits=5,
    C=1.0,
    max_iter=1000,
    user_interactions=True,
    seed=42,
    min_gain=0.0,
):
    """Rank units by cross-fitted estimate of I(Y; Z_A | Z_-A, U)."""
    all_cols, units = _unit_columns(phi, unit)
    Z, y, user = _stack_data(Zs, ys)
    n_users = len(Zs)
    folds, actual_splits = _within_user_folds(ys, n_splits, seed)

    full_X = _design_matrix(Z, user, n_users, all_cols, user_interactions)
    full_lpd = _cross_fitted_lpd(full_X, y, folds, C=C, max_iter=max_iter)
    full_mean, _, _ = _user_aggregate(full_lpd, user, n_users)

    rows = []
    for name, removed_cols in units.items():
        reduced_cols = [j for j in all_cols if j not in set(removed_cols)]
        reduced_X = _design_matrix(Z, user, n_users, reduced_cols, user_interactions)
        reduced_lpd = _cross_fitted_lpd(reduced_X, y, folds, C=C, max_iter=max_iter)
        gain, gain_se, per_user = _user_aggregate(full_lpd - reduced_lpd, user, n_users)
        reduced_mean, _, _ = _user_aggregate(reduced_lpd, user, n_users)
        rows.append({
            "unit": unit,
            "name": name,
            "removed_cols": list(removed_cols),
            "removed_features": [phi.feature_names[j] for j in removed_cols],
            "n_removed": len(removed_cols),
            "conditional_mi_nats": gain,
            "conditional_mi_bits": gain / np.log(2.0),
            "se_user_nats": gain_se,
            "per_user_gain_nats": per_user.tolist(),
            "full_cv_lpd_per_item": full_mean,
            "reduced_cv_lpd_per_item": reduced_mean,
            "important": bool(gain - gain_se > min_gain) if np.isfinite(gain_se) else bool(gain > min_gain),
        })

    rows.sort(key=lambda row: row["conditional_mi_nats"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return {
        "mode": "leave_one_out",
        "unit": unit,
        "n_users": n_users,
        "n_observations": len(y),
        "n_splits": actual_splits,
        "condition_on_user": True,
        "user_interactions": bool(user_interactions),
        "rows": rows,
    }


def forward_select(
    Zs,
    ys,
    phi,
    unit,
    *,
    n_splits=5,
    C=1.0,
    solver_max_iter=1000,
    user_interactions=True,
    seed=42,
    min_gain=0.0,
    max_steps=None,
    stop_at_nonpositive=True,
):
    """Greedy conditional-MI selection using cross-fitted log-score gains.

    At every step, choose A maximizing I(Y; Z_A | Z_S, U), where S is the
    set selected so far. This form chooses one representative from redundant
    statistics, unlike leave-one-out importance conditioned on all others.
    """
    _, units = _unit_columns(phi, unit)
    Z, y, user = _stack_data(Zs, ys)
    n_users = len(Zs)
    folds, actual_splits = _within_user_folds(ys, n_splits, seed)
    remaining = OrderedDict((name, list(cols)) for name, cols in units.items())
    selected_cols = []
    base_X = _design_matrix(Z, user, n_users, selected_cols, user_interactions)
    base_lpd = _cross_fitted_lpd(base_X, y, folds, C=C, max_iter=solver_max_iter)
    base_mean, _, _ = _user_aggregate(base_lpd, user, n_users)
    initial_mean = base_mean
    rows = []
    stopped_candidate = None

    while remaining and (max_steps is None or len(rows) < max_steps):
        candidates = []
        for name, added_cols in remaining.items():
            candidate_cols = selected_cols + added_cols
            design = _design_matrix(Z, user, n_users, candidate_cols, user_interactions)
            candidate_lpd = _cross_fitted_lpd(
                design, y, folds, C=C, max_iter=solver_max_iter
            )
            gain, gain_se, per_user = _user_aggregate(
                candidate_lpd - base_lpd, user, n_users
            )
            candidate_mean, _, _ = _user_aggregate(candidate_lpd, user, n_users)
            candidates.append((gain, name, added_cols, candidate_lpd,
                               candidate_mean, gain_se, per_user))

        best = max(candidates, key=lambda row: row[0])
        gain, name, added_cols, candidate_lpd, candidate_mean, gain_se, per_user = best
        if stop_at_nonpositive and gain <= min_gain:
            stopped_candidate = {
                "name": name,
                "conditional_mi_nats": float(gain),
                "se_user_nats": float(gain_se),
            }
            break

        selected_cols.extend(added_cols)
        del remaining[name]
        row = {
            "unit": unit,
            "step": len(rows) + 1,
            "name": name,
            "added_cols": list(added_cols),
            "added_features": [phi.feature_names[j] for j in added_cols],
            "selected_cols": list(selected_cols),
            "selected_features": [phi.feature_names[j] for j in selected_cols],
            "n_added": len(added_cols),
            "n_selected_features": len(selected_cols),
            "conditional_mi_nats": float(gain),
            "conditional_mi_bits": float(gain / np.log(2.0)),
            "se_user_nats": float(gain_se),
            "per_user_gain_nats": per_user.tolist(),
            "cv_lpd_per_item": float(candidate_mean),
            "important": bool(gain - gain_se > min_gain) if np.isfinite(gain_se) else bool(gain > min_gain),
        }
        rows.append(row)
        base_lpd = candidate_lpd
        base_mean = candidate_mean

    return {
        "mode": "forward",
        "unit": unit,
        "n_users": n_users,
        "n_observations": len(y),
        "n_splits": actual_splits,
        "condition_on_user": True,
        "user_interactions": bool(user_interactions),
        "baseline_cv_lpd_per_item": float(initial_mean),
        "stopped_candidate": stopped_candidate,
        "rows": rows,
    }


def select(cfg, phi, pop, train_data, unit="sensor"):
    names = pop.user_names
    Zs = [phi.transform(train_data[name][0]).astype(np.float64) for name in names]
    ys = [np.asarray(train_data[name][1], dtype=np.int64) for name in names]
    common = dict(
        n_splits=getattr(cfg, "cmi_n_splits", 5),
        C=getattr(cfg, "cmi_C", 1.0),
        user_interactions=getattr(cfg, "cmi_user_interactions", True),
        seed=getattr(cfg, "seed", 42),
        min_gain=getattr(cfg, "cmi_min_gain", 0.0),
    )
    mode = getattr(cfg, "cmi_mode", "forward")
    if mode == "forward":
        return forward_select(
            Zs,
            ys,
            phi,
            unit,
            solver_max_iter=getattr(cfg, "cmi_solver_max_iter", 1000),
            max_steps=getattr(cfg, "cmi_max_steps", None),
            stop_at_nonpositive=getattr(cfg, "cmi_stop_at_nonpositive", True),
            **common,
        )
    if mode == "leave_one_out":
        return rank_units(
            Zs,
            ys,
            phi,
            unit,
            max_iter=getattr(cfg, "cmi_solver_max_iter", 1000),
            **common,
        )
    raise ValueError("cmi_mode must be 'forward' or 'leave_one_out'")


def write_report(result, out_dir):
    unit = result["unit"]
    rows = result["rows"]
    if not rows:
        width = 18
    else:
        width = max(18, max(len(row["name"]) for row in rows) + 2)
    index_label = "step" if result["mode"] == "forward" else "rank"
    count_label = "n_total" if result["mode"] == "forward" else "n_feat"
    header = (f"{index_label:<5} {'unit':<{width}} {'CMI[nat]':>11} {'SE[user]':>11} "
              f"{'CMI[bit]':>11} {count_label:>8}  important")
    formula = "I(Y; Z_A | Z_S, U)" if result["mode"] == "forward" else "I(Y; Z_A | Z_-A, U)"
    lines = [
        f"Conditional-MI {unit} selection ({result['mode']}): {formula}",
        f"cross-fitted folds={result['n_splits']}, users={result['n_users']}, "
        f"observations={result['n_observations']}",
        "=" * len(header),
        "",
        header,
    ]
    for index, row in enumerate(rows, start=1):
        mark = "  *" if row["important"] else ""
        count = row.get("n_selected_features", row.get("n_removed", 0))
        lines.append(
            f"{index:<5} {row['name']:<{width}} {row['conditional_mi_nats']:>11.5g} "
            f"{row['se_user_nats']:>11.5g} {row['conditional_mi_bits']:>11.5g} "
            f"{count:>8}{mark}"
        )
    (out_dir / f"conditional_mi_{unit}.txt").write_text("\n".join(lines), encoding="utf-8")
    with open(out_dir / f"conditional_mi_{unit}.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)


def plot(result, save_path, top_k=25):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = result["rows"][:top_k]
    labels = [row["name"] for row in rows][::-1]
    values = [row["conditional_mi_nats"] for row in rows][::-1]
    errors = [row["se_user_nats"] for row in rows][::-1]
    colors = ["steelblue" if row["important"] else "lightgray" for row in rows][::-1]
    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.34 * len(rows))))
    y = np.arange(len(rows))
    ax.barh(y, values, xerr=errors, color=colors, alpha=0.9, capsize=3)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.tick_params(axis="x", labelsize=12.5)
    condition = "Z_S" if result["mode"] == "forward" else "Z_{-A}"
    ax.set_xlabel(rf"cross-fitted $I(Y;Z_A\mid {condition},U)$ [nats/item]", fontsize=12.5)
    ax.set_title(f"Conditional-MI {result['unit']} importance")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
