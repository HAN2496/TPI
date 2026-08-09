"""Projection-predictive selection at sensor and individual-feature levels.

The reference model is the fitted hierarchical ``Population`` using all
features.  For each retained posterior draw, a restricted logistic model is
projected onto a candidate column set by minimizing

    KL(Bernoulli(p_reference) || Bernoulli(q_subset)).

The projection is deterministic; it does not rerun Gibbs sampling.  Sensor
selection treats every statistic extracted from one sensor as a group, while
feature selection treats one ``sensor__statistic`` column as one unit.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
import json

import numpy as np
from scipy.special import expit


_EPS = 1e-12
_VALID_UNITS = {"sensor", "feature"}


def _clip_prob(p):
    return np.clip(np.asarray(p, dtype=np.float64), _EPS, 1.0 - _EPS)


def _cross_entropy(X, p, coef, ridge_diag):
    q = _clip_prob(expit(X @ coef))
    ce = -np.sum(p * np.log(q) + (1.0 - p) * np.log1p(-q))
    penalty = 0.5 * np.dot(ridge_diag * coef, coef)
    return float(ce + penalty)


def project(X, p_target, n_iter=50, ridge=1e-6, tol=1e-8):
    """Project soft Bernoulli targets onto ``sigmoid(X @ coef)``.

    Newton/IRLS is used with backtracking. Constant columns, including the
    bias, are not ridge-penalized.
    """
    X = np.asarray(X, dtype=np.float64)
    p = _clip_prob(p_target)
    if X.ndim != 2 or X.shape[0] != p.shape[0] or X.shape[1] == 0:
        raise ValueError("X must have shape [n_observations, n_selected_features]")

    coef = np.zeros(X.shape[1], dtype=np.float64)
    varying = np.ptp(X, axis=0) > 1e-12
    ridge_diag = ridge * varying.astype(np.float64)

    for _ in range(n_iter):
        q = _clip_prob(expit(X @ coef))
        weight = np.maximum(q * (1.0 - q), 1e-9)
        grad = X.T @ (q - p) + ridge_diag * coef
        hessian = (X.T * weight) @ X + np.diag(ridge_diag + 1e-10)
        step = np.linalg.solve(hessian, grad)
        if np.linalg.norm(step, ord=np.inf) <= tol:
            break

        current = _cross_entropy(X, p, coef, ridge_diag)
        directional = float(np.dot(grad, step))
        scale = 1.0
        for _ in range(20):
            candidate = coef - scale * step
            if _cross_entropy(X, p, candidate, ridge_diag) <= current - 1e-4 * scale * directional:
                coef = candidate
                break
            scale *= 0.5
        else:
            coef = coef - scale * step
    return coef


def bernoulli_kl(p, q):
    """Pointwise KL(Bernoulli(p) || Bernoulli(q))."""
    p = _clip_prob(p)
    q = _clip_prob(q)
    return p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))


def _unit_columns(phi, unit):
    """Return bias columns and ordered selectable column groups."""
    if unit not in _VALID_UNITS:
        raise ValueError(f"unit must be one of {sorted(_VALID_UNITS)}, got {unit!r}")

    feature_names = list(phi.feature_names)
    sensor_names = list(phi.groups)
    if len(feature_names) != len(sensor_names):
        raise ValueError("phi.feature_names and phi.groups must have the same length")

    bias = [j for j, (f, g) in enumerate(zip(feature_names, sensor_names))
            if f == "bias" or g == "bias"]
    raw_labels = sensor_names if unit == "sensor" else feature_names
    columns = OrderedDict()
    duplicate_count = defaultdict(int)
    for j, label in enumerate(raw_labels):
        if j in bias:
            continue
        key = str(label)
        if unit == "feature" and key in columns:
            duplicate_count[key] += 1
            key = f"{key}#{duplicate_count[key] + 1}"
        columns.setdefault(key, []).append(j)
    if not columns:
        raise ValueError(f"No selectable {unit} units were found")
    return bias, columns


def _reference_predictions(Zs, theta_samples, draws):
    references = []
    for m in draws:
        for u, Z in enumerate(Zs):
            references.append((Z, _clip_prob(expit(Z @ theta_samples[m, u]))))
    return references


def projection_kl(references, cols, *, n_iter=50, ridge=1e-6):
    """Mean Bernoulli KL for a selected set of columns.

    cols가 비면(bias 없이 feature 0개) intercept-only baseline으로 처리 —
    soft target의 CE를 최소화하는 상수는 mean(p)이라, bias 유무와 무관하게 동일.
    """
    total = 0.0
    count = 0
    for Z, p in references:
        if cols:
            X = Z[:, cols]
            coef = project(X, p, n_iter=n_iter, ridge=ridge)
            q = expit(X @ coef)
        else:
            q = np.full_like(p, float(np.mean(p)))
        total += float(np.sum(bernoulli_kl(p, q)))
        count += len(p)
    return total / max(count, 1)


def forward_search(
    references,
    bias_cols,
    unit_cols,
    *,
    target=0.95,
    n_iter=50,
    ridge=1e-6,
    stop_at_target=True,
):
    """Greedily add the unit yielding the lowest projected KL."""
    selected_cols = list(bias_cols)
    remaining = OrderedDict((name, list(cols)) for name, cols in unit_cols.items())
    baseline = projection_kl(references, selected_cols, n_iter=n_iter, ridge=ridge)
    order = [{
        "added": "(bias)",
        "cols": list(selected_cols),
        "n_features": len(selected_cols),
        "projection_kl": baseline,
        "captured": 0.0,
        "pick": False,
    }]

    while remaining:
        candidates = []
        for name, added_cols in remaining.items():
            cols = selected_cols + added_cols
            loss = projection_kl(references, cols, n_iter=n_iter, ridge=ridge)
            candidates.append((loss, name, added_cols))
        loss, name, added_cols = min(candidates, key=lambda row: row[0])
        selected_cols.extend(added_cols)
        del remaining[name]
        captured = 1.0 if baseline <= _EPS else 1.0 - loss / baseline
        order.append({
            "added": name,
            "cols": list(selected_cols),
            "n_features": len(selected_cols),
            "projection_kl": loss,
            "captured": float(np.clip(captured, -np.inf, 1.0)),
            "pick": False,
        })
        if stop_at_target and captured >= target:
            break

    pick = next((row for row in order[1:] if row["captured"] >= target), order[-1])
    pick["pick"] = True
    return order


def select(cfg, phi, pop, train_data, unit="sensor", n_draws=None):
    """Run projection search for ``unit='sensor'`` or ``unit='feature'``."""
    names = pop.user_names
    Zs = [phi.transform(train_data[name][0]).astype(np.float64) for name in names]
    n_draws = n_draws or getattr(cfg, "projpred_n_draws", 60)
    draw_idx = np.linspace(0, pop.n_samples - 1, min(n_draws, pop.n_samples)).astype(int)
    references = _reference_predictions(Zs, pop.theta_samples, draw_idx)
    bias, units = _unit_columns(phi, unit)
    order = forward_search(
        references,
        bias,
        units,
        target=getattr(cfg, "projpred_target", 0.95),
        n_iter=getattr(cfg, "projpred_max_iter", 50),
        ridge=getattr(cfg, "projpred_ridge", 1e-6),
        stop_at_target=getattr(cfg, "projpred_stop_at_target", True),
    )
    for row in order:
        row["unit"] = unit
        row["selected_features"] = [phi.feature_names[j] for j in row["cols"]]
    return order


def write_report(order, out_dir):
    unit = order[0]["unit"]
    width = max(18, max(len(row["added"]) for row in order) + 2)
    header = (f"{'step':<5} {'+added':<{width}} {'KL':>12} "
              f"{'captured':>10} {'n_feat':>8}  pick")
    lines = [
        f"Projection-predictive {unit} selection",
        "=" * len(header),
        "",
        header,
    ]
    for step, row in enumerate(order):
        mark = "  <== select" if row["pick"] else ""
        lines.append(
            f"{step:<5} {row['added']:<{width}} {row['projection_kl']:>12.6g} "
            f"{row['captured']:>10.3%} {row['n_features']:>8}{mark}"
        )
    (out_dir / f"projpred_{unit}.txt").write_text("\n".join(lines), encoding="utf-8")
    with open(out_dir / f"projpred_{unit}.json", "w", encoding="utf-8") as handle:
        json.dump(order, handle, ensure_ascii=False, indent=2)


def plot(order, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = np.arange(len(order))
    losses = [row["projection_kl"] for row in order]
    labels = [row["added"] for row in order]
    fig, ax = plt.subplots(figsize=(8, max(4, len(order) * 0.4)))
    ax.plot(losses, y, marker="o", color="steelblue")
    for i, row in enumerate(order):
        if row["pick"]:
            ax.scatter(row["projection_kl"], i, s=160, facecolors="none",
                       edgecolors="crimson", linewidths=2, zorder=5, label="selected")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.tick_params(axis="x", labelsize=12.5)
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.set_xlabel(r"mean $D_{KL}(p_{full}\Vert p_S)$", fontsize=12.5)
    ax.set_ylabel(f"added {order[0]['unit']} (forward-selection order)", fontsize=12.5)
    ax.set_title(f"Projection-predictive {order[0]['unit']} selection")
    ax.grid(alpha=0.3)
    if any(row["pick"] for row in order):
        ax.legend()
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main(cfg=None):
    from loader import Dataset
    from core import Run
    from .features import build_feature_pipeline
    from .model import Population

    if cfg is None:
        from run_fully_bayesian import Config
        cfg = Config(n_samples=300, n_burnin=300)
    run = Run("projpred", cfg)
    dataset = Dataset("datasets")
    train_data = {name: cfg.view(dataset[name]) for name in cfg.train}
    phi = build_feature_pipeline(cfg)
    Xs = [train_data[name][0] for name in cfg.train]
    ys = [train_data[name][1] for name in cfg.train]
    phi.fit(Xs, ys)
    pop = Population(cfg)
    pop.fit([phi.transform(X).astype(np.float64) for X in Xs], ys,
            phi.feature_names, list(cfg.train), phi.groups)

    for unit in getattr(cfg, "selection_units", ("sensor", "feature")):
        order = select(cfg, phi, pop, train_data, unit=unit)
        write_report(order, run.dir)
        plot(order, run.plots / f"projpred_{unit}.png")
    run.finish()


if __name__ == "__main__":
    main()
