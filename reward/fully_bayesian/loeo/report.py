"""Aggregate fold results into tables (csv / markdown) and figures."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from . import metrics as M

MAIN_KEYS = ["mlpd", "brier", "auroc", "auprc", "ece", "cal_slope", "epi_mean", "ale_mean",
             "epi_ale_spearman", "correctness_auroc_epi", "aurc_epi", "eaurc_epi", "reliable_frac"]


def write_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=_default), encoding="utf-8")


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set,)):
        return sorted(o)
    return str(o)


def macro_table(folds, methods, budgets):
    """rows: (method, budget, metric, mean, se, n_folds)."""
    rows = []
    for m in methods:
        for t in budgets:
            keys = set()
            for f in folds.values():
                keys |= set(f.get(m, {}).get(str(t), {}).keys())
            for k in sorted(keys):
                vals = [f[m][str(t)][k] for f in folds.values()
                        if m in f and str(t) in f[m] and isinstance(f[m][str(t)].get(k), (int, float))]
                mean, se, n = M.macro(vals)
                rows.append(dict(method=m, t=t, metric=k, mean=mean, se=se, n=n))
    return rows


def paired_delta(folds, method, t, t_ref=0, key="lpd"):
    """Per-evaluator paired ELPD difference between budget t and t_ref with SE (Vehtari eq. 24)."""
    out = {}
    for name, f in folds.items():
        L = f.get("lpd", {}).get(method, {})
        if str(t) in L and str(t_ref) in L:
            d = np.asarray(L[str(t)]) - np.asarray(L[str(t_ref)])
            s, se = M.sum_se(d)
            out[name] = dict(delta_elpd=s, se=se, n=int(d.size))
    return out


def write_csv(rows, path):
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def markdown_main(rows, budgets, methods, keys=("mlpd", "brier", "auroc", "ece", "reliable_frac")):
    lines = []
    for t in budgets:
        lines.append(f"\n### Budget t = {t}\n")
        lines.append("| method | " + " | ".join(keys) + " |")
        lines.append("|---|" + "---|" * len(keys))
        for m in methods:
            cells = []
            for k in keys:
                r = next((r for r in rows if r["method"] == m and r["t"] == t and r["metric"] == k), None)
                cells.append("--" if r is None or not np.isfinite(r["mean"]) else
                             (f"{r['mean']:.4f} ± {r['se']:.4f}" if np.isfinite(r["se"]) else f"{r['mean']:.4f}"))
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_curves(rows, methods, budgets, out_path, metric="mlpd", ylabel=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for m in methods:
        xs, ys, es = [], [], []
        for t in budgets:
            r = next((r for r in rows if r["method"] == m and r["t"] == t and r["metric"] == metric), None)
            if r and np.isfinite(r["mean"]):
                xs.append(t); ys.append(r["mean"]); es.append(0 if not np.isfinite(r["se"]) else r["se"])
        if xs:
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=m)
    ax.set_xlabel("context size t"); ax.set_ylabel(ylabel or metric); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140); plt.close(fig)


def plot_uncertainty(folds, budgets, out_path, method="proposed"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name, f in folds.items():
        r = f.get(method, {})
        ts = [t for t in budgets if str(t) in r]
        if not ts:
            continue
        ax.plot(ts, [r[str(t)]["epi_mean"] for t in ts], "-o", ms=3, label=f"{name} epi")
        ax.plot(ts, [r[str(t)]["ale_mean"] for t in ts], "--", alpha=0.6)
    ax.set_xlabel("context size t"); ax.set_ylabel("mean uncertainty (solid epi, dashed ale)")
    ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140); plt.close(fig)


def plot_size_curve(sizes, mean, se, chosen, full_size, out_path, xlabel="number of features"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.errorbar(sizes, [mean[k] for k in sizes], yerr=[0 if not np.isfinite(se.get(k, np.nan)) else se[k] for k in sizes],
                marker="o", capsize=3)
    ax.axhline(mean[full_size], color="gray", ls="--", label="full bank")
    ax.axvline(chosen, color="crimson", ls=":", label=f"selected k={chosen}")
    ax.set_xlabel(xlabel); ax.set_ylabel("held-out MLPD (macro)"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140); plt.close(fig)
