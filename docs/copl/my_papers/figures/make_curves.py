"""Fig. 2 of the CoPL paper: held-out MLPD and AUROC against the context budget t.

Reads report/main_macro.csv of the final run (docs/copl/claude_notes/03, E9).
    python docs/copl/my_papers/figures/make_curves.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = Path("outputs/copl_loeo_final2/20261007_233243")
OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})

rows = list(csv.DictReader(open(RUN / "report" / "main_macro.csv", encoding="utf-8")))
budgets = ["0", "5", "10", "20"]
xs = [0, 5, 10, 20]
METHODS = [("copl", "CoPL-LP (proposed)", "#1f5f8b", "o", "-"),
           ("pooled", "Pooled-CNN", "#8a4b08", "s", "--"),
           ("knn_vote", "kNN vote", "#2b6a3f", "^", ":"),
           ("indep", "Indep-CNN", "#8b2b2b", "d", "-.")]


def series(method, metric):
    out = []
    for t in budgets:
        r = [r for r in rows if r["method"] == method and r["t"] == t and r["metric"] == metric]
        out.append((float(r[0]["mean"]), float(r[0]["se"])) if r else (None, None))
    return out


fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.5))
for ax, metric, ylabel in zip(axes, ["mlpd", "auroc"], ["held-out MLPD (nat)", "held-out AUROC"]):
    for m, label, color, marker, ls in METHODS:
        s = series(m, metric)
        x = [xi for xi, (v, _) in zip(xs, s) if v is not None]
        y = [v for v, _ in s if v is not None]
        e = [se for v, se in s if v is not None]
        ax.errorbar(x, y, yerr=e, label=label, color=color, marker=marker, ls=ls, ms=4, lw=1.2, capsize=2)
    ax.set_xlabel("context labels $t$")
    ax.set_ylabel(ylabel)
    ax.set_xticks(xs)
    ax.grid(alpha=0.3)
axes[0].set_ylim(-1.25, -0.5)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(0.5, 1.0))
axes[0].text(0.02, 0.97, "(a)", transform=axes[0].transAxes, va="top", weight="bold")
axes[1].text(0.02, 0.97, "(b)", transform=axes[1].transAxes, va="top", weight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.92))
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_curves.{ext}", dpi=300 if ext == "png" else None)
print("written", OUT / "fig_curves.{pdf,png}")
