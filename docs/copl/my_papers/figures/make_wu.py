"""Fig. 3 of the CoPL paper: adaptation weights w_u at the full context, vote rule vs. likelihood rule.

Rows: held-out evaluators E1..E10 (pseudonyms of the paper); columns: population users ordered by
the number of labeled episodes (eligible evaluators by their counts, then the seven low-label users).
    python docs/copl/my_papers/figures/make_wu.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
RUNS = [("Vote (CoPL)", Path("outputs/copl_loeo_final/20261005_162500")),
        ("Likelihood posterior (proposed)", Path("outputs/copl_loeo_final2/20261007_233243"))]
# evaluator pseudonyms and episode counts (docs/copl/claude_notes/04, Table I)
PSEUDO = {"이지환": ("E1", 1056), "박재일": ("E2", 591), "한규택": ("E3", 369), "조현석": ("E4", 279), "강신길": ("E5", 205),
          "김태근": ("E6", 35), "김재호": ("E7", 34), "김진명": ("E8", 30), "이강근": ("E9", 18), "신민철": ("E10", 10)}
HELD = list(PSEUDO)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5})


def final_key(d):
    return max(d, key=lambda s: int(s))


def population_order(users):
    known = sorted([u for u in users if u in PSEUDO], key=lambda u: -PSEUDO[u][1])
    other = sorted(u for u in users if u not in PSEUDO)             # all have fewer than 10 labels
    return known + other


fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0), sharey=True)
for ax, (title, run) in zip(axes, RUNS):
    folds = {n: json.loads((run / "folds" / f"{n}.json").read_text(encoding="utf-8")) for n in HELD}
    # a common column order: every population user that appears in any fold, by size
    all_users = sorted({u for f in folds.values() for u in f["population_users"]})
    cols = population_order(all_users)
    M = np.full((len(HELD), len(cols)), np.nan)
    for i, n in enumerate(HELD):
        f = folds[n]
        w = np.asarray(f["w_u"][final_key(f["w_u"])])
        for u, wu in zip(f["population_users"], w):
            M[i, cols.index(u)] = wu
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([PSEUDO[u][0] if u in PSEUDO else "P%d" % (cols.index(u) - 9) for u in cols], rotation=90)
    ax.set_yticks(range(len(HELD)))
    ax.set_yticklabels([PSEUDO[n][0] for n in HELD])
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("population occupant (E1–E10 by label count; P1–P8 non-eligible)")
    for i in range(len(HELD)):                                       # the held-out user is not in its own population
        j = cols.index(HELD[i]) if HELD[i] in cols else None
        if j is not None:
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fc="#dddddd", ec="none", hatch="///", lw=0))
axes[0].set_ylabel("held-out evaluator")
fig.colorbar(im, ax=axes, label="$w_u$ at full context", fraction=0.03, pad=0.02)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_wu.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
print("written", OUT / "fig_wu.{pdf,png}")
