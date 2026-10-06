"""Aggregate CoPL fold results. Reuses the T-IV aggregation code so both papers' tables match."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from reward.fully_bayesian.loeo import metrics as M
from reward.fully_bayesian.loeo import report as R

METHODS = ("copl", "pooled", "indep", "knn_vote")
UNC_KEYS = ("epi_mean", "ale_mean", "epi_ale_spearman", "correctness_auroc_epi", "aurc_epi", "eaurc_epi")


def load_folds(run_dir, suffix=""):
    out = {}
    for p in sorted(Path(run_dir, "folds").glob(f"*{suffix}.json")):
        stem = p.stem
        if suffix:
            stem = stem[: -len(suffix)]
        elif any(stem.endswith(s) for s in ("_encoders", "_ablate", "_sweep", "_channels", "_adapt", "_tune")):
            continue
        out[stem] = json.loads(p.read_text(encoding="utf-8"))
    return out


def report_main(folds, cfg, rep):
    R.normalize_final(folds, cfg.budgets, methods=METHODS)
    budgets = R.budget_keys(folds, "copl")
    methods = [m for m in METHODS if any(m in f and f[m] for f in folds.values())]
    rows = R.macro_table(folds, methods, budgets)
    R.write_csv(rows, rep / "main_macro.csv")
    md = ["# CoPL LOEO main results", f"folds: {len(folds)} evaluators: {', '.join(folds)}",
          f"encoder: {next(iter(folds.values())).get('encoder')}  graph: {next(iter(folds.values())).get('graph_rule')} "
          f"k={next(iter(folds.values())).get('knn_k')}",
          R.markdown_main(rows, budgets, methods)]
    if any("epi_mean" in f["copl"].get(str(budgets[0]), {}) for f in folds.values()):
        md += ["\n## Uncertainty (copl)\n", R.markdown_main(rows, budgets, ["copl"], keys=UNC_KEYS)]
    ts_pos = [t for t in budgets if t != 0]
    md.append("\n## Paired ELPD change from cold start (copl)\n")
    md.append("| evaluator | " + " | ".join(f"t={t}" for t in ts_pos) + " |")
    md.append("|---|" + "---|" * len(ts_pos))
    deltas = {t: R.paired_delta(folds, "copl", t) for t in ts_pos}
    for name in folds:
        cells = []
        for t in ts_pos:
            dl = deltas[t].get(name)
            cells.append("--" if dl is None else f"{dl['delta_elpd']:+.2f} ± {dl['se']:.2f}")
        md.append(f"| {name} | " + " | ".join(cells) + " |")
    md.append("\n## Paired ELPD, copl minus pooled (same holdout)\n")
    md.append("| evaluator | " + " | ".join(f"t={t}" for t in budgets) + " |")
    md.append("|---|" + "---|" * len(budgets))
    for name, f in folds.items():
        cells = []
        for t in budgets:
            a = f["lpd"].get("copl", {}).get(str(t)); b = f["lpd"].get("pooled", {}).get(str(t))
            if a is None or b is None or len(a) != len(b):
                cells.append("--"); continue
            s, se = M.sum_se(np.asarray(a) - np.asarray(b))
            cells.append(f"{s:+.2f} ± {se:.2f}")
        md.append(f"| {name} | " + " | ".join(cells) + " |")
    md.append("\n## Adaptation weights: entropy of w_u (nats; log U = full spread) and population users\n")
    md.append("| evaluator | " + " | ".join(f"t={t}" for t in budgets) + " | log U |")
    md.append("|---|" + "---|" * (len(budgets) + 1))
    for name, f in folds.items():
        U = len(f.get("population_users", []))
        cells = [f"{f['copl'][str(t)]['w_entropy']:.2f}" if str(t) in f["copl"] and "w_entropy" in f["copl"][str(t)] else "--"
                 for t in budgets]
        md.append(f"| {name} | " + " | ".join(cells) + f" | {np.log(max(U, 1)):.2f} |")
    md.append("\n## Graph statistics (seed-first)\n")
    md.append("| evaluator | items | edges | rho_cross | mean deg | components | isolated users |")
    md.append("|---|---|---|---|---|---|---|")
    for name, f in folds.items():
        g = next(iter(f.get("graph", {}).values()), {})
        if g:
            md.append(f"| {name} | {g['n_items']} | {g['n_edges']} | {g['rho_cross']:.3f} | {g['mean_degree']:.1f} | "
                      f"{g['n_components']} | {g['n_isolated_users']} |")
    (rep / "main.md").write_text("\n".join(md), encoding="utf-8")
    R.plot_curves(rows, methods, budgets, rep / "curve_mlpd.png", metric="mlpd", ylabel="held-out MLPD (macro)")
    R.plot_curves(rows, methods, budgets, rep / "curve_auroc.png", metric="auroc", ylabel="held-out AUROC (macro)")
    # w_u heatmap at the final budget
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = list(folds)
        users = sorted({u for f in folds.values() for u in f.get("population_users", [])})
        Wm = np.full((len(names), len(users)), np.nan)
        for i, (n, f) in enumerate(folds.items()):
            key = R.FINAL if R.FINAL in f["w_u"] else str(max(int(k) for k in f["w_u"] if k != R.FINAL))
            w = f["w_u"].get(key)
            if w is None:
                continue
            for j, u in enumerate(f["population_users"]):
                Wm[i, users.index(u)] = w[j]
        fig, ax = plt.subplots(figsize=(0.45 * len(users) + 2, 0.4 * len(names) + 1.5))
        im = ax.imshow(Wm, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(users))); ax.set_xticklabels(users, rotation=90, fontsize=7)
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("population user"); ax.set_ylabel("held-out evaluator"); fig.colorbar(im, ax=ax, label="w_u at full context")
        fig.tight_layout(); fig.savefig(rep / "wu_heatmap.png", dpi=140); plt.close(fig)
    except Exception as e:                                       # plotting must never break the report
        print(f"[report] w_u heatmap skipped: {e}")
    return rows


def normalize_variants(folds, key, regular_budgets):
    """Map each variant's full-context budget (t == n_ctx) onto the shared key 'final' in place."""
    for f in folds.values():
        for lab, v in f.get(key, {}).items():
            n_ctx = (v.get("info") or {}).get("n_ctx")
            if n_ctx is None:
                continue
            k = str(n_ctx)
            for d in (v.get("copl") or {}, v.get("lpd") or {}):
                if k in d:
                    d[R.FINAL] = d[k]
                    if int(n_ctx) not in {int(t) for t in regular_budgets}:
                        del d[k]


def report_table(folds, key, rep, fname, label_key="label", metric="mlpd"):
    """Generic macro table over variants stored as {label: {"copl": {t: metrics}, ...}}."""
    rows = []
    labels = sorted({lab for f in folds.values() for lab in f.get(key, {})})
    for lab in labels:
        ts = sorted({t for f in folds.values() for t in f.get(key, {}).get(lab, {}).get("copl", {})},
                    key=lambda s: (s == R.FINAL, int(s) if s != R.FINAL else 0))
        for t in ts:
            vals = [f[key][lab]["copl"][t][metric] for f in folds.values()
                    if lab in f.get(key, {}) and t in f[key][lab]["copl"]]
            mean, se, n = M.macro(vals)
            extra = {}
            g = next((f[key][lab].get("graph") for f in folds.values() if lab in f.get(key, {})), None)
            if g:
                extra = dict(rho_cross=g.get("rho_cross"), n_edges=g.get("n_edges"), n_components=g.get("n_components"))
            rows.append(dict(**{label_key: lab}, t=t, metric=metric, mean=mean, se=se, n=n, **extra))
    R.write_csv(rows, rep / fname)
    return rows


def md_variant_table(rows, label_key, title):
    labels = []
    for r in rows:
        if r[label_key] not in labels:
            labels.append(r[label_key])
    ts = []
    for r in rows:
        if r["t"] not in ts:
            ts.append(r["t"])
    md = [f"\n## {title}\n", f"| {label_key} | " + " | ".join(f"t={t}" for t in ts) + " | rho_cross |", "|---|" + "---|" * (len(ts) + 1)]
    for lab in labels:
        cells = []
        for t in ts:
            r = next((r for r in rows if r[label_key] == lab and r["t"] == t), None)
            cells.append("--" if r is None or not np.isfinite(r["mean"]) else f"{r['mean']:.4f} ± {r['se']:.4f}")
        rc = next((r.get("rho_cross") for r in rows if r[label_key] == lab and r.get("rho_cross") is not None), None)
        md.append(f"| {lab} | " + " | ".join(cells) + f" | {'--' if rc is None else f'{rc:.3f}'} |")
    return "\n".join(md)


def report_encoders(folds, rep):
    """Intrinsic metrics per (encoder, channel set, k): macro over folds."""
    rows = []
    combos = sorted({(enc, cs) for f in folds.values() for enc in f["encoders"]
                     for cs, v in f["encoders"][enc].items() if "error" not in v})
    for enc, cs in combos:
        ks = sorted({int(k) for f in folds.values() for k in f["encoders"].get(enc, {}).get(cs, {}).get("pop", {})})
        for k in ks:
            row = dict(encoder=enc, channel_set=cs, k=k)
            for side, mets in (("pop", ("agree_cross", "agree_cross_excess", "rho_cross", "vote_auroc", "vote_mlpd")),
                               ("held", ("heldout_vote_auroc", "heldout_vote_mlpd"))):
                for met in mets:
                    vals = [f["encoders"][enc][cs][side][str(k)][met] for f in folds.values()
                            if str(k) in f["encoders"].get(enc, {}).get(cs, {}).get(side, {})]   # skips failed entries
                    row[met], row[met + "_se"], _ = M.macro(vals)
            rows.append(row)
    R.write_csv(rows, rep / "encoders.csv")
    md = ["# Encoder study (intrinsic metrics, macro over folds)"]
    for k in sorted({r["k"] for r in rows}):
        md.append(f"\n### k = {k}\n")
        md.append("| encoder | channels | agree_cross (excess) | rho_cross | vote AUROC (pop) | vote MLPD (pop) | held-out vote AUROC | held-out vote MLPD |")
        md.append("|---|---|---|---|---|---|---|---|")
        for r in sorted([r for r in rows if r["k"] == k], key=lambda r: -(r.get("heldout_vote_mlpd") or -9)):
            md.append(f"| {r['encoder']} | {r['channel_set']} | {r['agree_cross']:.3f} ({r['agree_cross_excess']:+.3f}) | "
                      f"{r['rho_cross']:.3f} | {r['vote_auroc']:.3f} | {r['vote_mlpd']:.3f} | "
                      f"{r['heldout_vote_auroc']:.3f} ± {r['heldout_vote_auroc_se']:.3f} | {r['heldout_vote_mlpd']:.3f} ± {r['heldout_vote_mlpd_se']:.3f} |")
    (rep / "encoders.md").write_text("\n".join(md), encoding="utf-8")
    return rows
