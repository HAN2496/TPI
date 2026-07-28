import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from .plots import STYLE


def compute_sequential_aurocs(probs, labels, context_sizes):
    """누적 AUROC: predictions[0..k] 전체로 계산, 양 클래스 미등장 시 skip."""
    seq_x, seq_y = [], []
    for k in range(len(probs)):
        if len(set(labels[:k + 1])) == 2:
            seq_y.append(roc_auc_score(labels[:k + 1], probs[:k + 1]))
            seq_x.append(context_sizes[k])
    return seq_x, seq_y


def pointwise_lpd(y, p1):
    """Per-point log posterior-predictive density (Vehtari 2017 eq.3). p1: P(y=1)."""
    p_true = np.where(y == 1, p1, 1.0 - p1)
    return np.log(np.clip(p_true, 1e-12, 1.0))


def sum_se(elpd_i):
    """합과 se = sqrt(n * Var) (Vehtari 2017 eq 23). 차이 벡터를 넣으면 paired 비교 (eq 24)."""
    return float(elpd_i.sum()), float(np.sqrt(elpd_i.size * elpd_i.var(ddof=1)))


def auroc_trust_interval(y, probs_M, seed, K=600, ci_lo_pct=2.5, ci_hi_pct=97.5,
                         width_max=0.15, lo_min=0.5):
    """Joint posterior x bootstrap 95% CI of AUROC. Trust iff CI_lo > 0.5 and width < width_max."""
    n = len(y)
    n_pos = int(y.sum()); n_neg = n - n_pos
    base = dict(n=n, n_pos=n_pos, n_neg=n_neg)
    nan = {**base, "mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
           "width": float("nan"), "trustworthy": False}
    if n_pos == 0 or n_neg == 0:
        return {**nan, "reason": "single-class holdout"}
    M, N = probs_M.shape
    rng = np.random.default_rng(seed)
    aurocs = []
    for _ in range(K):
        idx = rng.integers(0, N, size=N)
        if len(np.unique(y[idx])) < 2:
            continue
        aurocs.append(roc_auc_score(y[idx], probs_M[rng.integers(0, M)][idx]))
    if not aurocs:
        return {**nan, "reason": "no valid bootstrap resamples"}
    a = np.asarray(aurocs)
    mean, lo, hi = float(a.mean()), float(np.percentile(a, ci_lo_pct)), float(np.percentile(a, ci_hi_pct))
    width = hi - lo
    trustworthy = (lo > lo_min) and (width < width_max)
    reason = ""
    if not trustworthy:
        reason = f"CI_lo {lo:.3f} <= {lo_min:.2f} (random)" if lo <= lo_min else f"CI_width {width:.3f} >= {width_max:.2f}"
    return {**base, "mean": mean, "ci_lo": lo, "ci_hi": hi, "width": width,
            "trustworthy": bool(trustworthy), "reason": reason}


def trust_to_metric(t):
    return {"auroc_ci_lo": t["ci_lo"], "auroc_ci_hi": t["ci_hi"], "auroc_ci_width": t["width"],
            "trustworthy": t["trustworthy"], "n_holdout": t["n"], "n_pos": t["n_pos"], "n_neg": t["n_neg"]}


def save_metrics_txt(metrics, path):
    """{section/name: {metric: value}} -> 사람이 읽는 txt. 키에 있는 건 전부 찍는다."""
    lines = []
    for key, m in metrics.items():
        lines.append(key)
        for k, v in m.items():
            lines.append(f"  {k:<20}: " + (f"{v:.4f}" if isinstance(v, float) else f"{v}"))
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def evaluate_predictions(y_trues, probs_list, plots_dir, names, save_name="metrics", title=""):
    """driver별 ROC/scatter/hist를 (n, 3) subplot으로 한 장에. metric dict 리스트 반환."""
    n = len(names)
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    S = STYLE["diag"]
    fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n))
    axes = np.array(axes).reshape(n, 3)
    if title:
        fig.suptitle(title, fontsize=S["title_fs"])

    metrics_list = []
    for r, (y_true, probs, name) in enumerate(zip(y_trues, probs_list, names)):
        y_true = np.asarray(y_true, dtype=np.int64)
        probs = np.asarray(probs, dtype=np.float64).ravel()
        has_both = len(np.unique(y_true)) > 1
        auroc = roc_auc_score(y_true, probs) if has_both else float("nan")
        auprc = average_precision_score(y_true, probs) if has_both else float("nan")
        brier = float(np.mean((probs - y_true) ** 2))

        ax = axes[r, 0]
        if has_both:
            fpr, tpr, _ = roc_curve(y_true, probs)
            ax.plot(fpr, tpr, color="darkorange", lw=S["lw"], label=f"AUROC={auroc:.4f}")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
        ax.set_xlabel("FPR", fontsize=S["label_fs"])
        ax.set_ylabel(f"{name}\nTPR", fontsize=S["label_fs"])
        ax.set_title("ROC Curve" if r == 0 else "", fontsize=S["title_fs"])
        ax.tick_params(labelsize=S["tick_fs"])
        ax.legend(loc="lower right", fontsize=S["legend_fs"])

        ax = axes[r, 1]
        sort_idx = np.argsort(probs)
        colors = ["red" if y == 1 else "blue" for y in y_true[sort_idx]]
        ax.scatter(np.arange(len(probs)), probs[sort_idx], c=colors, s=6, alpha=0.5)
        ax.axhline(0.5, color="k", lw=0.8, linestyle="--")
        ax.set_xlabel("Sample (sorted by score)", fontsize=S["label_fs"])
        ax.set_ylabel("Predicted probability", fontsize=S["label_fs"])
        ax.set_title(f"Probability Scatter (AUPRC={auprc:.4f})" if has_both else "Probability Scatter", fontsize=S["title_fs"])
        ax.tick_params(labelsize=S["tick_fs"])
        ax.legend(handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=7, label="pos"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="blue", markersize=7, label="neg"),
        ], fontsize=S["legend_fs"])

        ax = axes[r, 2]
        bins = np.linspace(0, 1, 31)
        ax.hist(probs[y_true == 0], bins=bins, alpha=0.6, color="blue", label=f"neg n={int((y_true == 0).sum())}")
        ax.hist(probs[y_true == 1], bins=bins, alpha=0.6, color="red",  label=f"pos n={int((y_true == 1).sum())}")
        ax.set_xlabel("Predicted probability", fontsize=S["label_fs"])
        ax.set_ylabel("Count", fontsize=S["label_fs"])
        ax.set_title(f"Score Distribution (Brier={brier:.4f})", fontsize=S["title_fs"])
        ax.tick_params(labelsize=S["tick_fs"])
        ax.legend(fontsize=S["legend_fs"])

        metrics_list.append({"auroc": auroc, "auprc": auprc, "brier": brier})

    plt.tight_layout()
    fig.savefig(Path(plots_dir) / f"{save_name}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return metrics_list
