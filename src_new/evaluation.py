import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def evaluate_predictions(y_true, probs, plots_dir, name, title=""):
    y_true = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64).ravel()
    has_both = len(np.unique(y_true)) > 1

    auroc = roc_auc_score(y_true, probs) if has_both else float("nan")
    auprc = average_precision_score(y_true, probs) if has_both else float("nan")
    brier = float(np.mean((probs - y_true) ** 2))

    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(title or name, fontsize=11)

    # --- Panel 1: ROC curve ---
    ax = axes[0]
    if has_both:
        fpr, tpr, _ = roc_curve(y_true, probs)
        ax.plot(fpr, tpr, color="darkorange", lw=1.5, label=f"AUROC={auroc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right", fontsize=9)

    # --- Panel 2: Probability scatter (sorted by score) ---
    ax = axes[1]
    sort_idx = np.argsort(probs)
    colors = ["red" if y == 1 else "blue" for y in y_true[sort_idx]]
    ax.scatter(np.arange(len(probs)), probs[sort_idx], c=colors, s=6, alpha=0.5)
    ax.axhline(0.5, color="k", lw=0.8, linestyle="--")
    ax.set_xlabel("Sample (sorted by score)"); ax.set_ylabel("Predicted probability")
    ax.set_title(f"Probability Scatter  (AUPRC={auprc:.4f})" if has_both else "Probability Scatter")
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=7, label="pos"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="blue", markersize=7, label="neg"),
    ], fontsize=9)

    # --- Panel 3: Score histogram ---
    ax = axes[2]
    bins = np.linspace(0, 1, 31)
    pos_p = probs[y_true == 1]
    neg_p = probs[y_true == 0]
    ax.hist(neg_p, bins=bins, alpha=0.6, color="blue", label=f"neg  n={len(neg_p)}")
    ax.hist(pos_p, bins=bins, alpha=0.6, color="red",  label=f"pos  n={len(pos_p)}")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Count")
    ax.set_title(f"Score Distribution  (Brier={brier:.4f})")
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(Path(plots_dir) / f"{name}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    return {"auroc": auroc, "auprc": auprc, "brier": brier}


def compute_sequential_aurocs(probs, labels, context_sizes):
    """누적 AUROC: predictions[0..k] 전체로 계산, 양 클래스 미등장 시 skip."""
    seq_x, seq_y = [], []
    for k in range(len(probs)):
        if len(set(labels[:k + 1])) == 2:
            seq_y.append(roc_auc_score(labels[:k + 1], probs[:k + 1]))
            seq_x.append(context_sizes[k])
    return seq_x, seq_y


def plot_sequential_auroc(context_sizes, aurocs, plots_dir, name):
    if not aurocs:
        return
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    max_auroc = max(aurocs)
    max_ctx = context_sizes[aurocs.index(max_auroc)]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(context_sizes, aurocs, color="steelblue", lw=1.5)
    ax.axhline(max_auroc, color="crimson", lw=1.0, linestyle="--",
               label=f"Max AUROC={max_auroc:.4f} @ ctx={max_ctx}")
    ax.set_xlabel("Context size")
    ax.set_ylabel("Cumulative AUROC")
    ax.set_title(f"Sequential AUROC — {name}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(Path(plots_dir) / f"seq_auroc_{name}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history: dict, save_path, title=""):
    from collections import defaultdict
    groups = defaultdict(dict)
    for key, values in history.items():
        matched = False
        for prefix, split in [("train/", "train"), ("eval/", "eval"), ("val/", "val"),
                               ("train_", "train"), ("val_", "val")]:
            if key.startswith(prefix):
                groups[key[len(prefix):]][split] = values
                matched = True
                break
        if not matched:
            groups[key][""] = values

    n = len(groups)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    color_map = {"train": "steelblue", "eval": "darkorange", "val": "darkorange", "": "steelblue"}
    for ax, (name, splits) in zip(axes, groups.items()):
        for split, values in splits.items():
            ax.plot(values, color=color_map.get(split, "gray"), lw=1.2,
                    label=split if split else name)
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        if len(splits) > 1:
            ax.legend(fontsize=8)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_metrics_txt(metrics_by_name, out_path):
    lines = ["=" * 50, "Evaluation Metrics", "=" * 50, ""]

    for section in ("test", "train"):
        items = {k: v for k, v in metrics_by_name.items() if k.startswith(f"{section}/")}
        if not items:
            continue
        lines.append(f"[{section.upper()}]")
        for key, m in items.items():
            driver = key.split("/", 1)[1]
            auroc = f"{m['auroc']:.4f}" if not np.isnan(m["auroc"]) else "  N/A "
            auprc = f"{m['auprc']:.4f}" if not np.isnan(m["auprc"]) else "  N/A "
            lines += [
                f"  {driver}",
                f"    AUROC : {auroc}",
                f"    AUPRC : {auprc}",
                f"    Brier : {m['brier']:.4f}",
                "",
            ]

    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
