import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

from .plot_style import STYLE

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


def plot_sequential_auroc(ctx_sizes_list, aurocs_list, plots_dir, names, save_name="seq_auroc"):
    if not any(aurocs_list):
        return
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    n = len(names)
    S = STYLE["diag"]
    fig, axes = plt.subplots(n, 1, figsize=(8, 4 * n))
    axes = np.atleast_1d(axes)
    for ax, ctx_sizes, aurocs, name in zip(axes, ctx_sizes_list, aurocs_list, names):
        if not aurocs:
            ax.set_visible(False)
            continue
        max_auroc = max(aurocs)
        max_ctx = ctx_sizes[aurocs.index(max_auroc)]
        ax.plot(ctx_sizes, aurocs, color="steelblue", lw=S["lw"])
        ax.axhline(max_auroc, color="crimson", lw=1.0, linestyle="--",
                   label=f"Max AUROC={max_auroc:.4f} @ ctx={max_ctx}")
        ax.set_xlabel("Context size", fontsize=S["label_fs"])
        ax.set_ylabel("Cumulative AUROC", fontsize=S["label_fs"])
        ax.set_title(f"Sequential AUROC — {name}", fontsize=S["title_fs"])
        ax.tick_params(labelsize=S["tick_fs"])
        ax.legend(fontsize=S["legend_fs"])
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(Path(plots_dir) / f"{save_name}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def compute_sequential_aurocs(probs, labels, context_sizes):
    """누적 AUROC: predictions[0..k] 전체로 계산, 양 클래스 미등장 시 skip."""
    seq_x, seq_y = [], []
    for k in range(len(probs)):
        if len(set(labels[:k + 1])) == 2:
            seq_y.append(roc_auc_score(labels[:k + 1], probs[:k + 1]))
            seq_x.append(context_sizes[k])
    return seq_x, seq_y


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
    section_labels = {
        "test": "TEST",
        "wo_user_emb": "W/O USER EMB.",
        "train_user_pooled": "TRAIN-USER POOLED",
        "target_user_only": "TARGET-USER ONLY",
        "train": "TRAIN",
    }

    for section in ("test", "wo_user_emb", "train_user_pooled", "target_user_only", "train"):
        items = {k: v for k, v in metrics_by_name.items() if k.startswith(f"{section}/")}
        if not items:
            continue
        lines.append(f"[{section_labels.get(section, section.upper())}]")
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
