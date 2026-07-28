import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

STYLE = {
    "paper": dict(label_fs=22, tick_fs=20, legend_fs=16, lw=1.5),
    "diag":  dict(label_fs=14, tick_fs=12, legend_fs=14,
                  title_fs=18, lw=1.2, scatter_s=18, annot_fs=12),
}


def plot_sequential_auroc(ctx_sizes_list, aurocs_list, plots_dir, names, save_name="seq_auroc"):
    if not any(aurocs_list):
        return
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    n = len(names)
    S = STYLE["diag"]
    import numpy as np
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


def plot_training_curves(history, save_path, title=""):
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
