from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_fully_bayesian import Config
from loader import Dataset
from reward.fully_bayesian.features import build_feature_pipeline

OUT = ROOT / "outputs/lab/fully_bayesian/feature_correlation"


def build_phi(cfg):
    users = list(cfg.train) + list(cfg.test)
    data = Dataset(ROOT / "datasets")
    Xs, ys = map(list, zip(*(cfg.view(data[n]) for n in users)))
    pipe = build_feature_pipeline(cfg)
    pipe.fit(Xs, ys)
    Phi = np.concatenate([pipe.transform(X) for X in Xs], axis=0)
    return Phi, list(pipe.feature_names), list(pipe.groups)


def plot_heatmap(corr, names, title, path):
    n = len(names)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.16), max(7, n * 0.16)))
    im = ax.imshow(np.nan_to_num(corr), vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=90, fontsize=4)
    ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=4)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200); plt.close(fig)


def top_pairs(corr, names, k=25):
    iu = np.triu_indices_from(corr, k=1)
    r = corr[iu]
    keep = np.isfinite(r)
    i, j, r = iu[0][keep], iu[1][keep], r[keep]
    order = np.argsort(-np.abs(r))[:k]
    return [(names[i[o]], names[j[o]], float(r[o])) for o in order]


def main():
    base = Config()
    for method in ["manual", "manual_pca"]:
        cfg = replace(base, feature_method=method)
        Phi, names, _ = build_phi(cfg)
        corr = np.corrcoef(Phi.T)
        plot_heatmap(corr, names, f"feature corr  ({method})  {Phi.shape[0]} x {len(names)}",
                     OUT / f"corr_{method}.png")
        off = np.abs(corr[np.triu_indices_from(corr, 1)])
        off = off[np.isfinite(off)]
        print(f"\n=== {method}: {Phi.shape[0]} samples x {len(names)} feats  "
              f"(mean|r|={off.mean():.3f}, frac|r|>0.8={np.mean(off > 0.8):.3f}) ===")
        for a, b, r in top_pairs(corr, names):
            print(f"  {r:+.3f}  {a}  <->  {b}")
    print(f"\nsaved heatmaps -> {OUT}")


if __name__ == "__main__":
    main()
