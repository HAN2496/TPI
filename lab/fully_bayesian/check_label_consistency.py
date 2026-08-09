from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from run_fully_bayesian import Config
from loader import Dataset, View
from reward.fully_bayesian.features import build_feature_pipeline

OUT = ROOT / "outputs/lab/fully_bayesian/label_consistency"
ANOM_Q = 0.01          # 같은-라벨 쌍 거리의 하위 1% 보다 가까운 다른-라벨 쌍 = 모순
N_PLOTS = 10           # 랜덤 anchor 플롯 장수
SEED = 0               # 랜덤 사용자/base 선택 시드
MODE = "topk"      # "near_far"(base+최근접+최원거리) | "topk"(base+최근접 N개)
TOPK = 5               # topk 모드에서 겹칠 최근접 개수

# 비교에 쓸 센서 채널. 여기서 직접 고른다 (Config 기본값 대신 이걸 씀).
AROUND = (-2.0, 1.0)     # event_time(=5s) 기준. 구 time_range=(3.0, 6.0)과 동일
DOWNSAMPLE = 1
SMOOTH = None            # None | (cutoff_hz, order)
METHOD = "manual" # None | "manual" | "manual_pca" | "ae"
FEATURES = [
    # "Pitch_rate_6D",
    # "Bounce_rate_6D",
    # "IMU_LongAccelVal",
    "IMU_VerAccelVal",
    # "IMU_LatAccelVal",
    # "IMU_YawRtVal",
    # "Roll_rate_6D",
    # "SAS_AnglVal",
    # "SAS_SpdVal",
    # "IMU_RollRtVal",
    # "VCU_AccPedDepVal",
    # "IEB_StrkDpthPcVal",
    # "IEB_BrkActvSta",
    # "IEB_EstTtlBrkFrcNmV",
]
PLOT_CHANNELS = ["IMU_LongAccelVal", "IMU_LatAccelVal", "Pitch_rate_6D",
                 "Bounce_rate_6D", "IEB_StrkDpthPcVal"]


def load_users(cfg, root="datasets"):
    data = Dataset(root)
    out = {}
    for name in list(cfg.train) + list(cfg.test):
        eps = [ep for ep in data[name] if ep.label is not None]
        X = np.stack([cfg.view.matrix(ep) for ep in eps])
        y = np.array([int(ep.label) for ep in eps])
        out[name] = (X, y, eps)
    return out


def analyze_user(Phi, y):
    n = len(y)
    iu = np.triu_indices(n, 1)
    d = cdist(Phi, Phi)[iu]
    same = y[iu[0]] == y[iu[1]]
    d_same, d_diff = d[same], d[~same]

    r = rankdata(np.concatenate([d_diff, d_same]))[:len(d_diff)]
    auc = (r.sum() - len(d_diff) * (len(d_diff) + 1) / 2) / (len(d_diff) * len(d_same))

    q = np.searchsorted(np.sort(d_same), d) / len(d_same)   # 각 쌍의 같은-라벨 거리분포 내 분위
    anom = (~same) & (q < ANOM_Q)
    contra = sorted(zip(d[anom].tolist(), q[anom].tolist(), iu[0][anom].tolist(), iu[1][anom].tolist()))
    return dict(n=n, pos=int(y.sum()), auc=float(auc),
                med_same=float(np.median(d_same)), med_diff=float(np.median(d_diff)),
                n_contra=int(anom.sum()), contra=contra, d=d, same=same, iu=iu)


def plot_auc_bar(names, res, path):
    auc = [res[n]["auc"] for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(7, len(names) * 0.7), 4))
    ax.bar(x, auc, color=["#d62728" if a < 0.5 else "#2ca02c" for a in auc])
    ax.axhline(0.5, color="k", ls="--", lw=1, label="chance (no relation)")
    ax.set_xticks(x); ax.set_xticklabels([f"U{i}" for i in range(len(names))])
    ax.set_ylim(0, 1.0); ax.set_ylabel("separation AUC"); ax.legend()
    ax.set_title("does feature-similarity predict same label? (>0.5 = consistent)")
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150); plt.close(fig)


def plot_distance_agreement(res, names, path, bins=12):
    d = np.concatenate([res[n]["d"] for n in names])
    s = np.concatenate([res[n]["same"] for n in names]).astype(float)
    edges = np.quantile(d, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.digitize(d, edges[1:-1]), 0, bins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    rate = np.array([s[idx == b].mean() if (idx == b).any() else np.nan for b in range(bins)])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(centers, rate, "o-")
    ax.set_xlabel("feature distance (z-space)")
    ax.set_ylabel("label agreement rate"); ax.set_ylim(0, 1.05)
    ax.set_title("agreement vs similarity (pooled intra-user pairs)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def select_items(Phi, base, mode):
    dist = cdist(Phi[base:base + 1], Phi)[0]
    order = [int(i) for i in np.argsort(dist) if i != base]
    if mode == "near_far":
        far = int(np.argmax(dist))
        return [(order[0], float(dist[order[0]]), "tab:green"),
                (far, float(dist[far]), "tab:red")], "near(green) / far(red)"
    cmap = plt.cm.viridis(np.linspace(0.0, 0.85, TOPK))
    items = [(order[j], float(dist[order[j]]), cmap[j]) for j in range(min(TOPK, len(order)))]
    return items, f"top-{TOPK} nearest (dark=closest)"


def plot_anchor(name, X, y, base, items, features, path, subtitle):
    lab = lambda i: "T" if y[i] else "F"
    ls = lambda i: "-" if y[i] else "--"   # T=실선, F=점선
    chans = [c for c in PLOT_CHANNELS if c in features]
    ci = [features.index(c) for c in chans]
    fig, axes = plt.subplots(1, len(chans), figsize=(3.0 * len(chans), 3.4), squeeze=False, sharex=True)
    for c, (cn, idx) in enumerate(zip(chans, ci)):
        ax = axes[0][c]
        ax.plot(X[base][:, idx], color="k", ls=ls(base), lw=1.8, label=f"base ep{base}({lab(base)})")
        for e, dist, col in items:
            ax.plot(X[e][:, idx], color=col, ls=ls(e), lw=1.0, label=f"ep{e}({lab(e)}) d={dist:.2f}")
        ax.set_title(cn, fontsize=8); ax.tick_params(labelsize=6)
    axes[0][0].legend(fontsize=6, loc="best")
    fig.suptitle(f"{name}: {subtitle}   |  solid=True, dashed=False", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150); plt.close(fig)


def raw_phi(users, names):
    # feature 없이: 채널별 z-score 후 (T,D)->(T*D) flatten 한 raw 시계열 자체를 벡터로
    Xall = np.concatenate([users[n][0] for n in names], axis=0)
    mu = Xall.mean(axis=(0, 1)); sd = Xall.std(axis=(0, 1)) + 1e-8
    return {n: ((users[n][0] - mu) / sd).reshape(len(users[n][0]), -1) for n in names}


def main():
    cfg = replace(Config(),
                  view=View(features=tuple(FEATURES), around=AROUND,
                            downsample=DOWNSAMPLE, smooth=SMOOTH),
                  feature_method=METHOD)   # feature_method=None 이면 raw 직접 비교
    OUT.mkdir(parents=True, exist_ok=True)
    users = load_users(cfg)
    names = list(users)
    if cfg.feature_method is None:
        phi = raw_phi(users, names)
    else:
        pipe = build_feature_pipeline(cfg)
        pipe.fit([users[n][0] for n in names], [users[n][1] for n in names])
        phi = {n: pipe.transform(users[n][0]) for n in names}
    res = {n: analyze_user(phi[n], users[n][1]) for n in names}

    print(f"{'idx':<4}{'user':<10}{'n':>5}{'pos':>5}{'sep_auc':>9}{'med_same':>10}{'med_diff':>10}{'#contra':>9}")
    for i, n in enumerate(names):
        r = res[n]
        print(f"U{i:<3}{n:<10}{r['n']:>5}{r['pos']:>5}{r['auc']:>9.3f}"
              f"{r['med_same']:>10.3f}{r['med_diff']:>10.3f}{r['n_contra']:>9}")

    print(f"\n=== 모순쌍: 라벨 반대인데 같은-라벨 쌍 하위 {ANOM_Q:.0%} 보다 가까움 (사용자별 top 3) ===")
    for i, n in enumerate(names):
        for dist, q, a, b in res[n]["contra"][:3]:
            print(f"  U{i} {n}: ep{a} vs ep{b}  dist={dist:.3f}  (같은라벨 거리 분위 {q:.1%})")

    # 랜덤 사용자/base 로 base vs near/far 비교 플롯 N_PLOTS 장
    rng = np.random.default_rng(SEED)
    adir = OUT / "anchors"; adir.mkdir(exist_ok=True)
    print(f"\n=== random anchor plots (seed={SEED}, mode={MODE}) → {adir} ===")
    for k in range(N_PLOTS):
        u = str(rng.choice(names))
        X, y, eps = users[u]
        base = int(rng.integers(len(y)))
        items, subtitle = select_items(phi[u], base, MODE)
        plot_anchor(u, X, y, base, items, cfg.view.cols, adir / f"anchor_{k:02d}.png", subtitle)
        mk = lambda e: f"ep{e}({'T' if y[e] else 'F'},{eps[e].id})"
        others = " | ".join(f"{mk(e)} d={d:.2f}" for e, d, _ in items)
        print(f"  [{k:02d}] {u}: base {mk(base)} | {others}")

    plot_auc_bar(names, res, OUT / "separation_auc.png")
    plot_distance_agreement(res, names, OUT / "distance_agreement.png")
    print(f"\nidx->user: " + ", ".join(f"U{i}={n}" for i, n in enumerate(names)))
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
