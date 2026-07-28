"""
Wavelet (Morlet/Gabor) scattering encoder vs AE/VAE for the CoPL item-similarity graph.

Suspension/vibration 신호 → scattering transform이 transient(범프) 표현에 강함.
새 의존성 없이 scipy FFT로 1차+2차 scattering을 직접 구현 (T=40 같은 짧은 윈도우용).

비교 지표 (CoPL이 실제 쓰는 방식과 정합):
  - knn_auroc : test item을 train item에 cosine-kNN으로 잇고 라벨 투표 → CoPL get_affinity와 동일
  - logreg_auroc : train-driver 임베딩으로 학습 → test-driver에서 평가 (cross-driver 일반화)
  - sil_label : pos/neg(이벤트) 분리도
  - sil_driver : 운전자 분리도 (graph 전이 관점에선 너무 높으면 오히려 안 좋음)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataclasses import dataclass
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from core import seed_all
from compare_encoders import Config, load_data, train_one, get_latent, ExpConfig


# ── Wavelet scattering (Morlet/Gabor, FFT 기반, global average pooling) ──────────

class GaborScattering:
    """1차 + 2차 Gabor scattering. 채널별로 계산해 concat.
    S1_j  = mean_t |x * psi_j|
    S2_jk = mean_t ||x * psi_j| * psi_k|   (k가 j보다 낮은 주파수: k>j)
    """
    def __init__(self, T, n1=8, f_min=0.03, f_max=0.45, q=0.5, order2=True):
        self.T = T
        self.N = int(2 ** np.ceil(np.log2(T)))   # FFT 길이 (40 → 64)
        self.n1 = n1
        self.order2 = order2
        omega = np.fft.fftfreq(self.N)           # cycles/sample, [-0.5, 0.5)
        freqs = np.geomspace(f_max, f_min, n1)   # 고주파 → 저주파
        self.freqs = freqs
        psi = np.empty((n1, self.N))
        for i, f in enumerate(freqs):
            sigma = q * f                         # constant-Q: 대역폭 ∝ 중심주파수
            psi[i] = np.exp(-0.5 * ((omega - f) / sigma) ** 2)
        self.psi = psi

    @property
    def out_dim_per_channel(self):
        d = 1 + self.n1
        if self.order2:
            d += self.n1 * (self.n1 - 1) // 2
        return d

    def transform(self, X):                       # X: (B, T, D) → (B, D*Fc)
        B, T, D = X.shape
        x = np.zeros((B, D, self.N))
        x[:, :, :T] = X.transpose(0, 2, 1)
        Xf = np.fft.fft(x, axis=-1)               # (B, D, N)

        U1 = np.abs(np.fft.ifft(Xf[:, :, None, :] * self.psi[None, None], axis=-1))  # (B,D,n1,N)
        S1 = U1.mean(axis=-1)                      # (B, D, n1)
        S0 = np.abs(x).mean(axis=-1, keepdims=True)  # (B, D, 1)
        feats = [S0, S1]

        if self.order2:
            U1f = np.fft.fft(U1, axis=-1)
            pairs = [(j, k) for j in range(self.n1) for k in range(j + 1, self.n1)]
            S2 = np.stack([
                np.abs(np.fft.ifft(U1f[:, :, j, :] * self.psi[k][None, None], axis=-1)).mean(axis=-1)
                for j, k in pairs
            ], axis=-1)                            # (B, D, n_pairs)
            feats.append(S2)

        F = np.concatenate(feats, axis=-1)         # (B, D, Fc)
        F = np.log1p(F)                            # 진폭 압축
        return F.reshape(B, -1)


# ── Encoders → (Z_tr, Z_va, Z_te) ──────────────────────────────────────────────

def encode_ae(exp, data, T, D, device):
    X_tr, _, _, X_va, _, _, X_te, _ = data
    model, _ = train_one(exp, X_tr, X_va, T, D, device)
    f = lambda X: get_latent(model, exp.model, X, T, D, device)
    return f(X_tr), f(X_va), f(X_te)


def encode_raw(data):
    X_tr, _, _, X_va, _, _, X_te, _ = data
    flat = lambda X: X.reshape(len(X), -1)
    sc = StandardScaler().fit(flat(X_tr))
    return sc.transform(flat(X_tr)), sc.transform(flat(X_va)), sc.transform(flat(X_te))


def encode_scatter(data, T, D, n1=8, pca_dim=None):
    X_tr, _, _, X_va, _, _, X_te, _ = data
    sca = GaborScattering(T, n1=n1)
    F_tr, F_va, F_te = sca.transform(X_tr), sca.transform(X_va), sca.transform(X_te)
    sc = StandardScaler().fit(F_tr)
    Z_tr, Z_va, Z_te = sc.transform(F_tr), sc.transform(F_va), sc.transform(F_te)
    if pca_dim:
        pca = PCA(n_components=pca_dim, random_state=42).fit(Z_tr)
        Z_tr, Z_va, Z_te = pca.transform(Z_tr), pca.transform(Z_va), pca.transform(Z_te)
    print(f"    scatter dim={Z_tr.shape[1]} (per-channel={sca.out_dim_per_channel}, D={D})")
    return Z_tr, Z_va, Z_te


# ── Metrics ────────────────────────────────────────────────────────────────────

def _l2(Z):
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)


def knn_label_auroc(Z_tr, y_tr, Z_te, y_te, k=10):
    """test item → train item cosine-kNN → 이웃 라벨 평균 (CoPL get_affinity 메커니즘)."""
    nn = NearestNeighbors(n_neighbors=min(k, len(Z_tr)), metric="euclidean").fit(_l2(Z_tr))
    _, idx = nn.kneighbors(_l2(Z_te))
    score = y_tr[idx].mean(axis=1)
    return roc_auc_score(y_te, score)


def logreg_auroc(Z_tr, y_tr, Z_te, y_te):
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Z_tr, y_tr)
    return roc_auc_score(y_te, clf.predict_proba(Z_te)[:, 1])


def eval_encoder(name, Z_tr, Z_va, Z_te, data):
    _, y_tr, _, _, y_va, d_va, _, y_te = data
    n_drv = int(d_va.max()) + 1
    Z_all = np.vstack([Z_va, Z_te])
    y_all = np.concatenate([y_va, y_te])
    d_all = np.concatenate([d_va, np.full(len(Z_te), n_drv)])
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(Z_all) - 1))
    Z2 = tsne.fit_transform(Z_all)
    m = dict(
        name=name, dim=Z_tr.shape[1],
        knn_auroc=knn_label_auroc(Z_tr, y_tr, Z_te, y_te),
        logreg_auroc=logreg_auroc(Z_tr, y_tr, Z_te, y_te),
        sil_label=silhouette_score(Z_all, y_all) if len(np.unique(y_all)) > 1 else 0.0,
        sil_driver=silhouette_score(Z_all, d_all) if len(np.unique(d_all)) > 1 else 0.0,
        Z2=Z2, split=len(Z_va), y_all=y_all, d_all=d_all,
    )
    print(f"  {name:16s} dim={m['dim']:4d}  knn_auroc={m['knn_auroc']:.4f}  "
          f"logreg_auroc={m['logreg_auroc']:.4f}  sil_lbl={m['sil_label']:+.4f}  sil_drv={m['sil_driver']:+.4f}")
    return m


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_summary(metrics, out_dir):
    names = [m["name"] for m in metrics]
    x = np.arange(len(names))
    keys = [("knn_auroc", "kNN label AUROC (test→train)"),
            ("logreg_auroc", "LogReg AUROC (cross-driver)"),
            ("sil_label", "Silhouette by label"),
            ("sil_driver", "Silhouette by driver")]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for ax, (k, title) in zip(axes, keys):
        vals = [m[k] for m in metrics]
        bars = ax.bar(x, vals, color=["#888" if "Scatter" not in n else "#d6604d" for n in names])
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        if "auroc" in k:
            ax.axhline(0.5, color="gray", ls="--", lw=0.8); ax.set_ylim(0.4, 1.0)
        else:
            ax.axhline(0, color="gray", lw=0.8)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=7)
    fig.suptitle("Scattering vs AE/VAE for CoPL similarity graph", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "scattering_summary.png", dpi=120); plt.close(fig)


def plot_tsne(metrics, driver_names, out_dir):
    n = len(metrics)
    fig, axes = plt.subplots(2, n, figsize=(3.6 * n, 7.4), squeeze=False)
    for col, m in enumerate(metrics):
        Z2, split, y_all, d_all = m["Z2"], m["split"], m["y_all"], m["d_all"]
        for row, (lab, cmap, title) in enumerate([(d_all, "tab10", "driver"), (y_all, "bwr", "label")]):
            ax = axes[row, col]
            ax.scatter(Z2[:split, 0], Z2[:split, 1], c=lab[:split], cmap=cmap, s=7, alpha=0.5,
                       vmin=lab.min(), vmax=lab.max())
            ax.scatter(Z2[split:, 0], Z2[split:, 1], c=lab[split:], cmap=cmap, s=12, alpha=0.9,
                       marker="^", vmin=lab.min(), vmax=lab.max())
            ax.set_title(f"{m['name']} ({title})", fontsize=8); ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "scattering_tsne.png", dpi=120); plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    cfg = Config()
    seed_all(cfg.seed)
    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    import torch
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    print("Loading data...")
    data = load_data(cfg)
    X_tr = data[0]
    T, D = X_tr.shape[1], X_tr.shape[2]
    print(f"T={T}, D={D}  train={len(data[0])}  val={len(data[3])}  test={len(data[6])}")

    EP = 500
    metrics = []

    print("\n[Raw-flat]")
    metrics.append(eval_encoder("Raw-flat", *encode_raw(data), data))

    for name, exp in [
        ("AE-z8",  ExpConfig("AE-z8",  "ae",  latent_dim=8,  epochs=EP)),   # 현재 production
        ("AE-z16", ExpConfig("AE-z16", "ae",  latent_dim=16, epochs=EP)),
        ("VAE-z16", ExpConfig("VAE-z16", "vae", latent_dim=16, kl_weight=0.05, epochs=EP)),
    ]:
        print(f"\n[{name}] training...")
        metrics.append(eval_encoder(name, *encode_ae(exp, data, T, D, device), data))

    print("\n[Scatter] (Morlet scattering)")
    metrics.append(eval_encoder("Scatter", *encode_scatter(data, T, D, n1=8), data))
    print("[Scatter-PCA16]")
    metrics.append(eval_encoder("Scatter-PCA16", *encode_scatter(data, T, D, n1=8, pca_dim=16), data))

    print("\nSaving plots...")
    plot_summary(metrics, out_dir)
    plot_tsne(metrics, cfg.train_drivers, out_dir)

    print("\n=== Summary (sorted by knn_auroc) ===")
    for m in sorted(metrics, key=lambda r: -r["knn_auroc"]):
        print(f"  {m['name']:16s} knn={m['knn_auroc']:.4f}  logreg={m['logreg_auroc']:.4f}  "
              f"sil_lbl={m['sil_label']:+.4f}  sil_drv={m['sil_driver']:+.4f}")
    print(f"\nDone → {out_dir}")


if __name__ == "__main__":
    main()
