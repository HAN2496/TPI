"""
AE/VAE embedding quality: distance correlation + kNN overlap
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_encoders import Config, load_data, train_one, get_latent, plot_curves

K = 10
SUBSAMPLE = 1000
EXP_NAME = "AE-z16"
N_SHOW = 4  # 랜덤 anchor 수


def cosine_sim_matrix(Z):
    norm = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
    Zn = Z / norm
    return Zn @ Zn.T  # (N, N), [-1, 1]


def knn_overlap(Z_raw, Z_lat, k):
    N = len(Z_raw)
    k = min(k, N - 1)

    def topk_idx(S):
        np.fill_diagonal(S, -np.inf)
        return np.argpartition(-S, k, axis=1)[:, :k]

    raw_idx = topk_idx(cosine_sim_matrix(Z_raw))
    lat_idx = topk_idx(cosine_sim_matrix(Z_lat))
    overlaps = [len(set(r) & set(l)) / k for r, l in zip(raw_idx, lat_idx)]
    return float(np.mean(overlaps))


def dist_correlation(Z_raw, Z_lat, n=SUBSAMPLE, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Z_raw), size=min(n, len(Z_raw)), replace=False)
    d_raw = pdist(Z_raw[idx], metric="cosine")
    d_lat = pdist(Z_lat[idx], metric="cosine")
    return float(spearmanr(d_raw, d_lat).statistic)


def main():
    cfg = Config()
    out_dir = Path(cfg.out_dir)

    print("Loading data...")
    X_tr, y_tr, d_tr, X_va, y_va, d_va, X_te, y_te = load_data(cfg)
    T, D = X_tr.shape[1], X_tr.shape[2]

    # raw latent: flatten + standardize (baseline)
    X_va_flat = X_va.reshape(len(X_va), -1)
    X_va_flat = (X_va_flat - X_va_flat.mean(0)) / (X_va_flat.std(0) + 1e-6)

    exp = next(e for e in cfg.experiments if e.name == EXP_NAME)
    print(f"\nTraining [{exp.name}]...")
    model, hist = train_one(exp, X_tr, X_va, T, D, cfg.device)

    plot_curves([dict(exp=exp, hist=hist)], out_dir)

    Z_lat = get_latent(model, exp.model, X_va, T, D, cfg.device)
    rho = dist_correlation(X_va_flat, Z_lat)
    overlap = knn_overlap(X_va_flat, Z_lat, k=K)
    random_baseline = K / (len(X_va) - 1)

    print(f"\n[{exp.name}]")
    print(f"  dist_corr  (Spearman ρ) = {rho:.4f}")
    print(f"  knn_overlap (k={K})     = {overlap:.4f}  (random={random_baseline:.4f})")

    X_recon = get_recon(model, exp.model, X_va, T, D, cfg.device)
    plot_pairs(X_va, X_recon, Z_lat, X_va_flat, cfg.view.cols, exp.name, out_dir, seed=0)


def get_recon(model, model_type, X, T, D, device, bs=512):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from compare_encoders import to_tensor
    dl = DataLoader(TensorDataset(to_tensor(X, T, D)), batch_size=bs, shuffle=False)
    out = []
    with torch.no_grad():
        for (bx,) in dl:
            xh = model(bx.to(device))[0]
            out.append(xh.cpu().numpy().transpose(0, 2, 1))  # (B, T, D)
    return np.concatenate(out)


def plot_pairs(X, X_recon, Z_lat, X_flat, feature_names, exp_name, out_dir, seed=0):
    N, T, D = X.shape
    rng = np.random.default_rng(seed)
    anchors = rng.choice(N, size=N_SHOW, replace=False)

    S_lat = cosine_sim_matrix(Z_lat)
    S_raw = cosine_sim_matrix(X_flat)
    np.fill_diagonal(S_lat, -np.inf)
    near_idx    = S_lat.argmax(axis=1)
    np.fill_diagonal(S_lat, np.inf)
    furthest_idx = S_lat.argmin(axis=1)

    fig, axes = plt.subplots(N_SHOW, D, figsize=(D * 3, N_SHOW * 2.5), squeeze=False)
    t = np.arange(T)

    for row, i in enumerate(anchors):
        near_j = near_idx[i]
        far_j  = furthest_idx[i]

        raw_near = S_raw[i, near_j]
        lat_near = S_lat[i, near_j] if not np.isinf(S_lat[i, near_j]) else cosine_sim_matrix(Z_lat[[i, near_j]])[0, 1]
        raw_far  = S_raw[i, far_j]
        lat_far  = cosine_sim_matrix(Z_lat[[i, far_j]])[0, 1]

        for d in range(D):
            ax = axes[row, d]
            ax.plot(t, X[i, :, d],            label="gt")
            ax.plot(t, X_recon[i, :, d], "--", label="same (recon)")
            ax.plot(t, X[near_j, :, d],  "-.", label=f"nearest (raw={raw_near:.2f}, lat={lat_near:.2f})")
            ax.plot(t, X[far_j, :, d],   ":",  label=f"furthest (raw={raw_far:.2f}, lat={lat_far:.2f})")
            if d == 0:
                ax.set_ylabel(f"anchor {i}", fontsize=8)
            ax.set_title(feature_names[d], fontsize=8)
            ax.legend(fontsize=6)

    fig.suptitle(exp_name, fontsize=11)
    fig.tight_layout()
    path = out_dir / "trajectory_pairs.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"Saved → {path}")


if __name__ == "__main__":
    main()
