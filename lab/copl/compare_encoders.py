import copy
from dataclasses import dataclass, field
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from core import seed_all
from loader import Dataset, View
from reward.copl.similarity.vae import Conv1dVAE
from reward.copl.similarity.ae import Conv1dAE


@dataclass
class ExpConfig:
    name: str
    model: str = "vae"       # "vae" | "ae"
    latent_dim: int = 16
    hidden_channels: int = 32
    epochs: int = 1200
    lr: float = 0.001
    batch_size: int = 128
    kl_weight: float = 0.05  # VAE only


@dataclass
class Config:
    train_drivers: list = field(default_factory=lambda: [
        "김진명", "조현석", "한규택", "박재일", "이지환"
    ])
    test_driver: str = "강신길"
    view: View = View(
        features=("IMU_VerAccelVal", "Pitch_rate_6D", "Bounce_rate_6D", "IMU_LongAccelVal"),
        around=(0, 2), downsample=5, smooth=(15.0, 2))
    normalize: bool = True
    val_size: float = 0.1
    device: str = "cuda"
    seed: int = 42
    out_dir: str = "outputs/lab/copl"

    experiments: list = field(default_factory=lambda: [
        ExpConfig("AE-z8",            "ae",  latent_dim=8),
        ExpConfig("AE-z16",           "ae",  latent_dim=16),
        ExpConfig("AE-z32",           "ae",  latent_dim=32),
        ExpConfig("AE-z64",           "ae",  latent_dim=64),
        ExpConfig("VAE-z8-kl0.05",    "vae", latent_dim=8,  kl_weight=0.05),
        ExpConfig("VAE-z16-kl0.01",   "vae", latent_dim=16, kl_weight=0.01),
        ExpConfig("VAE-z16-kl0.05",   "vae", latent_dim=16, kl_weight=0.05),
        ExpConfig("VAE-z16-kl0.1",    "vae", latent_dim=16, kl_weight=0.1),
        ExpConfig("VAE-z16-kl0.5",    "vae", latent_dim=16, kl_weight=0.5),
        ExpConfig("VAE-z32-kl0.05",   "vae", latent_dim=32, kl_weight=0.05),
    ])


# ── Data ────────────────────────────────────────────────────────────────────

def load_data(cfg):
    data = Dataset("datasets")
    Xs, ys, ds = [], [], []
    for i, drv in enumerate(cfg.train_drivers):
        X, y = cfg.view(data[drv])
        Xs.append(X.astype(np.float32))
        ys.append(y.astype(np.int64))
        ds.append(np.full(len(X), i, dtype=np.int64))
        print(f"  {drv}: n={len(X)}, pos={int(y.sum())}")

    X_all = np.concatenate(Xs)
    y_all = np.concatenate(ys)
    d_all = np.concatenate(ds)

    X_te, y_te = cfg.view(data[cfg.test_driver])
    X_te = X_te.astype(np.float32)
    y_te = y_te.astype(np.int64)
    print(f"  {cfg.test_driver}(test): n={len(X_te)}, pos={int(y_te.sum())}")

    if cfg.normalize:
        mean = X_all.mean(axis=(0, 1), keepdims=True)
        std  = X_all.std(axis=(0, 1), keepdims=True) + 1e-6
        X_all = (X_all - mean) / std
        X_te  = (X_te  - mean) / std

    idx = np.arange(len(X_all))
    tr_idx, va_idx = train_test_split(idx, test_size=cfg.val_size, random_state=cfg.seed)
    return (X_all[tr_idx], y_all[tr_idx], d_all[tr_idx],
            X_all[va_idx], y_all[va_idx], d_all[va_idx],
            X_te, y_te)


# ── Model ───────────────────────────────────────────────────────────────────

def to_tensor(X, T, D):
    return torch.tensor(X.reshape(-1, T, D).transpose(0, 2, 1), dtype=torch.float32)


def train_one(exp, X_tr, X_va, T, D, device):
    ModelClass = Conv1dVAE if exp.model == "vae" else Conv1dAE
    model = ModelClass(D, T, exp.latent_dim, exp.hidden_channels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=exp.lr)

    tr_dl = DataLoader(TensorDataset(to_tensor(X_tr, T, D)), batch_size=exp.batch_size, shuffle=True)
    va_dl = DataLoader(TensorDataset(to_tensor(X_va, T, D)), batch_size=exp.batch_size, shuffle=False)

    is_vae = exp.model == "vae"
    hist = {"train_loss": [], "val_loss": []}
    if is_vae:
        hist.update({"train_recon": [], "val_recon": [], "train_kl": [], "val_kl": []})
    best_val, best_state = float("inf"), None

    for ep in range(exp.epochs):
        model.train()
        tl = tr = tk = 0.0
        for (bx,) in tr_dl:
            bx = bx.to(device)
            if is_vae:
                xh, mu, lv = model(bx)
                recon = F.mse_loss(xh, bx)
                kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
                loss = recon + exp.kl_weight * kl
                tr += recon.item() * bx.size(0)
                tk += kl.item() * bx.size(0)
            else:
                xh, _ = model(bx)
                loss = F.mse_loss(xh, bx)
            tl += loss.item() * bx.size(0)
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        vl = vr = vk = 0.0
        with torch.no_grad():
            for (bx,) in va_dl:
                bx = bx.to(device)
                if is_vae:
                    xh, mu, lv = model(bx)
                    recon = F.mse_loss(xh, bx)
                    kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
                    loss = recon + exp.kl_weight * kl
                    vr += recon.item() * bx.size(0)
                    vk += kl.item() * bx.size(0)
                else:
                    xh, _ = model(bx)
                    loss = F.mse_loss(xh, bx)
                vl += loss.item() * bx.size(0)

        n_tr, n_va = len(X_tr), len(X_va)
        hist["train_loss"].append(tl / n_tr)
        hist["val_loss"].append(vl / n_va)
        if is_vae:
            hist["train_recon"].append(tr / n_tr)
            hist["val_recon"].append(vr / n_va)
            hist["train_kl"].append(tk / n_tr)
            hist["val_kl"].append(vk / n_va)

        if vl / n_va < best_val:
            best_val = vl / n_va
            best_state = copy.deepcopy(model.state_dict())

        if (ep + 1) % 100 == 0:
            print(f"    [{exp.name}] ep{ep+1}/{exp.epochs}: val_loss={vl/n_va:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    return model, hist


# ── Metrics ─────────────────────────────────────────────────────────────────

def get_latent(model, model_type, X, T, D, device, bs=512):
    dl = DataLoader(TensorDataset(to_tensor(X, T, D)), batch_size=bs, shuffle=False)
    Z = []
    with torch.no_grad():
        for (bx,) in dl:
            bx = bx.to(device)
            if model_type == "vae":
                mu, _ = model.encode(bx)
                Z.append(mu.cpu().numpy())
            else:
                Z.append(model.encode(bx).cpu().numpy())
    return np.concatenate(Z)


def get_recon_mse(model, model_type, X, T, D, device, bs=512):
    dl = DataLoader(TensorDataset(to_tensor(X, T, D)), batch_size=bs, shuffle=False)
    total, feat = 0.0, np.zeros(D)
    with torch.no_grad():
        for (bx,) in dl:
            bx = bx.to(device)
            xh = model(bx)[0]
            per = F.mse_loss(xh, bx, reduction="none").mean(dim=2).cpu().numpy()  # (B, D)
            feat  += per.sum(axis=0)
            total += per.mean(axis=1).sum()
    return total / len(X), feat / len(X)


def compute_all_metrics(results, X_va, y_va, d_va, X_te, y_te, T, D, device, n_train_drivers):
    metrics = []
    for exp, model, hist in results:
        vm, fv = get_recon_mse(model, exp.model, X_va, T, D, device)
        tm, ft = get_recon_mse(model, exp.model, X_te, T, D, device)
        Z_va = get_latent(model, exp.model, X_va, T, D, device)
        Z_te = get_latent(model, exp.model, X_te, T, D, device)
        Z_all = np.vstack([Z_va, Z_te])
        d_all = np.concatenate([d_va, np.full(len(Z_te), n_train_drivers, dtype=np.int64)])
        y_all = np.concatenate([y_va, y_te])
        sil_d = silhouette_score(Z_all, d_all) if len(np.unique(d_all)) > 1 else 0.0
        sil_l = silhouette_score(Z_all, y_all) if len(np.unique(y_all)) > 1 else 0.0
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(Z_all) - 1))
        Z2 = tsne.fit_transform(Z_all)
        metrics.append(dict(
            exp=exp, hist=hist,
            val_mse=float(vm), test_mse=float(tm),
            fmse_val=fv, fmse_test=ft,
            Z_va=Z_va, Z_te=Z_te, Z2=Z2,
            d_all=d_all, y_all=y_all,
            sil_driver=sil_d, sil_label=sil_l,
        ))
        print(f"  {exp.name:25s}: val={vm:.4f}  test={tm:.4f}  sil_drv={sil_d:.4f}  sil_lbl={sil_l:.4f}")
    return metrics


# ── Plots ────────────────────────────────────────────────────────────────────

def plot_curves(metrics, out_dir):
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    for ax, m in zip(axes[0], metrics):
        h, exp = m["hist"], m["exp"]
        ax.plot(h["train_loss"], label="train")
        ax.plot(h["val_loss"],   label="val")
        if exp.model == "vae":
            ax.plot(h["val_recon"], "--", label="val_recon")
            ax.plot(h["val_kl"],    ":",  label="val_kl")
        ax.set_title(exp.name, fontsize=9)
        ax.legend(fontsize=7); ax.set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(out_dir / "curves.png", dpi=120); plt.close(fig)


def plot_recon_per_feature(metrics, features, out_dir):
    n_exp, D = len(metrics), len(features)
    x = np.arange(D)
    w = 0.8 / n_exp
    fig, axes = plt.subplots(1, 2, figsize=(max(10, D * 2), 4))
    for ax, key, title in [(axes[0], "fmse_val", "Val"), (axes[1], "fmse_test", "Test")]:
        for i, m in enumerate(metrics):
            ax.bar(x + i * w - 0.4 + w / 2, m[key], w, label=m["exp"].name)
        ax.set_xticks(x); ax.set_xticklabels(features, rotation=20, ha="right")
        ax.set_title(f"Per-Feature Recon MSE ({title})"); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "recon_per_feature.png", dpi=120); plt.close(fig)


def plot_tsne(metrics, driver_names, out_dir):
    n = len(metrics)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8), squeeze=False)
    for col, m in enumerate(metrics):
        Z2, d_all, y_all = m["Z2"], m["d_all"], m["y_all"]
        split = len(m["Z_va"])
        all_names = list(driver_names) + ["test"]
        for row, (labels, cmap, title) in enumerate([
            (d_all, "tab10", "by driver"),
            (y_all, "bwr",   "by label"),
        ]):
            ax = axes[row, col]
            sc = ax.scatter(Z2[:split, 0], Z2[:split, 1], c=labels[:split],
                            cmap=cmap, s=8, alpha=0.6, vmin=labels.min(), vmax=labels.max())
            ax.scatter(Z2[split:, 0], Z2[split:, 1], c=labels[split:],
                       cmap=cmap, s=8, alpha=0.9, marker="^", vmin=labels.min(), vmax=labels.max())
            ax.set_title(f"{m['exp'].name}\n{title}", fontsize=8)
            ax.axis("off")
            fig.colorbar(sc, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(out_dir / "tsne.png", dpi=120); plt.close(fig)


def plot_summary(metrics, out_dir):
    names = [m["exp"].name for m in metrics]
    val_mses  = [m["val_mse"]   for m in metrics]
    test_mses = [m["test_mse"]  for m in metrics]
    sil_drvs  = [m["sil_driver"] for m in metrics]
    sil_lbls  = [m["sil_label"]  for m in metrics]

    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(max(12, len(names) * 1.5), 4))

    axes[0].bar(x - 0.2, val_mses,  0.4, label="val")
    axes[0].bar(x + 0.2, test_mses, 0.4, label="test")
    axes[0].set_title("Reconstruction MSE"); axes[0].legend()

    axes[1].bar(x, sil_drvs)
    axes[1].axhline(0, color="gray", lw=0.8)
    axes[1].set_title("Silhouette by Driver")

    axes[2].bar(x, sil_lbls)
    axes[2].axhline(0, color="gray", lw=0.8)
    axes[2].set_title("Silhouette by Label (pos/neg)")

    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "summary.png", dpi=120); plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    cfg = Config()
    seed_all(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    print("Loading data...")
    X_tr, y_tr, d_tr, X_va, y_va, d_va, X_te, y_te = load_data(cfg)
    T, D = X_tr.shape[1], X_tr.shape[2]
    print(f"T={T}, D={D}  train={len(X_tr)}  val={len(X_va)}  test={len(X_te)}")

    results = []
    for exp in cfg.experiments:
        print(f"\nTraining [{exp.name}]  model={exp.model}  z={exp.latent_dim} ...")
        model, hist = train_one(exp, X_tr, X_va, T, D, device)
        results.append((exp, model, hist))

    print("\n=== Metrics ===")
    metrics = compute_all_metrics(results, X_va, y_va, d_va, X_te, y_te, T, D, device, len(cfg.train_drivers))

    print("\nSaving plots...")
    plot_curves(metrics, out_dir)
    plot_recon_per_feature(metrics, cfg.view.cols, out_dir)
    plot_tsne(metrics, cfg.train_drivers, out_dir)
    plot_summary(metrics, out_dir)
    print(f"Done → {out_dir}")


if __name__ == "__main__":
    main()
