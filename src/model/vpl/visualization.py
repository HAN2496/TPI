import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .utils import PreferenceDataset, convert_to_binary_context, compute_step_rewards

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def _dim_reduce(Z):
    n = Z.shape[0]
    perp = min(30, max(5, (n - 1) // 3))
    z_tsne = TSNE(n_components=2, random_state=42, perplexity=perp).fit_transform(Z)
    z_pca = PCA(n_components=2).fit_transform(Z)
    return z_tsne, z_pca


def collect_z_binary(model, train_driver_data, context_size, device):
    """각 train driver의 context들을 encode → dict[driver_name → (n_batches, latent_dim)]."""
    model.eval()
    z_by_driver = {}
    for name, (X, y) in train_driver_data.items():
        ctx = convert_to_binary_context(X, y, name, context_size, balanced=False)
        if ctx is None:
            continue
        loader = DataLoader(PreferenceDataset(ctx), batch_size=32, shuffle=False)
        means = []
        with torch.no_grad():
            for b in loader:
                m, _ = model.encode(b["observations"].to(device).float(),
                                    b["labels"].to(device).float())
                means.append(m.cpu().numpy())
        z_by_driver[name] = np.concatenate(means, axis=0)
    return z_by_driver


def collect_z_pairwise(model, train_dataset_dict, device):
    """Pairwise context들 encode → dict[driver_name → (n_batches, latent_dim)]."""
    model.eval()
    all_drivers = train_dataset_dict["driver_name"][:, 0]
    loader = DataLoader(PreferenceDataset(train_dataset_dict), batch_size=64, shuffle=False)
    all_means = []
    with torch.no_grad():
        for b in loader:
            m, _ = model.encode(
                b["observations"].to(device).float(),
                b["observations_2"].to(device).float(),
                b["labels"].to(device).float())
            all_means.append(m.cpu().numpy())
    all_means = np.concatenate(all_means, axis=0)
    return {name: all_means[all_drivers == name] for name in np.unique(all_drivers)}


def plot_z_space(z_by_driver, test_driver_names, save_path):
    """t-SNE + PCA scatter of latent z. Test driver → ★."""
    all_z, labels = [], []
    for name, z in z_by_driver.items():
        all_z.append(z)
        labels.extend([name] * len(z))
    Z = np.concatenate(all_z, axis=0)
    labels = np.array(labels)
    z_tsne, z_pca = _dim_reduce(Z)

    unique_names = list(z_by_driver.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_names), 1)))
    cmap = {n: colors[i] for i, n in enumerate(unique_names)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for proj, ax, title in [(z_tsne, axes[0], "t-SNE"), (z_pca, axes[1], "PCA")]:
        for name in unique_names:
            mask = labels == name
            test = name in test_driver_names
            ax.scatter(proj[mask, 0], proj[mask, 1],
                       color=cmap[name], s=80 if test else 20,
                       marker="*" if test else "o", alpha=1.0 if test else 0.6,
                       edgecolors="black" if test else "none",
                       label=name, zorder=5 if test else 1)
        ax.set_title(f"Latent z — {title}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_z_dims(z_by_driver, save_path):
    """Per-dim mean±std bar + dim 간 correlation heatmap."""
    all_z = np.concatenate(list(z_by_driver.values()), axis=0)
    means = all_z.mean(axis=0)
    stds = all_z.std(axis=0)
    D = len(means)
    corr = np.corrcoef(all_z.T) if D > 1 else np.array([[1.0]])

    fig, axes = plt.subplots(1, 2, figsize=(max(10, D * 0.8 + 6), 4))

    ax = axes[0]
    x = np.arange(D)
    ax.bar(x, means, color="steelblue", alpha=0.8)
    ax.errorbar(x, means, yerr=stds, fmt="none", color="black", capsize=3, lw=1)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Latent Dimension")
    ax.set_title("z dim mean ± std")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("z dim correlation")
    ax.set_xlabel("Dim")
    ax.set_ylabel("Dim")

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_z_evolution(z_trajectory, ctx_sizes, driver_name, save_path):
    """z_mean per dim이 context size에 따라 어떻게 수렴하는지."""
    if not z_trajectory:
        return
    traj = np.stack(z_trajectory)   # (n_steps, latent_dim)
    D = traj.shape[1]
    colors = plt.cm.viridis(np.linspace(0, 1, D))

    fig, ax = plt.subplots(figsize=(10, 4))
    for d in range(D):
        ax.plot(ctx_sizes, traj[:, d], lw=1.0, alpha=0.7, color=colors[d], label=f"dim {d}")
    ax.set_xlabel("Context Size (t)")
    ax.set_ylabel("z mean")
    ax.set_title(f"z Evolution — {driver_name}")
    ax.grid(True, alpha=0.3)
    if D <= 12:
        ax.legend(fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_reward_dist(model, driver_data, z_by_driver, device, save_dir):
    """Decoded reward 분포 (label=0 vs 1) per driver."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    for name, (X, y) in driver_data.items():
        if name not in z_by_driver:
            continue
        z_mean = z_by_driver[name].mean(axis=0)
        step_rewards = compute_step_rewards(model, X, z_mean, device)
        ep_rewards = step_rewards.mean(axis=1)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(ep_rewards[y == 0], bins=25, alpha=0.6, color="steelblue", label="Bad(0)", density=True)
        ax.hist(ep_rewards[y == 1], bins=25, alpha=0.6, color="tomato", label="Good(1)", density=True)
        ax.set_xlabel("Mean Episode Reward (logit)")
        ax.set_ylabel("Density")
        ax.set_title(f"Reward Distribution — {name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_dir / f"reward_dist_{name}.png", dpi=150)
        plt.close(fig)
