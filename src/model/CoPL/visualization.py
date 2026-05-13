from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, roc_curve

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

from ...plot_style import STYLE


def plot_roc(y_true, y_score, save_path: Path, title: str):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.0
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_reward_scatter(y_true, y_prob, save_path: Path, title: str):
    idx = np.arange(len(y_true))
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    m0 = (y_true == 0)
    m1 = (y_true == 1)
    plt.figure(figsize=(10, 5))
    plt.scatter(idx[m0], y_prob[m0], s=10, alpha=0.5, label="Bad(0)")
    plt.scatter(idx[m1], y_prob[m1], s=10, alpha=0.7, label="Good(1)")
    plt.axhline(0.5, linestyle="--", alpha=0.5)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Sample index")
    plt.ylabel("Predicted probability")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def compare_viz_plot(cfg, emb: np.ndarray, labels: list, save_path: Path, title: str, star_mask=None):
    uniq = sorted(set(labels))
    n_samples = emb.shape[0]
    perp = min(cfg.tsne_perplexity, max(5, (n_samples - 1) // 3))
    z_tsne = TSNE(n_components=2, random_state=cfg.seed, perplexity=perp).fit_transform(emb)
    z_pca = PCA(n_components=2, random_state=cfg.seed).fit_transform(emb)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax_idx, (name, z) in enumerate([("t-SNE", z_tsne), ("PCA", z_pca)]):
        ax = axes[ax_idx]
        for l in uniq:
            m = np.array([x == l for x in labels])
            if star_mask is not None and m.any() and star_mask[m].all():
                ax.scatter(z[m, 0], z[m, 1], marker="*", s=120, alpha=1.0, label=l, edgecolors="black")
            else:
                ax.scatter(z[m, 0], z[m, 1], s=20, alpha=0.6, label=l)
        ax.set_title(f"{title}\n[{name}]", fontsize=STYLE["diag"]["title_fs"])
        ax.grid(True, alpha=0.3)
        if ax_idx == 1:
            ax.legend(loc="best", fontsize=STYLE["diag"]["legend_fs"])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_distance_gamma_analysis(Z_train, gamma, save_path):
    plt.figure(figsize=(10, 5))
    if gamma is None:
        from sklearn.metrics.pairwise import cosine_similarity
        N = Z_train.shape[0]
        if N > 2000:
            rng = np.random.default_rng(42)
            Z_sub = Z_train[rng.choice(N, 2000, replace=False)]
        else:
            Z_sub = Z_train
        sims = cosine_similarity(Z_sub)
        sim_vals = sims[np.triu_indices_from(sims, k=1)]
        plt.hist(sim_vals, bins=50, alpha=0.7, color='blue', label='Cosine Similarity')
        plt.title("Latent Space Cosine Similarity Distribution")
        plt.xlabel("Cosine Similarity")
        plt.xlim(-1.0, 1.0)
    else:
        from scipy.spatial.distance import pdist
        dists = pdist(Z_train, metric="euclidean")
        plt.hist(dists, bins=50, alpha=0.7, color='gray', label='Pairwise Distances')
        ax2 = plt.gca().twinx()
        d_range = np.linspace(0, dists.max(), 100)
        ax2.plot(d_range, np.exp(-gamma * (d_range ** 2)), color='red', linewidth=2,
                 label=f'RBF Kernel (gamma={gamma:.4f})')
        ax2.set_ylabel('Similarity Weight')
        plt.title("Latent Space Distance Distribution vs. RBF Kernel")
        plt.xlabel("Euclidean Distance")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_driver_similarity_matrix(Aii_norm, item_owner_uid, train_drivers, save_path):
    adj = Aii_norm.to_dense().cpu().numpy()
    n_users = len(train_drivers)
    sim_matrix = np.zeros((n_users, n_users))
    for i in range(n_users):
        mask_i = (item_owner_uid == i)
        n_items_i = np.sum(mask_i)
        if n_items_i == 0:
            continue
        for j in range(n_users):
            mask_j = (item_owner_uid == j)
            if not np.any(mask_j):
                continue
            sim_matrix[i, j] = adj[mask_i][:, mask_j].sum() / n_items_i

    plt.figure(figsize=(8, 6))
    import seaborn as sns
    sns.heatmap(sim_matrix, xticklabels=train_drivers, yticklabels=train_drivers,
                annot=True, fmt=".4f", cmap="YlGnBu")
    plt.title("Item-Item Connectivity (Normalized by Source Item Count)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_vae_reconstruction(vae, item_series, feature_names, mu_stats, sd_stats, save_path, n_samples=6, seed=42):
    N, T, D = item_series.shape
    n = min(n_samples, N)
    idx = np.random.default_rng(seed).choice(N, n, replace=False)

    Xs_flat = (item_series[idx].reshape(n, T * D).astype(np.float32) - mu_stats) / sd_stats
    X_tensor = torch.tensor(Xs_flat.reshape(n, T, D).transpose(0, 2, 1))

    device = next(vae.parameters()).device
    vae.eval()
    with torch.no_grad():
        x_hat = vae(X_tensor.to(device))[0].cpu()

    x_hat_orig = (x_hat.permute(0, 2, 1).reshape(n, T * D).numpy() * sd_stats + mu_stats).reshape(n, T, D)
    x_orig = item_series[idx]

    fig, axes = plt.subplots(D, n, figsize=(3 * n, 2.5 * D))
    axes = np.array(axes).reshape(D, n)
    for d in range(D):
        for s in range(n):
            ax = axes[d, s]
            ax.plot(x_orig[s, :, d], color='gray', lw=1.2)
            ax.plot(x_hat_orig[s, :, d], color='red', lw=1, ls='--')
            if d == 0:
                ax.set_title(f'#{idx[s]}', fontsize=STYLE["diag"]["title_fs"])
            if s == 0:
                ax.set_ylabel(feature_names[d], fontsize=STYLE["diag"]["label_fs"])
            ax.tick_params(labelsize=STYLE["diag"]["tick_fs"])

    handles = [plt.Line2D([0], [0], color='gray', lw=STYLE["diag"]["lw"], label='Original'),
               plt.Line2D([0], [0], color='red', lw=STYLE["diag"]["lw"], ls='--', label='Reconstructed')]
    fig.legend(handles=handles, loc='upper right', fontsize=STYLE["diag"]["legend_fs"])
    plt.suptitle('VAE Reconstruction', fontsize=STYLE["diag"]["title_fs"])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _ae_paper_data(ae, item_series, mu_stats, sd_stats, seed=42):
    N, T, D = item_series.shape
    i = np.random.default_rng(seed).integers(N)
    Xs_flat = (item_series[[i]].reshape(1, T * D).astype(np.float32) - mu_stats) / sd_stats
    X_tensor = torch.tensor(Xs_flat.reshape(1, T, D).transpose(0, 2, 1))
    device = next(ae.parameters()).device
    ae.eval()
    with torch.no_grad():
        x_hat = ae(X_tensor.to(device))[0].cpu()
    x_orig = item_series[i]
    x_rec = (x_hat.permute(0, 2, 1).reshape(1, T * D).numpy() * sd_stats + mu_stats).reshape(T, D)
    return x_orig, x_rec


def plot_ae_orig_paper(ae, item_series, feature_names, mu_stats, sd_stats, save_path, seed=42):
    S = STYLE["paper"]
    x_orig, _ = _ae_paper_data(ae, item_series, mu_stats, sd_stats, seed)
    T, D = x_orig.shape
    fig, axes = plt.subplots(D, 1, figsize=(4 * 1.2, 2 * D))
    for d, ax in enumerate(axes):
        ax.plot(x_orig[:, d], lw=S["lw"], color='gray')
        ax.set_ylabel(feature_names[d], fontsize=S["label_fs"])
        ax.tick_params(axis='y', labelsize=S["tick_fs"])
        ax.set_xticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_ae_recon_paper(ae, item_series, feature_names, mu_stats, sd_stats, save_path, seed=42):
    S = STYLE["paper"]
    x_orig, x_rec = _ae_paper_data(ae, item_series, mu_stats, sd_stats, seed)
    T, D = x_orig.shape
    fig, axes = plt.subplots(D, 1, figsize=(4 * 1.2, 2 * D))
    for d, ax in enumerate(axes):
        ax.plot(x_orig[:, d], lw=S["lw"], color='gray')
        ax.plot(x_rec[:, d], lw=S["lw"], color='red', ls='--')
        ax.set_ylabel(feature_names[d], fontsize=S["label_fs"])
        ax.tick_params(axis='y', labelsize=S["tick_fs"])
        ax.set_xticks([])
    handles = [plt.Line2D([0], [0], color='gray', lw=S["lw"], label='Original'),
               plt.Line2D([0], [0], color='red', lw=S["lw"], ls='--', label='Reconstructed')]
    fig.legend(handles=handles, loc='upper right', fontsize=S["legend_fs"])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_vae_latent(Z_train, item_owner_uid, train_drivers, save_path, seed=42):
    n = Z_train.shape[0]
    perp = min(30, max(5, (n - 1) // 3))
    z_tsne = TSNE(n_components=2, random_state=seed, perplexity=perp).fit_transform(Z_train)
    z_pca = PCA(n_components=2).fit_transform(Z_train)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(train_drivers), 1)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, proj, title in [(axes[0], z_tsne, "t-SNE"), (axes[1], z_pca, "PCA")]:
        for uid, name in enumerate(train_drivers):
            m = item_owner_uid == uid
            ax.scatter(proj[m, 0], proj[m, 1], s=15, alpha=0.6, color=colors[uid], label=name)
        ax.set_title(f"VAE Latent Space [{title}]", fontsize=STYLE["diag"]["title_fs"])
        ax.legend(fontsize=STYLE["diag"]["legend_fs"])
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)



def plot_vae_feature_mse(vae, item_series, feature_names, mu_stats, sd_stats, save_path):
    N, T, D = item_series.shape
    Xs_flat = (item_series.reshape(N, T * D).astype(np.float32) - mu_stats) / sd_stats
    X_tensor = torch.tensor(Xs_flat.reshape(N, T, D).transpose(0, 2, 1))

    device = next(vae.parameters()).device
    vae.eval()
    with torch.no_grad():
        x_hat = vae(X_tensor.to(device))[0].cpu().numpy()

    orig = X_tensor.numpy()
    mse_per_feature = ((orig - x_hat) ** 2).mean(axis=(0, 2))

    plt.figure(figsize=(max(6, D * 1.5), 4))
    plt.bar(feature_names, mse_per_feature, color='steelblue')
    plt.xticks(rotation=30, ha='right')
    plt.ylabel('MSE (standardized space)')
    plt.title('VAE Reconstruction MSE per Feature')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_item_embeddings(E_i, item_owner_uid, train_drivers, save_dir,
                         E_i_tests=None, test_driver_names=None):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    E_np = E_i.detach().cpu().numpy() if torch.is_tensor(E_i) else np.asarray(E_i)
    test_arrs, test_names = [], []
    for t, n_ in zip(E_i_tests or [], test_driver_names or []):
        if t is None:
            continue
        test_arrs.append(t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t))
        test_names.append(n_)

    E_all = np.concatenate([E_np] + test_arrs, axis=0) if test_arrs else E_np
    n = E_all.shape[0]
    perp = min(30, max(5, (n - 1) // 3))
    z_tsne = TSNE(n_components=2, random_state=42, perplexity=perp).fit_transform(E_all)
    z_pca = PCA(n_components=2).fit_transform(E_all)
    n_train = E_np.shape[0]

    colors = [plt.cm.tab10(i) for i in range(len(train_drivers))]
    test_markers = ["*", "X", "P", "D"]
    test_colors = ["black", "red", "green", "purple"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, proj, title in [(axes[0], z_tsne, "t-SNE"), (axes[1], z_pca, "PCA")]:
        for uid, name in enumerate(train_drivers):
            m = item_owner_uid == uid
            ax.scatter(proj[:n_train][m, 0], proj[:n_train][m, 1], s=18, alpha=0.75, color=colors[uid], label=name)
        offset = n_train
        for ti, t_arr in enumerate(test_arrs):
            t_n = t_arr.shape[0]
            ax.scatter(proj[offset:offset+t_n, 0], proj[offset:offset+t_n, 1],
                       s=40, alpha=0.9, color=test_colors[ti % len(test_colors)],
                       marker=test_markers[ti % len(test_markers)],
                       label=test_names[ti], zorder=5)
            offset += t_n
        ax.set_title(f"Item Embeddings [{title}]", fontsize=STYLE["diag"]["title_fs"])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=STYLE["diag"]["legend_fs"])
    fig.tight_layout()
    fig.savefig(save_dir / "item_emb.png", dpi=150)
    plt.close(fig)


def plot_user_embeddings(E_u, train_drivers, e_u_tests, test_driver_names, save_dir):
    """Train user cosine sim heatmap + PCA with adapted test users."""
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    E_np = E_u.detach().cpu().numpy() if torch.is_tensor(E_u) else np.asarray(E_u)
    n_u = len(train_drivers)

    sim = cos_sim(E_np)
    fig, ax = plt.subplots(figsize=(max(6, n_u * 0.9), max(5, n_u * 0.8)))
    im = ax.imshow(sim, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n_u)); ax.set_yticks(range(n_u))
    ax.set_xticklabels(train_drivers, rotation=45, ha="right", fontsize=STYLE["diag"]["annot_fs"])
    ax.set_yticklabels(train_drivers, fontsize=STYLE["diag"]["annot_fs"])
    for i in range(n_u):
        for j in range(n_u):
            ax.text(j, i, f"{sim[i,j]:.2f}", ha="center", va="center", fontsize=STYLE["diag"]["annot_fs"])
    ax.set_title("User Embedding Cosine Similarity")
    fig.tight_layout()
    fig.savefig(save_dir / "user_sim.png", dpi=150)
    plt.close(fig)

    test_arrs, test_names = [], []
    for e, n_ in zip(e_u_tests or [], test_driver_names or []):
        if e is None:
            continue
        arr = e.detach().cpu().numpy() if torch.is_tensor(e) else np.asarray(e)
        test_arrs.append(arr.reshape(1, -1))
        test_names.append(n_)
    if not test_arrs:
        return
    Z_all = np.concatenate([E_np] + test_arrs, axis=0)
    names = list(train_drivers) + test_names
    z_pca = PCA(n_components=2).fit_transform(Z_all)

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(names), 1)))
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, name in enumerate(names):
        is_test = name in test_names
        ax.scatter(z_pca[i, 0], z_pca[i, 1],
                   color=colors[i], s=200 if is_test else 100,
                   marker="*" if is_test else "o",
                   edgecolors="black", lw=0.8, label=name,
                   zorder=5 if is_test else 1)
        ax.annotate(name, (z_pca[i, 0], z_pca[i, 1]),
                    textcoords="offset points", xytext=(5, 5), fontsize=STYLE["diag"]["annot_fs"])
    ax.set_title("User Embeddings (PCA)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "user_emb_with_test.png", dpi=150)
    plt.close(fig)


def plot_rm_distributions(rm, E_u, item_series, item_owner_uid, item_labels, train_drivers, device, save_dir):
    """RM score histogram per driver (label=0 vs 1)."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    rm.eval()
    for uid, name in enumerate(train_drivers):
        mask = item_owner_uid == uid
        if not mask.any():
            continue
        X_u = torch.tensor(item_series[mask], dtype=torch.float32).to(device)
        e_u = E_u[uid].unsqueeze(0).expand(X_u.shape[0], -1)
        with torch.no_grad():
            scores = torch.sigmoid(rm(e_u, X_u)).cpu().numpy().flatten()
        y_u = item_labels[mask]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(scores[y_u == 0], bins=20, alpha=0.6, color="steelblue", label="Bad(0)", density=True)
        ax.hist(scores[y_u == 1], bins=20, alpha=0.6, color="tomato", label="Good(1)", density=True)
        ax.set_xlabel("RM Score P(good)")
        ax.set_ylabel("Density")
        ax.set_title(f"RM Score Distribution — {name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_dir / f"rm_dist_{name}.png", dpi=150)
        plt.close(fig)


def plot_wu_evolution(wu_histories, ctx_sizes_list, train_driver_names, test_driver_names, save_path):
    """test driver별로 w_u 변화 한 행씩."""
    n = len(test_driver_names)
    if n == 0 or not any(wu_histories):
        return
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(train_driver_names), 1)))
    fig, axes = plt.subplots(n, 1, figsize=(10, 4 * n))
    if n == 1:
        axes = [axes]
    for ax, wu_history, ctx_sizes, test_name in zip(axes, wu_histories, ctx_sizes_list, test_driver_names):
        if not wu_history:
            ax.set_visible(False)
            continue
        traj = np.stack(wu_history)
        for uid, name in enumerate(train_driver_names):
            ax.plot(ctx_sizes, traj[:, uid], lw=1.5, color=colors[uid], label=name)
        ax.set_xlabel("Context Size (t)")
        ax.set_ylabel("Attention Weight (w_u)")
        ax.set_title(f"User Attention Weight Evolution — {test_name}")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=STYLE["diag"]["legend_fs"])
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_attention_bars(w_us, train_drivers, test_driver_names, save_path, suffix=""):
    n = len(test_driver_names)
    S = STYLE["diag"]
    fig, axes = plt.subplots(n, 1, figsize=(8, 4 * n))
    axes = np.atleast_1d(axes)
    for ax, w_u, name in zip(axes, w_us, test_driver_names):
        ax.bar(train_drivers, w_u, color="mediumseagreen")
        ax.set_title(f"User Attention (w_u) — {name}{suffix}", fontsize=S["title_fs"])
        ax.set_ylabel("Weight", fontsize=S["label_fs"])
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=100)
    plt.close(fig)


def plot_ablation_bars(abl_names_list, abl_vals_list, test_driver_names, save_path, title, xtick_fn=None):
    n = len(test_driver_names)
    S = STYLE["diag"]
    fig, axes = plt.subplots(n, 1, figsize=(9, 4.8 * n))
    axes = np.atleast_1d(axes)
    colors = ["steelblue", "mediumpurple", "seagreen", "salmon"]
    for ax, abl_names, vals, name in zip(axes, abl_names_list, abl_vals_list, test_driver_names):
        x = np.arange(len(abl_names))
        bars = ax.bar(x, vals, color=colors[:len(abl_names)])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=S["annot_fs"]+2)
        ax.set_title(f"{title} — {name}", fontsize=S["title_fs"])
        ax.set_ylabel("AUROC", fontsize=S["label_fs"])
        ax.set_xticks(x, xtick_fn(abl_names) if xtick_fn else abl_names)
        ax.tick_params(axis="x", labelsize=S["tick_fs"] + 3)
        ax.tick_params(axis="y", labelsize=S["tick_fs"])
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=100)
    plt.close(fig)


def plot_test_item_bridge(neigh_idxs, neigh_ws, item_owner_uid, train_drivers, test_driver_names, save_path):
    n = len(test_driver_names)
    user_item_counts = np.array([np.sum(item_owner_uid == i) for i in range(len(train_drivers))])
    fig, axes = plt.subplots(n, 1, figsize=(10, 5 * n))
    if n == 1:
        axes = [axes]
    for ax, neigh_idx, neigh_w, name in zip(axes, neigh_idxs, neigh_ws, test_driver_names):
        if neigh_idx is None:
            ax.set_visible(False)
            continue
        driver_contribution = np.zeros(len(train_drivers))
        for i in range(neigh_idx.shape[0]):
            owners = item_owner_uid[neigh_idx[i]]
            for owner, w in zip(owners, neigh_w[i]):
                driver_contribution[owner] += w
        driver_contribution = driver_contribution / (user_item_counts + 1e-8)
        driver_contribution /= (driver_contribution.sum() + 1e-8)
        ax.bar(train_drivers, driver_contribution, color='skyblue')
        ax.set_title(f"Bridge Contribution — {name}")
        ax.set_ylabel("Relative Contribution Per Item")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
