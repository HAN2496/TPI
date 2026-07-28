import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from core.plots import STYLE
from .model import sigmoid
from .utils import ensure_dir

FS = 3
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({
    "axes.titlesize": 14 + FS, "axes.labelsize": 10 + FS,
    "xtick.labelsize": 8 + FS, "ytick.labelsize": 8 + FS,
})


def top_feature_indices(model, pipe, top_k=18):
    keep = np.array([g != "bias" for g in pipe.groups])
    score = np.where(keep, model.theta_means.std(axis=0), -np.inf)
    return np.argsort(score)[::-1][:top_k][::-1]


def fit_theta_pca(model):
    pca = PCA(n_components=2, random_state=0)
    pca.fit(model.theta_samples.reshape(-1, model.d))
    return pca


def group_contributions_with_cov(pipe, theta, cov_diag, Phi, top_k=10000):
    rows = pipe.group_contributions(theta, Phi, cov=np.diag(cov_diag), top_k=top_k)
    return {name: (signed, unc) for name, signed, _, unc in rows}


def plot_gibbs_trace(model, save_path):
    tr = model.trace
    n_burn = model.n_burnin
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))

    ax = axes[0, 0]
    ax.plot(tr["loglik"], color="steelblue", lw=0.8)
    ax.axvline(n_burn, color="crimson", ls="--", lw=0.8, label=f"burn-in ({n_burn})")
    ax.set_title("Joint log-likelihood  sum_u log p(y_u | theta_u)")
    ax.set_xlabel("Gibbs iteration"); ax.set_ylabel("log-likelihood")
    ax.legend(fontsize=8 + FS); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(tr["mu_norm"], color="darkorange", lw=0.8)
    ax.axvline(n_burn, color="crimson", ls="--", lw=0.8)
    ax.set_title("||mu^(t)||  (population mean magnitude)")
    ax.set_xlabel("Gibbs iteration"); ax.set_ylabel("||mu||")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(tr["sigma_trace"], color="mediumseagreen", lw=0.8)
    ax.axvline(n_burn, color="crimson", ls="--", lw=0.8)
    ax.set_title("tr(Sigma^(t))  (total between-user variance)")
    ax.set_xlabel("Gibbs iteration"); ax.set_ylabel("tr(Sigma)")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    for u, name in enumerate(model.user_names):
        ax.plot(tr["theta_norm"][:, u], lw=0.7, label=name)
    ax.axvline(n_burn, color="crimson", ls="--", lw=0.8)
    ax.set_title("||theta_u^(t)||  (per-user weight magnitude)")
    ax.set_xlabel("Gibbs iteration"); ax.set_ylabel("||theta_u||")
    ax.legend(fontsize=7 + FS, ncol=2); ax.grid(alpha=0.3)

    fig.suptitle("Gibbs chain diagnostics", fontsize=12 + FS)
    fig.tight_layout()
    ensure_dir(save_path)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_spike_slab_inclusion(model, save_path):
    lab_fs, tick_fs, ytick_fs = 10 + FS + 2.5, 8 + FS + 2.5, 7 + FS + 2.5
    order = np.argsort(model.gamma_unit_pip)
    labels = [model.gamma_unit_names[i] for i in order]
    pips = model.gamma_unit_pip[order]
    height = max(5, 0.35 * len(labels) + 2)
    fig, axes = plt.subplots(1, 2, figsize=(15, height))

    ax = axes[0]
    iterations = np.arange(len(model.trace["gamma_count"]))
    ax.plot(iterations, model.trace["gamma_count"], color="steelblue", lw=0.8,
            label="included units")
    ax.axvline(model.n_burnin, color="crimson", ls="--", lw=0.8,
               label=f"burn-in ({model.n_burnin})")
    ax.set_xlabel("Gibbs iteration", fontsize=lab_fs)
    ax.set_ylabel("number of included units", color="steelblue", fontsize=lab_fs)
    ax.tick_params(axis="x", labelsize=tick_fs)
    ax.tick_params(axis="y", labelcolor="steelblue", labelsize=tick_fs)
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(iterations, model.trace["pi"], color="darkorange", lw=0.7,
             alpha=0.8, label="pi")
    ax2.set_ylabel("inclusion probability pi", color="darkorange", fontsize=lab_fs)
    ax2.tick_params(axis="y", labelcolor="darkorange", labelsize=tick_fs)
    ax2.set_ylim(0, 1)
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, fontsize=8 + FS)
    ax.set_title("Spike-and-slab chain")

    ax = axes[1]
    colors = ["seagreen" if pip >= 0.5 else "lightgray" for pip in pips]
    ax.barh(np.arange(len(labels)), pips, color=colors, alpha=0.9)
    ax.axvline(0.5, color="crimson", ls="--", lw=1.0,
               label="PIP = 0.5")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=ytick_fs)
    ax.tick_params(axis="x", labelsize=tick_fs)
    ax.set_xlim(0, 1)
    ax.set_xlabel("posterior inclusion probability", fontsize=lab_fs)
    ax.set_title(f"{model.spike_slab_unit}-level posterior inclusion")
    ax.legend(fontsize=8 + FS)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    ensure_dir(save_path)
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_population_weight_forest(model, pipe, save_path, top_k=18):
    top_idx = top_feature_indices(model, pipe, top_k)
    labels = [model.feature_names[j] for j in top_idx]
    mu = model.mu_samples[:, top_idx]
    mu_med = np.median(mu, axis=0)
    mu_lo = np.percentile(mu, 2.5, axis=0)
    mu_hi = np.percentile(mu, 97.5, axis=0)
    theta_user = model.theta_means[:, top_idx]
    K = len(top_idx)
    y_base = np.arange(K)
    fig, ax = plt.subplots(figsize=(11, max(7, K * 0.45)))

    ax.errorbar(mu_med, y_base, xerr=[mu_med - mu_lo, mu_hi - mu_med],
                fmt="s", color="black", ms=7, capsize=4, lw=1.8,
                label="population mu  (median, 95% CI)")
    cmap = plt.get_cmap("tab10")
    for u, name in enumerate(model.user_names):
        offset = (u - (model.U - 1) / 2) * 0.16
        ax.scatter(theta_user[u], y_base + offset, s=28, color=cmap(u % 10),
                   label=name, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="black", lw=0.7, ls="--", alpha=0.6)
    ax.set_yticks(y_base); ax.set_yticklabels(labels, fontsize=8 + FS)
    ax.set_xlabel("theta value  (mu: 95% CI; users: posterior mean theta_u)")
    ax.set_title("Population posterior:  mu  vs  per-user theta_u  (top features by across-user std)")
    ax.legend(loc="best", fontsize=8 + FS, ncol=2, framealpha=0.85)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    ensure_dir(save_path)
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_population_covariance(model, pipe, save_path):
    Sigma = model.Sigma_bar
    std = np.sqrt(np.maximum(np.diag(Sigma), 1e-12))
    corr = Sigma / np.outer(std, std)
    keep = np.array([g != "bias" for g in pipe.groups])
    idx = np.where(keep)[0]
    order = idx[np.argsort(-std[idx])[:30]] if len(idx) > 30 else idx
    corr_sub = corr[np.ix_(order, order)]
    labels = [model.feature_names[j] for j in order]
    K = len(order)

    fig, axes = plt.subplots(1, 2, figsize=(max(11, K * 0.6), max(6, K * 0.35)))
    ax = axes[0]
    im = ax.imshow(corr_sub, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(K)); ax.set_yticks(np.arange(K))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7 + FS)
    ax.set_yticklabels(labels, fontsize=7 + FS)
    ax.set_title("corr(Sigma_bar)  - which preference dimensions co-vary across users")

    ax = axes[1]
    ax.barh(np.arange(K), std[order], color="steelblue", alpha=0.8)
    ax.set_yticks(np.arange(K)); ax.set_yticklabels(labels, fontsize=7 + FS)
    ax.invert_yaxis()
    ax.set_xlabel("sqrt(diag(Sigma_bar))  - per-feature across-user std")
    ax.set_title("Between-user variability")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    ensure_dir(save_path)
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_population_eigenspectrum(model, save_path):
    eig = np.sort(np.linalg.eigvalsh(model.Sigma_bar))[::-1]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(len(eig)), eig, color="mediumseagreen", alpha=0.8)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("eigenvalue index"); ax.set_ylabel("eigenvalue (log)")
    ax.set_title("Sigma_bar eigenvalue spectrum  (variation modes across users)")
    ax.grid(axis="y", alpha=0.3, which="both")
    fig.tight_layout()
    ensure_dir(save_path)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_user_similarity(model, pipe, train_X, save_path, test_thetas=None):
    train_names = list(model.user_names)
    if len(train_names) <= 1:
        return
    train_theta = model.theta_means
    test_thetas = test_thetas or []
    test_names = [n for n, _ in test_thetas]
    names = train_names + test_names
    theta = np.vstack([train_theta] + [np.asarray(t).reshape(1, -1) for _, t in test_thetas]) \
            if test_thetas else train_theta
    n_train = len(train_names)
    n = len(names)
    mu = model.mu_bar

    Phi_pool = np.vstack([pipe.transform(train_X[nm]) for nm in train_names])
    probs = sigmoid(Phi_pool @ theta.T)
    p_c = probs - probs.mean(axis=0, keepdims=True)
    p_n = p_c / np.maximum(np.linalg.norm(p_c, axis=0, keepdims=True), 1e-12)
    func_sim = p_n.T @ p_n

    theta_n = theta / np.maximum(np.linalg.norm(theta, axis=1, keepdims=True), 1e-12)
    cos_sim = theta_n @ theta_n.T

    diff = theta - mu[None, :]
    euclid = np.linalg.norm(diff, axis=1)
    prec = np.linalg.inv(model.Sigma_bar + 1e-6 * np.eye(theta.shape[1]))
    mahal = np.sqrt(np.einsum("md,de,me->m", diff, prec, diff))

    fig, axes = plt.subplots(2, 2, figsize=(max(13, n * 1.7), max(10, n * 1.4)))
    mask_off = ~np.eye(n, dtype=bool)
    func_off, cos_off = func_sim[mask_off], cos_sim[mask_off]
    for ax, mat, title, vmin, vmax, cmap in [
        (axes[0, 0], func_sim,
         f"Functional agreement: corr(sigma(Phi theta_i), sigma(Phi theta_j))\n[off-diag {func_off.min():.3f}~{func_off.max():.3f}, |max|={np.abs(func_off).max():.3f}]",
         -np.abs(func_off).max(), np.abs(func_off).max(), "RdBu_r"),
        (axes[0, 1], cos_sim,
         f"Raw cosine sim(theta_i, theta_j) [reference]\n[off-diag {cos_off.min():.3f}~{cos_off.max():.3f}]",
         cos_off.min(), cos_off.max(), "YlOrRd"),
    ]:
        mat_disp = np.where(mask_off, mat, np.nan)
        im = ax.imshow(mat_disp, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(np.arange(n)); ax.set_yticks(np.arange(n))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9 + FS)
        ax.set_yticklabels(names, fontsize=9 + FS)
        ax.set_title(title, fontsize=10 + FS)
        for i in range(n):
            for j in range(n):
                txt = "1.00" if i == j else f"{mat[i, j]:.3f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8 + FS)
        if test_thetas:
            ax.axhline(n_train - 0.5, color="black", lw=2.0)
            ax.axvline(n_train - 0.5, color="black", lw=2.0)
        for tick in ax.get_xticklabels()[n_train:]:
            tick.set_color("darkorange"); tick.set_fontweight("bold")
        for tick in ax.get_yticklabels()[n_train:]:
            tick.set_color("darkorange"); tick.set_fontweight("bold")

    bar_colors = ["steelblue"] * n_train + ["darkorange"] * (n - n_train)
    for ax, vals, title, ylabel in [
        (axes[1, 0], mahal, "Mahalanobis: sqrt((theta-mu_bar)^T Sigma_bar^-1 (theta-mu_bar))", "between-user std units"),
        (axes[1, 1], euclid, "Euclidean: ||theta - mu_bar||", "raw distance"),
    ]:
        ax.bar(np.arange(n), vals, color=bar_colors, alpha=0.85)
        ax.set_xticks(np.arange(n))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9 + FS)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8 + FS)
        for tick in ax.get_xticklabels()[n_train:]:
            tick.set_color("darkorange"); tick.set_fontweight("bold")

    suptitle = "Per-user posterior + test (peak theta): similarity (top) / deviation from mu_bar (bottom)" \
               if test_thetas else "Per-user posterior: similarity (top) / deviation from mu_bar (bottom)"
    fig.suptitle(suptitle, fontsize=12 + FS)
    fig.tight_layout()
    ensure_dir(save_path)
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_train_theta_pca(model, pca, save_path):
    flat = model.theta_samples
    M, U, _ = flat.shape
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))

    ax = axes[0]
    for u, name in enumerate(model.user_names):
        z = pca.transform(flat[:, u, :])
        ax.scatter(z[:, 0], z[:, 1], s=4, alpha=0.18, color=cmap(u % 10))
        zm = z.mean(axis=0)
        ax.scatter(zm[0], zm[1], s=120, marker="*", color=cmap(u % 10),
                   edgecolor="black", linewidth=0.8, label=name, zorder=5)
    mu_z = pca.transform(model.mu_samples)
    ax.scatter(mu_z[:, 0], mu_z[:, 1], s=3, alpha=0.15, color="black", label="mu samples")
    mu_zm = mu_z.mean(axis=0)
    ax.scatter(mu_zm[0], mu_zm[1], s=160, marker="P", color="black", edgecolor="white",
               linewidth=0.9, label="mu_bar", zorder=6)
    ax.axhline(0, color="gray", lw=0.4); ax.axvline(0, color="gray", lw=0.4)
    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1  ({ev[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2  ({ev[1]*100:.1f}%)")
    ax.set_title("Per-user theta_u posterior samples in 2D PCA space (train)")
    ax.legend(fontsize=8 + FS, loc="best", framealpha=0.85)
    ax.grid(alpha=0.25)

    ax = axes[1]
    sub = max(1, M // 200)
    blocks = [flat[::sub, u, :] for u in range(U)] + [model.mu_samples[::sub]]
    counts = [b.shape[0] for b in blocks]
    n_pts = sum(counts)
    emb = TSNE(n_components=2, random_state=0, init="pca",
               perplexity=min(30, (n_pts - 1) // 3)).fit_transform(np.concatenate(blocks, axis=0))
    parts = np.split(emb, np.cumsum(counts)[:-1])
    for u, name in enumerate(model.user_names):
        z = parts[u]
        ax.scatter(z[:, 0], z[:, 1], s=6, alpha=0.3, color=cmap(u % 10))
        zm = z.mean(axis=0)
        ax.scatter(zm[0], zm[1], s=120, marker="*", color=cmap(u % 10),
                   edgecolor="black", linewidth=0.8, label=name, zorder=5)
    z_mu = parts[U]
    ax.scatter(z_mu[:, 0], z_mu[:, 1], s=4, alpha=0.2, color="black", label="mu samples")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.set_title(f"t-SNE of theta_u posterior samples  (sub={sub}, n={n_pts})")
    ax.legend(fontsize=8 + FS, loc="best", framealpha=0.85)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    ensure_dir(save_path)
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_test_theta_pca(model, pca, drivers, save_path):
    flat = model.theta_samples
    fig, ax = plt.subplots(figsize=(9, 7.5))
    cmap = plt.get_cmap("tab10")
    for u, name in enumerate(model.user_names):
        z = pca.transform(flat[:, u, :])
        ax.scatter(z[:, 0], z[:, 1], s=4, alpha=0.12, color=cmap(u % 10))
        zm = z.mean(axis=0)
        ax.scatter(zm[0], zm[1], s=110, marker="*", color=cmap(u % 10),
                   edgecolor="black", linewidth=0.7, label=name, zorder=5)
    test_cmap = plt.get_cmap("Dark2")
    test_markers = ["X", "P", "D", "s", "^", "v", "p", "h"]
    for k, d in enumerate(drivers):
        zm = pca.transform(d["peak_theta_mean"].reshape(1, -1))[0]
        ax.scatter(zm[0], zm[1], s=210, marker=test_markers[k % len(test_markers)],
                   color=test_cmap(k % test_cmap.N), edgecolor="yellow", linewidth=2.0,
                   label=f"{d['name']} (peak t={d['peak_t']})", zorder=7)
    mu_zm = pca.transform(model.mu_samples).mean(axis=0)
    ax.scatter(mu_zm[0], mu_zm[1], s=180, marker="P", color="black", edgecolor="white",
               linewidth=0.9, label="mu_bar", zorder=6)
    ax.axhline(0, color="gray", lw=0.4); ax.axvline(0, color="gray", lw=0.4)
    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1  ({ev[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2  ({ev[1]*100:.1f}%)")
    ax.set_title("theta posterior in PCA space:  train clouds + test peak-thetas")
    ax.legend(fontsize=8 + FS, loc="best", framealpha=0.85, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    ensure_dir(save_path)
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_theta_trajectory(model, pca, theta_traj, traj_ts, save_path, tname):
    flat = model.theta_samples
    fig, ax = plt.subplots(figsize=(9, 7.5))
    cmap = plt.get_cmap("tab10")
    for u, name in enumerate(model.user_names):
        z = pca.transform(flat[:, u, :])
        ax.scatter(z[:, 0], z[:, 1], s=4, alpha=0.10, color=cmap(u % 10))
        zm = z.mean(axis=0)
        ax.scatter(zm[0], zm[1], s=90, marker="*", color=cmap(u % 10),
                   edgecolor="black", linewidth=0.6, label=name, zorder=4)
    traj_z = pca.transform(np.stack(theta_traj, axis=0))
    T = traj_z.shape[0]
    norm = plt.Normalize(0, max(traj_ts[-1], 1))
    colors = plt.cm.plasma(norm(np.asarray(traj_ts)))
    ax.plot(traj_z[:, 0], traj_z[:, 1], color="black", lw=1.0, alpha=0.6, zorder=6)
    for k in range(T):
        ax.scatter(traj_z[k, 0], traj_z[k, 1], s=22, color=colors[k], zorder=7)
        ax.annotate(f"t={traj_ts[k]}", (traj_z[k, 0], traj_z[k, 1]),
                    fontsize=7 + FS, xytext=(4, 4), textcoords="offset points")
    ax.scatter(traj_z[0, 0], traj_z[0, 1], s=140, marker="o", color="white",
               edgecolor="black", linewidth=1.2, label=f"{tname} ctx=0 (prior)", zorder=8)
    ax.scatter(traj_z[-1, 0], traj_z[-1, 1], s=160, marker="X", color="red",
               edgecolor="black", linewidth=1.2, label=f"{tname} ctx={traj_ts[-1]}", zorder=8)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.02).set_label("context size t")
    ax.axhline(0, color="gray", lw=0.4); ax.axvline(0, color="gray", lw=0.4)
    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1  ({ev[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2  ({ev[1]*100:.1f}%)")
    ax.set_title(f"theta_*^bar trajectory for {tname}: prior -> personalized")
    ax.legend(fontsize=8 + FS, loc="best", framealpha=0.85, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    ensure_dir(save_path)
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_predictive_panel(y, probs_M, save_path, title=""):
    """probs_M [M, N] posterior samples of P(y=1). epi + ale = mean_p (1 - mean_p)."""
    mean_p = probs_M.mean(axis=0)
    epi_var = probs_M.var(axis=0, ddof=1)
    ale_var = (probs_M * (1.0 - probs_M)).mean(axis=0)
    order = np.argsort(mean_p)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    if title:
        fig.suptitle(title, fontsize=11 + FS)

    ax = axes[0, 0]
    x = np.arange(len(mean_p))
    mp, ys = mean_p[order], y[order]
    ax.plot(x, mp, color="black", lw=1.0, label="mean prob")
    ax.scatter(x[ys == 1], mp[ys == 1], s=12, c="red", alpha=0.75, label="pos")
    ax.scatter(x[ys == 0], mp[ys == 0], s=12, c="blue", alpha=0.75, label="neg")
    ax.axhline(0.5, ls="--", color="k", lw=0.7)
    ax.set_xlabel("sample (sorted by mean prob)"); ax.set_ylabel("predicted prob")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Posterior predictive  (mean only)")
    ax.legend(fontsize=7 + FS, loc="upper left"); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    bins = np.linspace(0, 1, 11)
    bin_idx = np.clip(np.digitize(mean_p, bins) - 1, 0, 9)
    emp = np.zeros(10); pred = np.zeros(10); cnt = np.zeros(10)
    for b in range(10):
        mask = bin_idx == b
        cnt[b] = mask.sum()
        if cnt[b] > 0:
            emp[b] = y[mask].mean()
            pred[b] = mean_p[mask].mean()
    mask = cnt > 0
    ax.plot([0, 1], [0, 1], ls="--", color="k", lw=0.7)
    ax.scatter(pred[mask], emp[mask], s=(cnt[mask] / max(cnt.max(), 1) * 200) + 25,
               color="steelblue", alpha=0.8, edgecolor="black", linewidth=0.5)
    for b in np.where(mask)[0]:
        ax.text(pred[b], emp[b], f"n={int(cnt[b])}", fontsize=7 + FS, ha="center", va="bottom")
    ece = float(np.sum(cnt[mask] * np.abs(emp[mask] - pred[mask])) / cnt.sum()) if mask.any() else float("nan")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("mean predicted prob"); ax.set_ylabel("empirical pos rate")
    ax.set_title(f"Calibration (ECE={ece:.3f})")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    xp = mean_p[order]
    ale_s, epi_s = ale_var[order], epi_var[order]
    ax.fill_between(xp, 0, ale_s, color="seagreen", alpha=0.65, label="aleatoric  E[p(1-p)]")
    ax.fill_between(xp, ale_s, ale_s + epi_s, color="crimson", alpha=0.65, label="epistemic  Var[p]")
    ax.plot(xp, ale_s + epi_s, color="black", lw=0.9, alpha=0.8, label="total variance")
    pad = 0.02 * (xp[-1] - xp[0] + 1e-9)
    ax.set_xlim(xp[0] - pad, xp[-1] + pad)
    ax.set_xlabel("predicted prob"); ax.set_ylabel("variance")
    ax.set_title("Per-sample variance decomposition")
    ax.legend(fontsize=8 + FS, loc="upper right"); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    correct = (mean_p > 0.5).astype(np.int64) == y
    epi_std = np.sqrt(epi_var); ale_std = np.sqrt(ale_var)
    hi_val = max(float(epi_std.max()), float(ale_std.max()), 1e-3)
    bins = np.linspace(0, hi_val, 24)
    ax.hist(epi_std, bins=bins, alpha=0.55, color="crimson", label="epistemic std")
    ax.hist(ale_std, bins=bins, alpha=0.55, color="seagreen", label="aleatoric std")
    ax.set_xlabel("std per sample"); ax.set_ylabel("count")
    ax.set_title(f"Uncertainty histogram  (correct {correct.sum()}/{len(y)})")
    ax.legend(fontsize=8 + FS); ax.grid(alpha=0.3)

    fig.tight_layout()
    ensure_dir(save_path)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_prior_posterior_contributions(pipe, model, Phi, mu_bar, peak_theta, peak_std,
                                       save_path, tname, top_k=16):
    prior = group_contributions_with_cov(pipe, mu_bar, np.diag(model.Sigma_bar), Phi)
    post = group_contributions_with_cov(pipe, peak_theta, peak_std ** 2, Phi)
    groups = sorted(set(prior) | set(post),
                    key=lambda g: max(abs(prior.get(g, (0.0, None))[0]), abs(post.get(g, (0.0, None))[0])),
                    reverse=True)[:top_k]
    if not groups:
        return
    y = np.arange(len(groups))
    pv = np.array([prior.get(g, (0.0, 0.0))[0] for g in groups])
    po = np.array([post.get(g, (0.0, 0.0))[0] for g in groups])
    pv_err = np.array([prior.get(g, (0.0, 0.0))[1] or 0.0 for g in groups])
    po_err = np.array([post.get(g, (0.0, 0.0))[1] or 0.0 for g in groups])

    fig, ax = plt.subplots(figsize=(11, max(5.5, 0.4 * len(groups))))
    ax.barh(y + 0.2, pv, height=0.36, color="gray", alpha=0.65, xerr=pv_err,
            capsize=2, label="Prior (mu_bar)  +/-1 sigma")
    ax.barh(y - 0.2, po, height=0.36, color="mediumseagreen", alpha=0.85, xerr=po_err,
            capsize=2, label="Posterior (theta_* peak)  +/-1 sigma")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(groups, fontsize=8 + FS); ax.invert_yaxis()
    ax.set_xlabel("Average logit contribution")
    ax.set_title(f"{tname}:  prior vs personalized posterior reward contributions")
    ax.legend(fontsize=9 + FS); ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    ensure_dir(save_path)
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_weight_forest_with_test(model, pipe, peak_theta, peak_std, save_path, tname, top_k=18):
    top_idx = top_feature_indices(model, pipe, top_k)
    labels = [model.feature_names[j] for j in top_idx]
    train_theta = model.theta_means[:, top_idx]
    train_sigma = model.theta_stds[:, top_idx]
    K = len(top_idx)
    y_base = np.arange(K)
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(11, max(7, K * 0.45)))
    n_total = model.U + 1
    for u, name in enumerate(model.user_names):
        offset = (u - (n_total - 1) / 2) * 0.7 / max(n_total, 1)
        ax.errorbar(train_theta[u], y_base + offset, xerr=2 * train_sigma[u],
                    fmt="o", color=cmap(u % 10), ms=5, capsize=2,
                    elinewidth=1.0, alpha=0.85, label=name)
    offset = ((n_total - 1) - (n_total - 1) / 2) * 0.7 / max(n_total, 1)
    ax.errorbar(peak_theta[top_idx], y_base + offset, xerr=2 * peak_std[top_idx],
                fmt="D", color="black", ms=6, capsize=2.5,
                elinewidth=1.3, alpha=0.95, label=f"{tname} (peak)")
    ax.axvline(0, color="black", lw=0.7, ls="--", alpha=0.6)
    ax.set_yticks(y_base); ax.set_yticklabels(labels, fontsize=8 + FS)
    ax.set_xlabel("theta value  (point=posterior mean, bar=+/-2 sigma)")
    ax.set_title(f"Per-feature posterior:  train users + {tname} peak theta")
    ax.legend(loc="best", fontsize=8 + FS, ncol=2, framealpha=0.85)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    ensure_dir(save_path)
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_uncertainty_decay(ctx, mean_epi, mean_ale, prior_epi, prior_ale, save_path, tname):
    epi = np.asarray(mean_epi); ale = np.asarray(mean_ale)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(ctx, epi, color="crimson", lw=1.5, marker="o", ms=3, label="epistemic Var[p]")
    ax.plot(ctx, ale, color="seagreen", lw=1.5, marker="s", ms=3, label="aleatoric E[p(1-p)]")
    ax.plot(ctx, epi + ale, color="black", lw=1.0, ls="--", alpha=0.6, label="total")
    ax.axhline(prior_epi, color="crimson", ls=":", lw=0.9, alpha=0.6, label=f"prior epi mean = {prior_epi:.3f}")
    ax.axhline(prior_ale, color="seagreen", ls=":", lw=0.9, alpha=0.6, label=f"prior ale mean = {prior_ale:.3f}")
    ax.set_xlabel("context size t"); ax.set_ylabel("mean variance on holdout")
    ax.set_title(f"{tname}: predictive variance decomposition over context")
    ax.legend(fontsize=8 + FS); ax.grid(alpha=0.3)
    fig.tight_layout()
    ensure_dir(save_path)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_uncertainty_decay_overlay(drivers, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    cmap = plt.get_cmap("tab10")
    ax = axes[0]
    for i, d in enumerate(drivers):
        ax.plot(d["t_grid"], d["mean_epistemic"], color=cmap(i % 10), lw=1.4, label=d["name"], marker="o", ms=3)
    ax.set_xlabel("context size t"); ax.set_ylabel("mean epistemic Var[p]")
    ax.set_title("Epistemic uncertainty decay across test users")
    ax.legend(fontsize=8 + FS, ncol=2); ax.grid(alpha=0.3)

    ax = axes[1]
    for i, d in enumerate(drivers):
        ax.plot(d["t_grid"], d["mean_aleatoric"], color=cmap(i % 10), lw=1.4, label=d["name"], marker="s", ms=3)
    ax.set_xlabel("context size t"); ax.set_ylabel("mean aleatoric E[p(1-p)]")
    ax.set_title("Aleatoric uncertainty trend across test users")
    ax.legend(fontsize=8 + FS, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout()
    ensure_dir(save_path)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_trust_forest(drivers, save_path):
    rows = []
    for d in drivers:
        for label, key in [("prior", "trust_prior"), ("final", "trust_final"), ("peak", "trust_peak")]:
            rows.append((f"{d['name']}  ({label})", d[key]))
    n_rows = len(rows)
    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.35 * n_rows + 1)))
    for i, (name, t) in enumerate(rows):
        if np.isnan(t["mean"]):
            ax.scatter([0.5], [i], color="lightgray", s=40, marker="x")
            ax.text(0.52, i, f"N/A  ({t['reason']})  n={t['n']} pos={t['n_pos']} neg={t['n_neg']}",
                    va="center", fontsize=7 + FS, color="dimgray")
            continue
        color = "mediumseagreen" if t["trustworthy"] else "crimson"
        ax.errorbar([t["mean"]], [i], xerr=[[t["mean"] - t["ci_lo"]], [t["ci_hi"] - t["mean"]]],
                    fmt="o", color=color, ms=7, capsize=4, lw=2.0, alpha=0.9)
        verdict = "trust" if t["trustworthy"] else "REJECT"
        ax.text(1.01, i, f"{t['mean']:.3f}  [{t['ci_lo']:.3f}, {t['ci_hi']:.3f}]  w={t['width']:.3f}  [{verdict}]",
                va="center", fontsize=7 + FS, color="darkgreen" if t["trustworthy"] else "darkred",
                transform=ax.get_yaxis_transform())
    ax.axvline(0.5, color="black", ls="--", lw=0.8, alpha=0.6, label="random (0.5)")
    for j in range(3, n_rows, 3):
        ax.axhline(j - 0.5, color="gray", lw=0.3, alpha=0.5)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8 + FS)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("AUROC")
    ax.set_title("Trust intervals:  posterior x bootstrap 95% CI of AUROC\n"
                 "green = trustworthy (CI_lo > 0.5 AND CI_width < 0.15),  red = reject")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8 + FS)
    fig.tight_layout()
    ensure_dir(save_path)
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_train_user_predictions(model, plots_dir, Phis, ys, names):
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    y_kept, probs_kept, names_kept = [], [], []
    for name, Phi, y in zip(names, Phis, ys):
        if name not in model.user_names:
            continue
        theta = model.theta_means[model.user_names.index(name)]
        y_kept.append(y)
        probs_kept.append(sigmoid(Phi @ theta))
        names_kept.append(name)
    evaluate_predictions(y_kept, probs_kept, plots_dir, names_kept,
                         save_name="metrics", title="Posterior-mean train predictions")


def plot_train_predictive_panels(model, plots_dir, Phis, ys, names):
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    for name, Phi, y in zip(names, Phis, ys):
        if name not in model.user_names:
            continue
        theta_samples = model.theta_samples[:, model.user_names.index(name), :]
        probs_M = sigmoid(theta_samples @ Phi.T)
        plot_predictive_panel(y, probs_M, Path(plots_dir) / f"predictive_panel_{name}.png",
                              title=f"{name}  (train posterior, N={len(y)})")


def evaluate_predictions(y_trues, probs_list, plots_dir, names, save_name="metrics", title=""):
    n = len(names)
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    S = STYLE["diag"]
    fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n))
    axes = np.array(axes).reshape(n, 3)
    if title:
        fig.suptitle(title, fontsize=S["title_fs"])

    metrics = []
    for r, (y_true, probs, name) in enumerate(zip(y_trues, probs_list, names)):
        probs = np.asarray(probs).ravel()
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
        colors = ["red" if v == 1 else "blue" for v in y_true[sort_idx]]
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
        ax.hist(probs[y_true == 1], bins=bins, alpha=0.6, color="red", label=f"pos n={int((y_true == 1).sum())}")
        ax.set_xlabel("Predicted probability", fontsize=S["label_fs"])
        ax.set_ylabel("Count", fontsize=S["label_fs"])
        ax.set_title(f"Score Distribution (Brier={brier:.4f})", fontsize=S["title_fs"])
        ax.tick_params(labelsize=S["tick_fs"])
        ax.legend(fontsize=S["legend_fs"])

        metrics.append({"auroc": auroc, "auprc": auprc, "brier": brier})

    plt.tight_layout()
    fig.savefig(Path(plots_dir) / f"{save_name}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return metrics


def plot_lppd(drivers, out_path):
    """Log predictive density per point (higher = better). prior/final/peak: holdout n/2;
    prior_all/loo_all: all n (held-out refit: LOO 또는 K-fold). Error bars: se of mean."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    S = STYLE["diag"]
    names = [d["name"] for d in drivers] + ["POOLED"]
    ns = [f"{len(d['lpd_final'])}/{len(d['y_all'])}" for d in drivers]
    ns.append(f"{sum(len(d['lpd_final']) for d in drivers)}/{sum(len(d['y_all']) for d in drivers)}")
    colors = {"prior": "#b0b0b0", "final": "#1f77b4", "peak": "#2ca02c",
              "prior_all": "#636363", "loo_all": "#d62728"}
    keys = {"prior": "lpd_prior", "final": "lpd_final", "peak": "lpd_peak",
            "prior_all": "lpd_prior_all", "loo_all": "elpd_loo_i"}
    keys = {lab: kk for lab, kk in keys.items() if kk in drivers[0]}   # loo 생략 시 해당 시리즈 제외
    series, errs = {}, {}
    for lab, kk in keys.items():
        vecs = [d[kk] for d in drivers] + [np.concatenate([d[kk] for d in drivers])]
        series[lab] = [v.mean() for v in vecs]
        errs[lab] = [v.std(ddof=1) / np.sqrt(v.size) for v in vecs]

    x = np.arange(len(names))
    w = 0.16
    fig, ax = plt.subplots(figsize=(1.9 * len(names) + 3, 5))
    for i, lab in enumerate(keys):
        ax.bar(x + (i - (len(keys) - 1) / 2) * w, series[lab], w, yerr=errs[lab], capsize=2, label=lab, color=colors[lab])
    ax.axhline(np.log(0.5), color="crimson", lw=1.0, linestyle="--", label="uninformed (p=0.5)")
    ax.axvline(len(drivers) - 0.5, color="k", lw=0.8, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{nm}\nn={n}" for nm, n in zip(names, ns)], fontsize=S["tick_fs"])
    ax.set_ylabel("lppd / elpd_loo per point  (higher = better)", fontsize=S["label_fs"])
    ax.set_title("Log predictive density per test user  (holdout vs held-out refit all-data, +/- se)", fontsize=S["title_fs"])
    ax.tick_params(labelsize=S["tick_fs"])
    ax.legend(fontsize=S["legend_fs"])
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)




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
