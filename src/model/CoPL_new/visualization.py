from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, roc_curve

from scripts.train_backbone import CFG

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

def compare_viz_plot(cfg: CFG, emb: np.ndarray, labels: list[str], save_path: Path, title: str, star_mask=None):
    # 1. 데이터 준비
    uniq = sorted(list(set(labels)))
    n_samples = emb.shape[0]

    # 2. t-SNE 계산
    perp = min(cfg.tsne_perplexity, max(5, (n_samples - 1) // 3))
    z_tsne = TSNE(n_components=2, random_state=cfg.seed, perplexity=perp).fit_transform(emb)

    # 3. PCA 계산 (신규 추가)
    z_pca = PCA(n_components=2, random_state=cfg.seed).fit_transform(emb)

    # 4. 시각화 (1행 2열)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    methods = [("t-SNE", z_tsne), ("PCA", z_pca)]
    
    for ax_idx, (name, z) in enumerate(methods):
        ax = axes[ax_idx]
        for l in uniq:
            m = np.array([x == l for x in labels])
            if star_mask is not None and m.any() and star_mask[m].all():
                ax.scatter(z[m, 0], z[m, 1], marker="*", s=120, alpha=1.0, label=l, edgecolors="black")
            else:
                ax.scatter(z[m, 0], z[m, 1], s=20, alpha=0.6, label=l)
        
        ax.set_title(f"{title}\n[{name}]", fontsize=12)
        ax.grid(True, alpha=0.3)
        if ax_idx == 1: # 오른쪽 그래프에만 범례 표시 (공간 확보)
            ax.legend(loc="best", fontsize=9)
            
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_distance_gamma_analysis(Z_train, gamma, save_path):
    plt.figure(figsize=(10, 5))
    
    if gamma is None:
        # Plot Cosine Similarity Distribution
        from sklearn.metrics.pairwise import cosine_similarity
        # Sample if too large to avoid memory issues
        N = Z_train.shape[0]
        if N > 2000:
            rng = np.random.default_rng(42)
            idx = rng.choice(N, 2000, replace=False)
            Z_sub = Z_train[idx]
        else:
            Z_sub = Z_train
            
        sims = cosine_similarity(Z_sub)
        # Upper triangle only (exclude self-sim 1.0)
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
        w_range = np.exp(-gamma * (d_range**2))
        ax2.plot(d_range, w_range, color='red', linewidth=2, label=f'RBF Kernel ($\gamma$={gamma:.4f})')
        ax2.set_ylabel('Similarity Weight')
        plt.title("Latent Space Distance Distribution vs. RBF Kernel")
        plt.xlabel("Euclidean Distance")

    plt.legend(loc='upper right'); plt.grid(True, alpha=0.2); plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()

def plot_driver_similarity_matrix(Aii_norm, item_owner_uid, train_drivers, save_path):
    # Sparse matrix에서 직접 계산하여 메모리 효율과 정확도 확보
    adj = Aii_norm.to_dense().cpu().numpy()
    n_users = len(train_drivers)
    sim_matrix = np.zeros((n_users, n_users))
    
    for i in range(n_users):
        mask_i = (item_owner_uid == i)
        n_items_i = np.sum(mask_i)
        if n_items_i == 0: continue
            
        for j in range(n_users):
            mask_j = (item_owner_uid == j)
            if not np.any(mask_j): continue
            
            # [수정] 전체 평균이 아닌 '아이템 i당 u->j로 나가는 가중치의 합'을 계산
            # 이렇게 하면 i의 데이터가 아무리 많아도 '아이템당 기여도'로 정규화됩니다.
            total_weight = adj[mask_i][:, mask_j].sum()
            sim_matrix[i, j] = total_weight / n_items_i 

    plt.figure(figsize=(8, 6))
    import seaborn as sns
    # 값이 너무 작을 수 있으므로 log scale을 고려하거나 fmt를 조절합니다.
    sns.heatmap(sim_matrix, xticklabels=train_drivers, yticklabels=train_drivers, 
                annot=True, fmt=".4f", cmap="YlGnBu")
    plt.title("Item-Item Connectivity (Normalized by Source Item Count)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_test_item_bridge(neigh_idx, neigh_w, item_owner_uid, train_drivers, save_path):
    driver_contribution = np.zeros(len(train_drivers))
    # 드라이버별 총 아이템 개수 계산
    user_item_counts = np.array([np.sum(item_owner_uid == i) for i in range(len(train_drivers))])
    
    for i in range(neigh_idx.shape[0]):
        indices = neigh_idx[i]
        weights = neigh_w[i]
        owners = item_owner_uid[indices]
        for owner, w in zip(owners, weights):
            driver_contribution[owner] += w
            
    # [수정] 데이터 양에 의한 편향 제거: '아이템당 평균 기여도'로 변환
    driver_contribution = driver_contribution / (user_item_counts + 1e-8)
    # 다시 비율(%)로 보기 위해 정규화
    driver_contribution /= (driver_contribution.sum() + 1e-8)
    
    plt.figure(figsize=(10, 5))
    plt.bar(train_drivers, driver_contribution, color='skyblue')
    plt.title("Bridge Contribution Density (Normalized by Train Data Size)")
    plt.ylabel("Relative Contribution Per Item")
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

