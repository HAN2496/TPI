import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import silhouette_score

from src.utils.utils import _load_dataset_sequences

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

FEATURES = ["IMU_VerAccelVal", "Bounce_rate_6D", "Pitch_rate_6D", "IMU_LongAccelVal"]
DRIVER_NAMES = ["강신길", "조현석", "한규택", "박재일"]
TIME_RANGE = (5, 7)
DOWNSAMPLE = 5
GAMMA = None

from scipy.spatial.distance import pdist, squareform

def get_median_heuristic_gamma(X):
    # X: (N, Feature_dim) flattened trajectories
    # 모든 쌍 사이의 유클리드 거리 계산
    dists = pdist(X, metric='euclidean')
    
    # 거리의 중간값(Median) 계산
    sigma = np.median(dists)
    
    # Gamma 계산 (sklearn rbf_kernel 기준: exp(-gamma * d^2))
    gamma = 1.0 / (2 * (sigma ** 2))
    
    print(f"Computed Median Heuristic Gamma: {gamma:.6f}")
    return gamma


def load_all_drivers(driver_names, features, time_range, downsample):
    config = {'features': features}
    all_data = {}

    for driver_name in driver_names:
        X, y = _load_dataset_sequences(driver_name, time_range, downsample, config)
        all_data[driver_name] = {'X': X, 'y': y}
        print(f"  {driver_name}: {X.shape[0]} trajectories, shape={X.shape}")

    return all_data


def flatten_trajectories(X):
    N, T, D = X.shape
    return X.reshape(N, T * D)


def compute_similarity_matrix(all_data, gamma=None):
    all_X = []
    labels = []

    for driver_name, data in all_data.items():
        X_flat = flatten_trajectories(data['X'])
        y = data['y']

        for i in range(len(X_flat)):
            all_X.append(X_flat[i])
            labels.append((driver_name, int(y[i])))

    all_X = np.array(all_X)
    print(f"\nTotal trajectories: {len(all_X)}, feature dim: {all_X.shape[1]}")

    K = rbf_kernel(all_X, gamma=gamma)

    return K, labels, all_X


def plot_full_similarity_matrix(K, labels, gamma, save_path):
    plt.figure(figsize=(14, 12))

    driver_names = sorted(set(l[0] for l in labels))
    driver_to_idx = {name: i for i, name in enumerate(driver_names)}

    sorted_indices = sorted(range(len(labels)), key=lambda i: (driver_to_idx[labels[i][0]], labels[i][1]))
    K_sorted = K[np.ix_(sorted_indices, sorted_indices)]
    sorted_labels = [labels[i] for i in sorted_indices]

    ax = sns.heatmap(K_sorted, cmap='viridis', vmin=0, vmax=1, xticklabels=False, yticklabels=False)

    boundaries = [0]
    current_driver = sorted_labels[0][0]
    for i, (driver, _) in enumerate(sorted_labels):
        if driver != current_driver:
            boundaries.append(i)
            current_driver = driver
    boundaries.append(len(sorted_labels))

    for b in boundaries[1:-1]:
        ax.axhline(y=b, color='white', linewidth=2)
        ax.axvline(x=b, color='white', linewidth=2)

    for i in range(len(boundaries) - 1):
        mid = (boundaries[i] + boundaries[i+1]) / 2
        driver = sorted_labels[boundaries[i]][0]
        ax.text(-2, mid, driver, ha='right', va='center', fontsize=10)

    gamma_str = f"γ={gamma:.6f}" if gamma else "γ=1/n_features"
    plt.title(f"RBF Kernel Similarity Matrix ({gamma_str})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def compute_clustering_metrics(K, labels):
    driver_labels = np.array([l[0] for l in labels])
    unique_drivers = np.unique(driver_labels)
    driver_to_int = {d: i for i, d in enumerate(unique_drivers)}
    int_labels = np.array([driver_to_int[d] for d in driver_labels])

    # Silhouette Score (kernel을 거리로 변환: D = 1 - K)
    D = 1 - K
    np.fill_diagonal(D, 0)
    sil_score = silhouette_score(D, int_labels, metric='precomputed')

    # Separation Ratio (within-cluster similarity / between-cluster similarity)
    n = len(labels)
    within_sum, within_count = 0, 0
    between_sum, between_count = 0, 0

    for i in range(n):
        for j in range(i + 1, n):
            if driver_labels[i] == driver_labels[j]:
                within_sum += K[i, j]
                within_count += 1
            else:
                between_sum += K[i, j]
                between_count += 1

    within_avg = within_sum / within_count
    between_avg = between_sum / between_count
    separation_ratio = within_avg / between_avg

    return sil_score, separation_ratio, within_avg, between_avg


def analyze_gamma_sensitivity(all_data, gammas=[0.0001, 0.001, 0.01, 0.1, 1.0],
                               save_path="artifacts/analysis/rkhs_gamma_sensitivity.png"):
    X_sample = []
    y_sample = []

    for data in all_data.values():
        X_flat = flatten_trajectories(data['X'])
        X_sample.append(X_flat)
        y_sample.append(data['y'])

    X_all = np.concatenate(X_sample)
    y_all = np.concatenate(y_sample)

    if len(X_all) > 500:
        indices = np.random.choice(len(X_all), 500, replace=False)
        X_all = X_all[indices]
        y_all = y_all[indices]

    fig, axes = plt.subplots(2, len(gammas), figsize=(4*len(gammas), 8))

    for i, gamma in enumerate(gammas):
        K = rbf_kernel(X_all, gamma=gamma)

        ax = axes[0, i]
        sns.heatmap(K, cmap='viridis', ax=ax, cbar=False, vmin=0, vmax=1)
        ax.set_title(f"γ={gamma}")
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axes[1, i]
        mask = ~np.eye(len(K), dtype=bool)
        values = K[mask]
        ax.hist(values, bins=50, alpha=0.7, density=True)
        ax.set_xlabel("Similarity")
        ax.set_title(f"μ={values.mean():.3f}, σ={values.std():.3f}")
        ax.grid(True, alpha=0.3)

    axes[0, 0].set_ylabel("Sample Index")
    axes[1, 0].set_ylabel("Density")

    plt.suptitle("Gamma Sensitivity Analysis", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    import os
    os.makedirs("artifacts/analysis", exist_ok=True)

    print("=" * 60)
    print("RKHS Trajectory Similarity Analysis")
    print("=" * 60)

    # 1. Load Data
    print(f"\n[1] Loading data (time_range={TIME_RANGE}, features={len(FEATURES)})...")
    all_data = load_all_drivers(DRIVER_NAMES, FEATURES, TIME_RANGE, DOWNSAMPLE)

    # 2. Compute full similarity matrix (default gamma)
    print("\n[2] Computing full similarity matrix (default gamma)...")
    K_default, labels, all_X = compute_similarity_matrix(all_data, gamma=None)
    plot_full_similarity_matrix(K_default, labels, None, "artifacts/analysis/rkhs_default_gamma.png")

    # 3. Compute full similarity matrix (median heuristic gamma)
    print("\n[3] Computing full similarity matrix (median heuristic gamma)...")
    # median_gamma = get_median_heuristic_gamma(all_X)
    median_gamma = 0.001
    K_median, _, _ = compute_similarity_matrix(all_data, gamma=median_gamma)
    plot_full_similarity_matrix(K_median, labels, median_gamma, "artifacts/analysis/rkhs_median_gamma.png")

    # 4. Clustering metrics
    print("\n[4] Computing clustering metrics (median heuristic gamma)...")
    sil_score, sep_ratio, within_avg, between_avg = compute_clustering_metrics(K_median, labels)
    print(f"  Silhouette Score: {sil_score:.4f}")
    print(f"  Separation Ratio: {sep_ratio:.4f} (within={within_avg:.4f}, between={between_avg:.4f})")

    # 5. Gamma sensitivity
    print("\n[5] Analyzing gamma sensitivity...")
    analyze_gamma_sensitivity(all_data)

    print("\nDone!")


if __name__ == "__main__":
    main()
