import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import rbf_kernel, cosine_similarity
from scipy.spatial.distance import pdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from src.utils.utils import _load_dataset_sequences

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

FEATURES = ["IMU_VerAccelVal", "Bounce_rate_6D", "Pitch_rate_6D", "IMU_LongAccelVal"]
DRIVER_NAMES = ["강신길", "김진명", "김태근", "조현석", "한규택", "박재일", "이지환"]
TIME_RANGE = (5, 7)
DOWNSAMPLE = 1


# =========================
# Similarity helpers
# =========================
def get_median_heuristic_gamma(X):
    # X: (N, dim) standardized/embedded
    dists = pdist(X, metric='euclidean')
    sigma = np.median(dists)
    gamma = 1.0 / (2 * (sigma ** 2) + 1e-12)
    return float(gamma)


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


def standardize(X):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-6
    return (X - mu) / sd


def apply_pca(X, n_components=50, whiten=False):
    # 안전: n_components가 feature dim보다 크면 줄이기
    n_components = int(min(n_components, X.shape[1]))
    pca = PCA(n_components=n_components, whiten=whiten, random_state=42)
    Z = pca.fit_transform(X)
    return Z, float(pca.explained_variance_ratio_.sum())


def topk_stats(K, k=20):
    N = K.shape[0]
    k = int(min(k, N - 1))
    K2 = K.copy()
    np.fill_diagonal(K2, -np.inf)
    idx = np.argpartition(-K2, kth=k, axis=1)[:, :k]
    vals = np.take_along_axis(K2, idx, axis=1)
    top1 = vals.max(axis=1)
    topk = vals.mean(axis=1)
    return {
        "top1_mean": float(np.mean(top1)),
        "top1_med": float(np.median(top1)),
        "topk_mean": float(np.mean(topk)),
        "topk_med": float(np.median(topk)),
    }


def random_pair_stats(K, n_samples=200000, seed=0):
    rng = np.random.default_rng(seed)
    N = K.shape[0]
    i = rng.integers(0, N, size=n_samples)
    j = rng.integers(0, N, size=n_samples)
    mask = (i != j)
    vals = K[i[mask], j[mask]]
    return float(vals.mean()), float(vals.std())


def build_knn_graph_from_K(K, k=20, mutual=False):
    N = K.shape[0]
    k = int(min(k, N - 1))

    K2 = K.copy()
    np.fill_diagonal(K2, -np.inf)

    nbr = np.argpartition(-K2, kth=k, axis=1)[:, :k]  # (N,k)
    rows = np.repeat(np.arange(N), k)
    cols = nbr.reshape(-1)
    vals = K2[rows, cols]

    keep = vals > 0
    rows, cols, vals = rows[keep], cols[keep], vals[keep]

    if mutual:
        # mutual kNN: keep i->j only if j->i also exists
        S = set(zip(rows.tolist(), cols.tolist()))
        keep2 = [(j, i) in S for (i, j) in zip(rows, cols)]
        keep2 = np.array(keep2, dtype=bool)
        rows, cols, vals = rows[keep2], cols[keep2], vals[keep2]

    A = csr_matrix((vals, (rows, cols)), shape=(N, N))
    A = 0.5 * (A + A.T)  # symmetrize
    return A


def graph_stats(A):
    n_components, comp = connected_components(A, directed=False, connection='weak')
    counts = np.bincount(comp)
    giant = counts.max() / A.shape[0]
    deg = np.array(A.getnnz(axis=1))
    return {
        "n_components": int(n_components),
        "giant_ratio": float(giant),
        "deg_mean": float(deg.mean()),
        "deg_med": float(np.median(deg)),
        "deg_max": int(deg.max()),
    }


def driver_nn_hitrate(K, driver_labels):
    K2 = K.copy()
    np.fill_diagonal(K2, -np.inf)
    nn = np.argmax(K2, axis=1)
    hit = (driver_labels[nn] == driver_labels)
    return float(hit.mean())


def eval_similarity_pipeline(X_flat, driver_labels, kind="rbf", gamma=None,
                            use_pca=False, pca_dim=50, k=20, mutual=False):
    X = standardize(X_flat)

    pca_info = None
    if use_pca:
        X, ev = apply_pca(X, n_components=pca_dim, whiten=False)
        pca_info = ev

    if kind == "rbf":
        if gamma is None:
            raise ValueError("RBF requires gamma")
        K = rbf_kernel(X, gamma=gamma)
    elif kind == "cosine":
        K = cosine_similarity(X)
        K = (K + 1) / 2  # [-1,1] -> [0,1]
    else:
        raise ValueError("kind must be 'rbf' or 'cosine'")

    tstats = topk_stats(K, k=k)
    rmean, rstd = random_pair_stats(K)
    A = build_knn_graph_from_K(K, k=k, mutual=mutual)
    gstats = graph_stats(A)
    nn_acc = driver_nn_hitrate(K, driver_labels)

    out = {
        "kind": kind,
        "gamma": gamma,
        "use_pca": use_pca,
        "pca_dim": int(pca_dim) if use_pca else None,
        "pca_ev_sum": pca_info,
        "k": int(k),
        "mutual": bool(mutual),
        "rand_mean": rmean,
        "rand_std": rstd,
        "nn_driver_acc": nn_acc,
        **tstats,
        **gstats
    }
    return out


# =========================
# EXPERIMENT GRID (SWEEPS)
# =========================
# k_list = [5, 10, 15, 20, 30, 40, 60]
k_list = [5, 10]
mutual_list = [False, True]
# pca_dims = [None, 5, 10, 20, 30, 50, 80, 120]  # None이면 PCA 미사용
pca_dims = [1, 2, 3, 4, 5]

fixed_gammas = [5e-4, 8e-4, 1e-3, 1.5e-3, 2e-3, 3e-3, 5e-3, 1e-2]
median_multipliers = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

# 너무 많으면 제한(원하면 None으로)
MAX_EXPERIMENTS = None  # 예: 2000


def build_experiments(gamma_med_flat, gamma_med_pca):
    experiments = []

    def add_exp(name, kind, gamma, use_pca, pca_dim, k, mutual):
        experiments.append(dict(
            name=name,
            kind=kind,
            gamma=gamma,
            use_pca=use_pca,
            pca_dim=pca_dim,
            k=k,
            mutual=mutual
        ))

    # 1) COSINE experiments (gamma 없음)
    for pca_dim in pca_dims:
        use_pca = (pca_dim is not None)
        for k in k_list:
            for mutual in mutual_list:
                tag = f"pca{pca_dim}" if use_pca else "flat"
                add_exp(
                    name=f"{tag}_cosine_k{k}_mut{int(mutual)}",
                    kind="cosine",
                    gamma=None,
                    use_pca=use_pca,
                    pca_dim=(pca_dim if use_pca else 50),
                    k=k,
                    mutual=mutual
                )

    # 2) RBF experiments: fixed gamma sweep
    for pca_dim in pca_dims:
        use_pca = (pca_dim is not None)
        for gamma in fixed_gammas:
            for k in k_list:
                for mutual in mutual_list:
                    tag = f"pca{pca_dim}" if use_pca else "flat"
                    add_exp(
                        name=f"{tag}_rbf_g{gamma:g}_k{k}_mut{int(mutual)}",
                        kind="rbf",
                        gamma=float(gamma),
                        use_pca=use_pca,
                        pca_dim=(pca_dim if use_pca else 50),
                        k=k,
                        mutual=mutual
                    )

    # 3) RBF experiments: median gamma * multipliers (space별 median 사용)
    for pca_dim in pca_dims:
        use_pca = (pca_dim is not None)

        if use_pca:
            # 안전: 혹시 없으면 flatten median으로 fallback
            base_gamma = gamma_med_pca.get(pca_dim, gamma_med_flat)
        else:
            base_gamma = gamma_med_flat

        for mul in median_multipliers:
            gamma = float(base_gamma * mul)
            for k in k_list:
                for mutual in mutual_list:
                    tag = f"pca{pca_dim}" if use_pca else "flat"
                    add_exp(
                        name=f"{tag}_rbf_medx{mul:g}_k{k}_mut{int(mutual)}",
                        kind="rbf",
                        gamma=gamma,
                        use_pca=use_pca,
                        pca_dim=(pca_dim if use_pca else 50),
                        k=k,
                        mutual=mutual
                    )

    # (선택) 실험 수 제한
    if MAX_EXPERIMENTS is not None and len(experiments) > MAX_EXPERIMENTS:
        rng = np.random.default_rng(0)
        keep_idx = rng.choice(len(experiments), size=MAX_EXPERIMENTS, replace=False)
        experiments = [experiments[i] for i in keep_idx]

    return experiments


def main():
    os.makedirs("artifacts/analysis", exist_ok=True)

    print("=" * 60)
    print("Similarity Pipeline Evaluation (kNN graph ready)")
    print("=" * 60)

    # 1) Load Data
    print(f"\n[1] Loading data (time_range={TIME_RANGE}, features={len(FEATURES)})...")
    all_data = load_all_drivers(DRIVER_NAMES, FEATURES, TIME_RANGE, DOWNSAMPLE)

    # 2) Collect all trajectories + driver labels
    all_X = []
    driver_labels = []

    driver_names_sorted = sorted(all_data.keys())
    driver_to_int = {d: i for i, d in enumerate(driver_names_sorted)}

    for dname in driver_names_sorted:
        X = all_data[dname]["X"]
        X_flat = flatten_trajectories(X)
        all_X.append(X_flat)
        driver_labels.append(np.full((X_flat.shape[0],), driver_to_int[dname], dtype=np.int64))

    X_flat = np.concatenate(all_X, axis=0)
    driver_labels = np.concatenate(driver_labels, axis=0)

    print(f"  Total trajectories: {X_flat.shape[0]}")
    print(f"  Flatten dim       : {X_flat.shape[1]}")
    print(f"  Drivers           : {driver_names_sorted}")

    # 3) median gamma (flatten/std space)
    X_std = standardize(X_flat)
    gamma_med_flat = get_median_heuristic_gamma(X_std)
    print(f"\n[2] Median-heuristic gamma (flatten/std space): {gamma_med_flat:.6g}")

    # 4) PCA space median gammas (None은 건너뜀!)
    gamma_med_pca = {}
    ev_pca = {}

    for pdim in pca_dims:
        if pdim is None:
            continue
        Z, ev = apply_pca(X_std, n_components=pdim, whiten=False)
        ev_pca[pdim] = ev
        gamma_med_pca[pdim] = get_median_heuristic_gamma(Z)
        print(f"  PCA{pdim}: EV_sum={ev_pca[pdim]:.4f}, median_gamma={gamma_med_pca[pdim]:.6g}")

    print("  gamma_med_pca keys:", sorted(gamma_med_pca.keys()))

    # 5) Build experiments
    experiments = build_experiments(gamma_med_flat, gamma_med_pca)
    print(f"\n[3] Total experiments: {len(experiments)}")
    print("  Example configs:")
    for e in experiments[:5]:
        print("   ", e)

    # 6) Run experiments
    results = []
    for exp in experiments:
        out = eval_similarity_pipeline(
            X_flat=X_flat,
            driver_labels=driver_labels,
            kind=exp["kind"],
            gamma=exp["gamma"],
            use_pca=exp["use_pca"],
            pca_dim=exp["pca_dim"],
            k=exp["k"],
            mutual=exp["mutual"]
        )
        out["name"] = exp["name"]
        results.append(out)

        gap = out["topk_mean"] - out["rand_mean"]
        print(
            f"  - {exp['name']:<28} "
            f"topk_mean={out['topk_mean']:.4f} rand_mean={out['rand_mean']:.4f} gap={gap:+.4f} | "
            f"giant={out['giant_ratio']:.3f} comps={out['n_components']:<4} "
            f"deg_mean={out['deg_mean']:.1f} deg_max={out['deg_max']:<4} | "
            f"NN@1(driver)={out['nn_driver_acc']:.3f}"
        )

    # 7) Save CSV
    csv_path = "artifacts/analysis/similarity_eval_results.csv"
    fieldnames = [
        "name", "kind", "gamma", "use_pca", "pca_dim", "pca_ev_sum", "k", "mutual",
        "rand_mean", "rand_std", "top1_mean", "top1_med", "topk_mean", "topk_med",
        "nn_driver_acc", "n_components", "giant_ratio", "deg_mean", "deg_med", "deg_max"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, None) for k in fieldnames})

    print(f"\n[4] Saved results CSV: {csv_path}")

    # 8) Simple ranking heuristic
    def score(r):
        gap = (r["topk_mean"] - r["rand_mean"])
        frag_penalty = max(0.0, 0.8 - r["giant_ratio"])
        # use r['k'] per row here (not outer k)
        k0 = max(1, int(r["k"]))
        hub_penalty = max(0.0, (r["deg_max"] - 3 * k0) / (10 * k0))
        return gap - 0.2 * frag_penalty - 0.1 * hub_penalty

    ranked = sorted(results, key=score, reverse=True)

    print("\n[5] Top candidates (by gap/graph sanity heuristic):")
    for r in ranked[:10]:
        gap = r["topk_mean"] - r["rand_mean"]
        print(
            f"  * {r['name']:<28} score={score(r):+.4f} gap={gap:+.4f} "
            f"giant={r['giant_ratio']:.3f} comps={r['n_components']} deg_max={r['deg_max']}"
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
