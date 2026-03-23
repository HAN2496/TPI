from abc import ABC, abstractmethod
import numpy as np
import torch


def _normalize_square_adj(adj: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """D^-1/2 * A * D^-1/2 for (n x n) sparse COO."""
    adj = adj.coalesce()
    deg = torch.sparse.sum(adj, dim=1).to_dense()
    idx = adj.indices()
    val = adj.values()
    norm = torch.sqrt(deg[idx[0]] * deg[idx[1]] + eps)
    val = val / norm
    return torch.sparse_coo_tensor(idx, val, adj.size()).coalesce()


def standardize_fit(X: np.ndarray):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-6
    return mu, sd


def standardize_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray):
    return (X - mu) / sd


def median_heuristic_gamma(Z: np.ndarray, max_pairs: int = 200000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    N = Z.shape[0]
    if N <= 2000:
        from scipy.spatial.distance import pdist
        d = pdist(Z, metric="euclidean")
        sigma = np.median(d)
    else:
        i = rng.integers(0, N, size=max_pairs)
        j = rng.integers(0, N, size=max_pairs)
        mask = i != j
        di = Z[i[mask]] - Z[j[mask]]
        dist = np.sqrt((di * di).sum(axis=1))
        sigma = np.median(dist)
    return float(1.0 / (2.0 * (sigma ** 2) + 1e-12))


class ItemSimilarityBuilder(ABC):

    @abstractmethod
    def fit(self, item_series: np.ndarray, cfg) -> dict:
        pass

    @abstractmethod
    def get_affinity(self, Z_query: np.ndarray, Z_target: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        pass

    @staticmethod
    def _compute_rbf_affinity(dist_sq: np.ndarray, gamma: float) -> np.ndarray:
        w = np.exp(-gamma * dist_sq)
        return w / (w.sum(axis=1, keepdims=True) + 1e-12)

    @staticmethod
    def _compute_laplacian_affinity(dist: np.ndarray, gamma: float) -> np.ndarray:
        w = np.exp(-gamma * dist)
        return w / (w.sum(axis=1, keepdims=True) + 1e-12)

    @staticmethod
    def _compute_cosine_affinity(sim: np.ndarray, temperature: float = 0.5) -> np.ndarray:
        w = np.exp(sim / temperature)
        return w / (w.sum(axis=1, keepdims=True) + 1e-12)

    @staticmethod
    def build_knn_graph(Z: np.ndarray, knn_k: int, gamma: float, mutual: bool = False) -> torch.Tensor:
        from sklearn.neighbors import NearestNeighbors
        N = Z.shape[0]
        nnbrs = NearestNeighbors(n_neighbors=min(knn_k + 1, N), metric="euclidean")
        nnbrs.fit(Z)
        dist, nbr = nnbrs.kneighbors(Z, return_distance=True)

        rows, cols, vals = [], [], []
        for i in range(N):
            for j, d in zip(nbr[i], dist[i]):
                if j == i:
                    continue
                w = float(np.exp(-gamma * (d ** 2)))
                if w <= 1e-8:
                    continue
                rows.append(i); cols.append(int(j)); vals.append(w)

        return ItemSimilarityBuilder._make_symmetric_adj(rows, cols, vals, N, mutual)

    @staticmethod
    def build_cosine_knn_graph(Z: np.ndarray, knn_k: int, mutual: bool = False, temperature: float = 0.2) -> torch.Tensor:
        from sklearn.neighbors import NearestNeighbors
        norm = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
        Z_norm = Z / norm
        N = Z.shape[0]
        nnbrs = NearestNeighbors(n_neighbors=min(knn_k + 1, N), metric="euclidean")
        nnbrs.fit(Z_norm)
        _, nbr = nnbrs.kneighbors(Z_norm, return_distance=True)

        rows, cols, vals = [], [], []
        for i in range(N):
            for j in nbr[i]:
                if j == i:
                    continue
                sim = np.dot(Z_norm[i], Z_norm[j])
                w = float(np.exp(sim / temperature))
                rows.append(i); cols.append(int(j)); vals.append(w)

        return ItemSimilarityBuilder._make_symmetric_adj(rows, cols, vals, N, mutual)

    @staticmethod
    def build_laplacian_knn_graph(Z: np.ndarray, knn_k: int, gamma: float, mutual: bool = False) -> torch.Tensor:
        from sklearn.neighbors import NearestNeighbors
        N = Z.shape[0]
        nnbrs = NearestNeighbors(n_neighbors=min(knn_k + 1, N), metric="euclidean")
        nnbrs.fit(Z)
        dist, nbr = nnbrs.kneighbors(Z, return_distance=True)

        rows, cols, vals = [], [], []
        for i in range(N):
            for j, d in zip(nbr[i], dist[i]):
                if j == i:
                    continue
                w = float(np.exp(-gamma * d))
                if w <= 1e-8:
                    continue
                rows.append(i); cols.append(int(j)); vals.append(w)

        return ItemSimilarityBuilder._make_symmetric_adj(rows, cols, vals, N, mutual)

    @staticmethod
    def build_wasserstein_knn_graph(Z_mu: np.ndarray, Z_sigma: np.ndarray, knn_k: int,
                                    gamma: float, mutual: bool = False, kernel: str = "rbf") -> torch.Tensor:
        """W2 distance between diagonal Gaussians: sqrt(||μ_i-μ_j||² + ||σ_i-σ_j||²)."""
        from sklearn.neighbors import NearestNeighbors
        Z = np.concatenate([Z_mu, Z_sigma], axis=1)
        N = Z.shape[0]
        nnbrs = NearestNeighbors(n_neighbors=min(knn_k + 1, N), metric="euclidean")
        nnbrs.fit(Z)
        dist, nbr = nnbrs.kneighbors(Z, return_distance=True)

        rows, cols, vals = [], [], []
        for i in range(N):
            for j, d in zip(nbr[i], dist[i]):
                if j == i:
                    continue
                w = float(np.exp(-gamma * d) if "laplacian" in kernel else np.exp(-gamma * (d ** 2)))
                if w <= 1e-8:
                    continue
                rows.append(i); cols.append(int(j)); vals.append(w)

        return ItemSimilarityBuilder._make_symmetric_adj(rows, cols, vals, N, mutual)

    def visualize(self, save_dir, item_series, item_owner_uid,
                  train_drivers, feature_names, item_labels):
        pass

    @staticmethod
    def _make_symmetric_adj(rows, cols, vals, N, mutual):
        if mutual:
            S = set(zip(rows, cols))
            keep = [(j, i) in S for (i, j) in zip(rows, cols)]
            rows = [r for r, k in zip(rows, keep) if k]
            cols = [c for c, k in zip(cols, keep) if k]
            vals = [v for v, k in zip(vals, keep) if k]

        idx = torch.tensor([rows, cols], dtype=torch.long)
        val = torch.tensor(vals, dtype=torch.float32)
        A = torch.sparse_coo_tensor(idx, val, size=(N, N)).coalesce()
        A = (A + A.transpose(0, 1)).coalesce()
        A = torch.sparse_coo_tensor(A.indices(), 0.5 * A.values(), A.size()).coalesce()
        return _normalize_square_adj(A)
