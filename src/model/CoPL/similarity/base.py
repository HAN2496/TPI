"""
Item-Item 유사도 그래프 구축을 위한 추상 베이스 클래스.

Strategy 패턴으로 PCA, VAE 등 다양한 방식을 교체 가능하게 합니다.
"""
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
    """
    Item-Item 유사도 그래프를 구축하는 전략 인터페이스.

    구현체는 다음을 제공해야 합니다:
      - fit(): 학습 아이템으로 유사도 공간 + A_ii 구축
      - transform_test(): 테스트 아이템을 학습 유사도 공간에 투영
    """

    @abstractmethod
    def fit(self, item_series: np.ndarray, cfg) -> dict:
        """
        학습 아이템 시계열로 유사도 공간을 구축하고 A_ii 그래프를 생성합니다.

        Args:
            item_series: (n_items, T, D) 학습 아이템 시계열
            cfg: CFG 설정 객체

        Returns: dict with keys:
            "Aii_norm":  (n_items, n_items) 정규화된 sparse 인접행렬
            "Z_train":   (n_items, latent_dim) 학습 아이템의 latent 표현
            "gamma":     float, kNN 가중치 계산용 (또는 None)
            "meta":      dict, 메타 정보 (로깅/재현용)
        """

    @abstractmethod
    def get_affinity(self, Z_query: np.ndarray, Z_target: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Z_query의 각 포인트에 대해 Z_target에서 k개의 이웃을 찾고 가중치를 계산합니다.

        Args:
            Z_query: (n_query, d)
            Z_target: (n_target, d)
            k: 이웃 수

        Returns:
            indices: (n_query, k) 이웃 인덱스
            weights: (n_query, k) 이웃 가중치 (row-normalized, sum=1)
        """

    # ------------------------------------------------------------------
    # 공통 유틸리티
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_rbf_affinity(dist_sq: np.ndarray, gamma: float) -> np.ndarray:
        w = np.exp(-gamma * dist_sq)
        # Avoid division by zero
        w_sum = w.sum(axis=1, keepdims=True) + 1e-12
        return w / w_sum

    @staticmethod
    def _compute_cosine_affinity(sim: np.ndarray, temperature: float = 0.5) -> np.ndarray:
        # sim is cosine similarity [-1, 1]
        # Transform to weights: e.g., softmax(sim / temp) or (sim + 1) / 2
        # Here we use softmax for sharper attention
        w = np.exp(sim / temperature)
        w_sum = w.sum(axis=1, keepdims=True) + 1e-12
        return w / w_sum

    @staticmethod
    def build_knn_graph(
        Z: np.ndarray,
        knn_k: int,
        gamma: float,
        mutual: bool = False,
    ) -> torch.Tensor:
        """Z 공간에서 kNN + RBF 커널로 A_ii를 구축합니다."""
        from sklearn.neighbors import NearestNeighbors

        N = Z.shape[0]
        nnbrs = NearestNeighbors(
            n_neighbors=min(knn_k + 1, N), metric="euclidean")
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
    def build_cosine_knn_graph(
        Z: np.ndarray,
        knn_k: int,
        mutual: bool = False,
        temperature: float = 0.2,
    ) -> torch.Tensor:
        """Z 공간에서 Cosine Similarity 기반 kNN 그래프를 구축합니다. Z는 정규화되어 있어야 합니다."""
        from sklearn.neighbors import NearestNeighbors
        
        # Z 정규화 (Cosine Similarity를 Euclidean으로 근사하거나 직접 내적)
        # sklearn NearestNeighbors with metric='cosine' is feasible but 'euclidean' on normalized vectors is faster/equivalent for ranking
        # Euclidean d^2 = 2(1 - cos). So smaller d -> larger cos.
        # But to get actual cosine values, we need inner product.
        
        norm = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
        Z_norm = Z / norm
        
        N = Z.shape[0]
        # Use simple matrix multiplication for small N, or NearestNeighbors for large N
        # For consistency and scalability, use NearestNeighbors with metric='cosine' (or 'euclidean' on unit vectors)
        
        nnbrs = NearestNeighbors(
            n_neighbors=min(knn_k + 1, N), metric="euclidean")
        nnbrs.fit(Z_norm)
        dist, nbr = nnbrs.kneighbors(Z_norm, return_distance=True)
        
        # dist^2 = 2(1 - cos) => cos = 1 - dist^2 / 2
        # Or just compute dot product for selected neighbors
        
        rows, cols, vals = [], [], []
        for i in range(N):
            for j in nbr[i]:
                if j == i:
                    continue
                # Cosine similarity
                sim = np.dot(Z_norm[i], Z_norm[j])
                # Weight: softmax-like or simple power
                # w = exp(sim / T) is good for attention-like behavior
                w = float(np.exp(sim / temperature))
                rows.append(i); cols.append(int(j)); vals.append(w)

        return ItemSimilarityBuilder._make_symmetric_adj(rows, cols, vals, N, mutual)

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
        # Average duplicate entries (if any, separate from transpose sum) - usually coalesce sums them.
        # But we want (A + A.T) / 2 logic for undirected.
        # Actually standard logic: A <- A + A.T. 
        # But here A_ii is typically directed in kNN. 
        # Making it symmetric: A_sym = (A + A.T) / 2 is good.
        
        # Re-create to ensure symmetry scaling
        A = torch.sparse_coo_tensor(
            A.indices(), 0.5 * A.values(), A.size()).coalesce()

        return _normalize_square_adj(A)
