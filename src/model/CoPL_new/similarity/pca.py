"""
PCA 기반 Item-Item 유사도 그래프 빌더.

기존 방식: Flatten → 표준화 → PCA → Median Heuristic Gamma → kNN + RBF → A_ii
"""
import numpy as np
from sklearn.decomposition import PCA, KernelPCA

from src.model.CoPL_new.similarity.base import (
    ItemSimilarityBuilder, standardize_fit, standardize_apply, median_heuristic_gamma
)


class PCASimilarity(ItemSimilarityBuilder):
    """PCA + kNN + RBF 커널 기반 유사도 그래프 빌더."""

    def fit(self, item_series: np.ndarray, cfg) -> dict:
        N, T, D = item_series.shape
        X = item_series.reshape(N, T * D).astype(np.float32)

        # 표준화
        self.mu, self.sd = standardize_fit(X)
        Xs = standardize_apply(X, self.mu, self.sd)

        # PCA
        n_components = min(cfg.pca_dim, Xs.shape[1])
        if cfg.verbose > 0:
            print(f"  [PCA] components: {n_components} / {Xs.shape[1]}")

        self.pca = PCA(n_components=n_components, random_state=cfg.seed)
        Z = self.pca.fit_transform(Xs)

        # Gamma (RBF bandwidth)
        gamma_med = median_heuristic_gamma(Z, seed=cfg.seed)
        self.gamma = gamma_med * cfg.gamma_mul

        # kNN graph
        Aii_norm = self.build_knn_graph(
            Z, knn_k=cfg.knn_k, gamma=self.gamma, mutual=cfg.mutual)

        meta = {
            "method": "pca",
            "pca_dim": n_components,
            "explained_var_sum": float(self.pca.explained_variance_ratio_.sum()),
            "gamma_med": float(gamma_med),
            "gamma_mul": cfg.gamma_mul,
            "gamma": float(self.gamma),
            "knn_k": cfg.knn_k,
            "mutual": cfg.mutual,
        }
        if cfg.verbose > 0:
            print(f"  [PCA] meta: {meta}")

        return {
            "Aii_norm": Aii_norm,
            "Z_train": Z,
            "gamma": self.gamma,
            "meta": meta,
        }

    def transform_test(self, X_test: np.ndarray) -> np.ndarray:
        n_test, T, D = X_test.shape
        Xf = X_test.reshape(n_test, T * D).astype(np.float32)
        Xs = standardize_apply(Xf, self.mu, self.sd)
        return self.pca.transform(Xs)

    def get_affinity(self, Z_query: np.ndarray, Z_target: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        from sklearn.neighbors import NearestNeighbors
        nnbrs = NearestNeighbors(n_neighbors=min(k, Z_target.shape[0]), metric="euclidean")
        nnbrs.fit(Z_target)
        dist, nbr = nnbrs.kneighbors(Z_query, return_distance=True)
        # Compute weights using stored gamma
        w = self._compute_rbf_affinity(dist ** 2, self.gamma)
        return nbr, w


class KernelPCASimilarity(ItemSimilarityBuilder):
    """Kernel PCA (RBF) + kNN + RBF 커널 기반 유사도 그래프 빌더.

    PCA 대신 RBF Kernel PCA를 사용하여 비선형 매니폴드 구조를 포착합니다.
    """

    def fit(self, item_series: np.ndarray, cfg) -> dict:
        N, T, D = item_series.shape
        X = item_series.reshape(N, T * D).astype(np.float32)

        # 표준화
        self.mu, self.sd = standardize_fit(X)
        Xs = standardize_apply(X, self.mu, self.sd)

        n_components = min(cfg.pca_dim, Xs.shape[1])

        # Kernel PCA용 gamma: median heuristic을 원본 공간에서 계산
        kpca_gamma = median_heuristic_gamma(Xs, seed=cfg.seed)

        if cfg.verbose > 0:
            print(f"  [KernelPCA] components: {n_components}, "
                  f"kernel_gamma: {kpca_gamma:.6f}")

        self.kpca = KernelPCA(
            n_components=n_components,
            kernel="rbf",
            gamma=kpca_gamma,
            fit_inverse_transform=False,
            random_state=cfg.seed,
        )
        Z = self.kpca.fit_transform(Xs)

        # kNN 그래프용 gamma: latent 공간에서 다시 계산
        gamma_med = median_heuristic_gamma(Z, seed=cfg.seed)
        self.gamma = gamma_med * cfg.gamma_mul

        # kNN graph
        Aii_norm = self.build_knn_graph(
            Z, knn_k=cfg.knn_k, gamma=self.gamma, mutual=cfg.mutual)

        meta = {
            "method": "kernel_pca",
            "kernel": "rbf",
            "kpca_gamma": float(kpca_gamma),
            "pca_dim": n_components,
            "gamma_med": float(gamma_med),
            "gamma_mul": cfg.gamma_mul,
            "gamma": float(self.gamma),
            "knn_k": cfg.knn_k,
            "mutual": cfg.mutual,
        }
        if cfg.verbose > 0:
            print(f"  [KernelPCA] meta: {meta}")

        return {
            "Aii_norm": Aii_norm,
            "Z_train": Z,
            "gamma": self.gamma,
            "meta": meta,
        }

    def transform_test(self, X_test: np.ndarray) -> np.ndarray:
        n_test, T, D = X_test.shape
        Xf = X_test.reshape(n_test, T * D).astype(np.float32)
        Xs = standardize_apply(Xf, self.mu, self.sd)
        return self.kpca.transform(Xs)

    def get_affinity(self, Z_query: np.ndarray, Z_target: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        from sklearn.neighbors import NearestNeighbors
        nnbrs = NearestNeighbors(n_neighbors=min(k, Z_target.shape[0]), metric="euclidean")
        nnbrs.fit(Z_target)
        dist, nbr = nnbrs.kneighbors(Z_query, return_distance=True)
        w = self._compute_rbf_affinity(dist ** 2, self.gamma)
        return nbr, w
