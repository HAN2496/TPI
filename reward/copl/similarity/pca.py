import numpy as np
from sklearn.decomposition import PCA, KernelPCA

from .base import ItemSimilarityBuilder, standardize_fit, standardize_apply, median_heuristic_gamma


class PCASimilarity(ItemSimilarityBuilder):
    def fit(self, item_series: np.ndarray, cfg) -> dict:
        N, T, D = item_series.shape
        X = item_series.reshape(N, T * D).astype(np.float32)
        self.mu, self.sd = standardize_fit(X)
        Xs = standardize_apply(X, self.mu, self.sd)

        n_components = min(cfg.pca_dim, Xs.shape[1])
        if cfg.verbose > 0:
            print(f"  [PCA] components: {n_components} / {Xs.shape[1]}")

        self.pca = PCA(n_components=n_components, random_state=cfg.seed)
        Z = self.pca.fit_transform(Xs)

        gamma_med = median_heuristic_gamma(Z, seed=cfg.seed)
        self.gamma = gamma_med * cfg.gamma_mul
        Aii_norm = self.build_knn_graph(Z, knn_k=cfg.knn_k, gamma=self.gamma, mutual=cfg.mutual)

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

        return {"Aii_norm": Aii_norm, "Z_train": Z, "gamma": self.gamma, "meta": meta}

    def transform_test(self, X_test: np.ndarray) -> np.ndarray:
        n_test, T, D = X_test.shape
        Xf = X_test.reshape(n_test, T * D).astype(np.float32)
        return self.pca.transform(standardize_apply(Xf, self.mu, self.sd))

    def get_affinity(self, Z_query, Z_target, k):
        from sklearn.neighbors import NearestNeighbors
        nnbrs = NearestNeighbors(n_neighbors=min(k, Z_target.shape[0]), metric="euclidean")
        nnbrs.fit(Z_target)
        dist, nbr = nnbrs.kneighbors(Z_query, return_distance=True)
        return nbr, self._compute_rbf_affinity(dist ** 2, self.gamma)


class KernelPCASimilarity(ItemSimilarityBuilder):
    def fit(self, item_series: np.ndarray, cfg) -> dict:
        N, T, D = item_series.shape
        X = item_series.reshape(N, T * D).astype(np.float32)
        self.mu, self.sd = standardize_fit(X)
        Xs = standardize_apply(X, self.mu, self.sd)

        n_components = min(cfg.pca_dim, Xs.shape[1])
        kpca_gamma = median_heuristic_gamma(Xs, seed=cfg.seed)

        if cfg.verbose > 0:
            print(f"  [KernelPCA] components: {n_components}, kernel_gamma: {kpca_gamma:.6f}")

        self.kpca = KernelPCA(
            n_components=n_components, kernel="rbf", gamma=kpca_gamma,
            fit_inverse_transform=False, random_state=cfg.seed)
        Z = self.kpca.fit_transform(Xs)

        gamma_med = median_heuristic_gamma(Z, seed=cfg.seed)
        self.gamma = gamma_med * cfg.gamma_mul
        Aii_norm = self.build_knn_graph(Z, knn_k=cfg.knn_k, gamma=self.gamma, mutual=cfg.mutual)

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

        return {"Aii_norm": Aii_norm, "Z_train": Z, "gamma": self.gamma, "meta": meta}

    def transform_test(self, X_test: np.ndarray) -> np.ndarray:
        n_test, T, D = X_test.shape
        Xf = X_test.reshape(n_test, T * D).astype(np.float32)
        return self.kpca.transform(standardize_apply(Xf, self.mu, self.sd))

    def get_affinity(self, Z_query, Z_target, k):
        from sklearn.neighbors import NearestNeighbors
        nnbrs = NearestNeighbors(n_neighbors=min(k, Z_target.shape[0]), metric="euclidean")
        nnbrs.fit(Z_target)
        dist, nbr = nnbrs.kneighbors(Z_query, return_distance=True)
        return nbr, self._compute_rbf_affinity(dist ** 2, self.gamma)
