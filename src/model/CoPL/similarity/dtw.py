import numpy as np
import torch
from .base import ItemSimilarityBuilder, standardize_fit, standardize_apply


class DTWSimilarity(ItemSimilarityBuilder):
    def fit(self, item_series: np.ndarray, cfg) -> dict:
        N, T, D = item_series.shape
        self._T, self._D = T, D

        self.mu_stats, self.sd_stats = standardize_fit(item_series.reshape(N, T * D))
        Xs = standardize_apply(item_series.reshape(N, T * D), self.mu_stats, self.sd_stats)

        device = getattr(cfg, "device", "cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, str):
            device = torch.device(device)

        gamma_dtw = getattr(cfg, "dtw_gamma", 1.0)
        self.dtw_gamma = gamma_dtw

        X_tensor = torch.tensor(Xs.reshape(N, T, D), dtype=torch.float32).to(device)

        if cfg.verbose > 0:
            print(f"  [DTW] Computing Soft-DTW matrix (N={N}, T={T})...")

        dist_matrix = torch.zeros((N, N), device=device)
        batch_size = getattr(cfg, "dtw_batch_size", 100)

        for start_i in range(0, N, batch_size):
            end_i = min(start_i + batch_size, N)
            batch_q = X_tensor[start_i:end_i]
            B = batch_q.size(0)

            for b in range(B):
                idx = start_i + b
                q = batch_q[b:b+1]
                target_chunk_size = 500
                for tj in range(0, N, target_chunk_size):
                    tk = min(tj + target_chunk_size, N)
                    targets = X_tensor[tj:tk]
                    M = targets.size(0)
                    q_expand = q.expand(M, -1, -1)
                    d_mat = torch.sum((q_expand.unsqueeze(2) - targets.unsqueeze(1)) ** 2, dim=-1)

                    R = torch.full((M, T + 1, T + 1), float("inf"), device=device)
                    R[:, 0, 0] = 0
                    for i in range(1, T + 1):
                        for j in range(1, T + 1):
                            cat = torch.stack([R[:, i-1, j], R[:, i, j-1], R[:, i-1, j-1]], dim=1)
                            soft_min = -gamma_dtw * torch.logsumexp(-cat / gamma_dtw, dim=1)
                            R[:, i, j] = d_mat[:, i-1, j-1] + soft_min

                    dist_matrix[idx, tj:tk] = R[:, T, T]

            if cfg.verbose > 0 and (start_i // batch_size) % 5 == 0:
                print(f"    [DTW] Processed {min(end_i, N)}/{N} rows")

        self.dist_matrix = dist_matrix.cpu().numpy()

        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        d_vals = self.dist_matrix[mask]
        med_d = np.median(d_vals)
        self.gamma = 1.0 / (2.0 * (med_d ** 2) + 1e-12) * cfg.gamma_mul

        if cfg.verbose > 0:
            print(f"  [DTW] median_dist={med_d:.4f}, gamma={self.gamma:.6f}")

        rows, cols, vals = [], [], []
        for i in range(N):
            dists = self.dist_matrix[i].copy()
            dists[i] = float('inf')
            knns = np.argpartition(dists, cfg.knn_k)[:cfg.knn_k]
            for j in knns:
                w = np.exp(-self.gamma * (dists[j] ** 2))
                if w <= 1e-8:
                    continue
                rows.append(i); cols.append(int(j)); vals.append(float(w))

        A = ItemSimilarityBuilder._make_symmetric_adj(rows, cols, vals, N, cfg.mutual)
        meta = {"method": "soft_dtw", "dtw_gamma": gamma_dtw, "gamma": float(self.gamma), "median_dist": float(med_d)}

        return {"Aii_norm": A, "Z_train": Xs, "gamma": self.gamma, "meta": meta}

    def transform_test(self, X_test: np.ndarray) -> np.ndarray:
        n_test, T, D = X_test.shape
        Xf = X_test.reshape(n_test, T * D).astype(np.float32)
        return standardize_apply(Xf, self.mu_stats, self.sd_stats)

    def get_affinity(self, Z_query, Z_target, k):
        n_q = Z_query.shape[0]
        n_t = Z_target.shape[0]
        T, D = self._T, self._D
        device = "cuda" if torch.cuda.is_available() else "cpu"
        Q = torch.tensor(Z_query.reshape(n_q, T, D), dtype=torch.float32, device=device)
        Targets = torch.tensor(Z_target.reshape(n_t, T, D), dtype=torch.float32, device=device)
        gamma_dtw = self.dtw_gamma

        nbrs, weights = [], []
        chunk_size = 200

        for i in range(n_q):
            q = Q[i:i+1]
            dists_i = []
            for tj in range(0, n_t, chunk_size):
                tk = min(tj + chunk_size, n_t)
                t_chunk = Targets[tj:tk]
                M = t_chunk.size(0)
                q_exp = q.expand(M, -1, -1)
                d_mat = torch.sum((q_exp.unsqueeze(2) - t_chunk.unsqueeze(1)) ** 2, dim=-1)

                R = torch.full((M, T + 1, T + 1), float("inf"), device=device)
                R[:, 0, 0] = 0
                for ti in range(1, T + 1):
                    for tj_dp in range(1, T + 1):
                        cat = torch.stack([R[:, ti-1, tj_dp], R[:, ti, tj_dp-1], R[:, ti-1, tj_dp-1]], dim=1)
                        soft_min = -gamma_dtw * torch.logsumexp(-cat / gamma_dtw, dim=1)
                        R[:, ti, tj_dp] = d_mat[:, ti-1, tj_dp-1] + soft_min

                dists_i.append(R[:, T, T])

            dists_i = torch.cat(dists_i)
            vals, idxs = torch.topk(dists_i, k, largest=False)
            w = torch.exp(-self.gamma * (vals ** 2))
            w = w / (w.sum() + 1e-12)
            nbrs.append(idxs.cpu().numpy())
            weights.append(w.cpu().numpy())

        return np.stack(nbrs), np.stack(weights)
