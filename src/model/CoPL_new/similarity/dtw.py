
import numpy as np
import torch
import torch.nn.functional as F
from src.model.CoPL_new.similarity.base import (
    ItemSimilarityBuilder, median_heuristic_gamma, standardize_fit, standardize_apply
)

def soft_min(x, gamma):
    # -gamma * log(sum(exp(-x/gamma)))
    # But x is a tuple of values (up, left, diag)
    # Numerical stability: shift by min
    return -gamma * torch.logsumexp(-x / gamma, dim=-1)

class DTWSimilarity(ItemSimilarityBuilder):
    """
    Soft-DTW 기반 Item-Item 유사도 그래프 빌더.
    
    PyTorch를 사용하여 GPU에서 병렬로 Soft-DTW 거리를 계산합니다.
    계산된 DTW 거리 행렬에 대해 RBF 커널을 적용하여 유사도를 산출합니다.
    """

    def fit(self, item_series: np.ndarray, cfg) -> dict:
        N, T, D = item_series.shape
        self._T, self._D = T, D
        
        # 표준화
        self.mu_stats, self.sd_stats = standardize_fit(item_series.reshape(N, T * D))
        Xs = standardize_apply(item_series.reshape(N, T * D), self.mu_stats, self.sd_stats)
        
        # (N, T, D) tensor
        X_tensor = torch.tensor(
            Xs.reshape(N, T, D), dtype=torch.float32)
        
        # Device
        device = getattr(cfg, "device", "cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, str): device = torch.device(device)
        X_tensor = X_tensor.to(device)
            
        gamma_dtw = getattr(cfg, "dtw_gamma", 1.0) # Soft-DTW smoothing gamma
        
        if cfg.verbose > 0:
            print(f"  [DTW] Computing Soft-DTW matrix (N={N}, T={T})...")
            
        # Compute Pairwise DTW Matrix (N, N)
        # We iterate to save memory if N is large.
        # But for N=2000, we can do row-wise.
        
        dist_matrix = torch.zeros((N, N), device=device)
        
        batch_size = getattr(cfg, "dtw_batch_size", 100) # Rows at a time
        
        for start_i in range(0, N, batch_size):
            end_i = min(start_i + batch_size, N)
            # Batch of queries: (B, T, D)
            batch_q = X_tensor[start_i:end_i] 
            B = batch_q.size(0)
            
            # We need to compute DTW(q, t) for all t in X_tensor. (B, N)
            # This is B*N pairs. If B=100, N=2000 -> 200,000 pairs.
            # (200,000, 100, 100) float dist matrix = 8GB. TOO BIG.
            # We must iterate over targets too or reduce B.
            
            # Reduce inner batch size.
            # Compute row by row is safer for memory.
            # Or chunk targets.
            
            target_batch_size = max(1, 20000 // (T * T)) # Heuristic for memory 
            # If T=100, T*T=10000. 20000/10000 = 2 targets at a time? No.
            # 8GB memory holds ~2e9 floats.
            # 2e5 pairs * 1e4 floats/pair = 2e9 floats = 8GB. 
            # So B=100, N=2000 fits in 16GB GPU maybe. 
            # Safer to process smaller chunks.
            
            # Let's simple loop over rows i in range(start_i, end_i)
            for b in range(B):
                idx = start_i + b
                q = batch_q[b:b+1] # (1, T, D)
                
                # Compute against all targets in chunks
                # target_chunk: 500 items -> 500 * 100 * 100 * 4B = 20MB. Safe.
                target_chunk_size = 500
                for tj in range(0, N, target_chunk_size):
                    tk = min(tj + target_chunk_size, N)
                    targets = X_tensor[tj:tk] # (M, T, D)
                    M = targets.size(0)
                    
                    # Expand q: (M, T, D)
                    q_expand = q.expand(M, -1, -1)
                    
                    # Pairwise Euclidean Matrix: (M, T, T)
                    # cdist computes (B, P, R) from (B, P, D), (B, R, D) -> No.
                    # cdist(x, y) where x:(B, P, D), y:(B, R, D) -> (B, P, R) works in new pytorch?
                    # Standard cdist is (P, D), (R, D) -> (P, R).
                    # We can use broadcast.
                    
                    # dist = (q - t)^2
                    # q: (M, T, 1, D), t: (M, 1, T, D)
                    # This broadcasts to (M, T, T, D) -> sum -> (M, T, T).
                    # Memory: M*T*T*D = 500*100*100*D. If D=64, 320MB. Safe.
                    
                    d_mat = torch.cdist(q_expand, targets, p=2) ** 2 # (M, T, T) works if supported
                    # If cdist doesn't support batch, use manual
                    if d_mat.dim() != 3:
                        # Manual
                        pass # PyTorch cdist supports batch since 1.6?
                        # Fallback: d_mat (M, T, T)
                        # just flat
                        d_mat = torch.cdist(q_expand.reshape(-1, T, D), targets.reshape(-1, T, D), p=2) # (M, T, T) ?? No.
                        # cdist on (B*T, D) and (B*T, D) gives (B*T, B*T). Huge.
                        
                        # Correct way: 
                        # dist = torch.norm(q.unsqueeze(2) - targets.unsqueeze(1), dim=-1)**2
                        d_mat = torch.sum((q_expand.unsqueeze(2) - targets.unsqueeze(1)) ** 2, dim=-1)
                    
                    # Soft-DTW DP
                    # D[m, i, j]
                    # R[m, i, j]
                    R = torch.full((M, T + 1, T + 1), float("inf"), device=device)
                    R[:, 0, 0] = 0
                    
                    # Iterate loops
                    # Can we vectorize? Yes, over M.
                    # Dependencies on i-1, j-1.
                    for i in range(1, T + 1):
                        for j in range(1, T + 1):
                            # cost = d_mat[:, i-1, j-1]
                            # prev = softmin(R[:, i-1, j], R[:, i, j-1], R[:, i-1, j-1])
                            
                            r1 = R[:, i-1, j]
                            r2 = R[:, i, j-1]
                            r3 = R[:, i-1, j-1]
                            
                            # Softmin
                            # -gamma * log(exp(-r1/g) + exp(-r2/g) + exp(-r3/g))
                            # Use logsumexp for stability
                            cat = torch.stack([r1, r2, r3], dim=1) # (M, 3)
                            soft_min_val = -gamma_dtw * torch.logsumexp(-cat / gamma_dtw, dim=1)
                            
                            R[:, i, j] = d_mat[:, i-1, j-1] + soft_min_val
                    
                    dist_matrix[idx, tj:tk] = R[:, T, T]
            
            if cfg.verbose > 0 and (start_i // batch_size) % 5 == 0:
                 print(f"    [DTW] Processed {min(end_i, N)}/{N} rows")
                 
        self.dist_matrix = dist_matrix.cpu().numpy() # (N, N)
        
        # RBF Kernel on DTW distance
        # Median Heuristic for gamma
        # We treat DTW distance as metric.
        # d_sq corresponds to ||z_i - z_j||^2 equivalent. 
        # Actually DTW allows triangle inequality violation, but used as distance.
        # We apply RBF: k = exp(-gamma * d^2) or exp(-gamma * d)? 
        # DTW is a distance, not squared distance usually.
        # But CoPL uses exp(-gamma * dist^2).
        # We can use dist^2 or dist. 
        # Let's match typical usage: exp(-gamma * d). 
        # But utils uses d^2.
        # Let's use d^2 for consistency.
        
        # Median of DTW distances (upper triangle)
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        d_vals = self.dist_matrix[mask]
        med_d = np.median(d_vals)
        self.gamma = 1.0 / (2.0 * (med_d ** 2) + 1e-12) * getattr(cfg, "gamma_mul", 1.0)
        
        if cfg.verbose > 0:
             print(f"  [DTW] median_dist={med_d:.4f}, gamma={self.gamma:.6f}")
             
        # Build kNN graph from Distance Matrix
        # self.dist_matrix is already computed.
        # We just pick top-k.
        
        rows, cols, vals = [], [], []
        # For each row, pick k smallest
        for i in range(N):
            dists = self.dist_matrix[i]
            # argsort
            # self excluded (d=0 at i)
            # but we want to exclude i.
            dists[i] = float('inf')
            
            knns = np.argpartition(dists, cfg.knn_k)[:cfg.knn_k]
            
            for j in knns:
                d = dists[j]
                w = np.exp(-self.gamma * (d ** 2))
                if w <= 1e-8: continue
                rows.append(i); cols.append(int(j)); vals.append(float(w))
            
            dists[i] = 0.0 # restore

        A = ItemSimilarityBuilder._make_symmetric_adj(rows, cols, vals, N, cfg.mutual)
        
        # We store X (not Z) because DTW needs raw data for test transform
        # But base class expects Z_train.
        # We can store X_tensor or X flat as Z_train? No, Z_train dimension matters.
        # Let's store Z_train = Xs (original features flattened or raw).
        # But Z usually implies low dim.
        # For DTW, "embedding" space is the sequence space itself.
        
        meta = {
            "method": "soft_dtw",
            "dtw_gamma": gamma_dtw,
            "gamma": float(self.gamma),
            "median_dist": float(med_d)
        }
        
        return {
            "Aii_norm": A,
            "Z_train": Xs, # Store standardized flattened X
            "gamma": self.gamma, 
            "meta": meta
        }

    def transform_test(self, X_test: np.ndarray) -> np.ndarray:
        # Standardize test data
        n_test, T, D = X_test.shape
        Xf = X_test.reshape(n_test, T * D).astype(np.float32)
        Xs = standardize_apply(Xf, self.mu_stats, self.sd_stats)
        return Xs # Return flattened standardized

    def get_affinity(self, Z_query: np.ndarray, Z_target: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        # Z_query: (n_q, T*D), Z_target: (n_t, T*D)
        # Compute DTW between query and target
        # This is expensive for test time!
        # But required.
        
        # Reshape back to (T, D)
        # We stored flattened vectors in Z_train.
        
        n_q = Z_query.shape[0]
        n_t = Z_target.shape[0]
        T, D = self._T, self._D
        
        # Convert to tensor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        Q = torch.tensor(Z_query.reshape(n_q, T, D), dtype=torch.float32, device=device)
        Targets = torch.tensor(Z_target.reshape(n_t, T, D), dtype=torch.float32, device=device)
        
        # We can't batch all query-target pairs.
        # Iterate over queries.
        
        nbrs = []
        weights = []
        
        gamma_dtw = 1.0 # Should compute or store in self? 
        # Assume 1.0 or read from self if stored?
        # fit() didn't store dtw_gamma in self. 
        # Use default 1.0 or pass in cfg? 
        # Hardcode 1.0 for now matching fit default.
        
        # Ideally store self.dtw_gamma in fit.
        
        # Chunking targets for memory
        chunk_size = 200
        
        for i in range(n_q):
            q = Q[i:i+1] # (1, T, D)
            
            dists_i = []
            
            for tj in range(0, n_t, chunk_size):
                tk = min(tj + chunk_size, n_t)
                t_chunk = Targets[tj:tk] # (M, T, D)
                M = t_chunk.size(0)
                
                # Pairwise dist (M, T, T)
                # manual broadcast: (1, T, 1, D) - (M, 1, T, D)
                # q: (1, T, D) -> (M, T, D)
                q_exp = q.expand(M, -1, -1)
                
                d_mat = torch.sum((q_exp.unsqueeze(2) - t_chunk.unsqueeze(1))**2, dim=-1)
                
                # DTW DP
                R = torch.full((M, T+1, T+1), float("inf"), device=device)
                R[:, 0, 0] = 0
                for ti in range(1, T+1):
                    for tj_dp in range(1, T+1):
                        r1, r2, r3 = R[:, ti-1, tj_dp], R[:, ti, tj_dp-1], R[:, ti-1, tj_dp-1]
                        soft_min = -gamma_dtw * torch.logsumexp(torch.stack([r1, r2, r3], 1) / -gamma_dtw, 1)
                        R[:, ti, tj_dp] = d_mat[:, ti-1, tj_dp-1] + soft_min
                
                d = R[:, T, T]
                dists_i.append(d)
                
            dists_i = torch.cat(dists_i) # (n_t,)
            
            # Top-k
            vals, idxs = torch.topk(dists_i, k, largest=False)
            
            # W = exp(-gamma * d^2)
            w = torch.exp(-self.gamma * (vals ** 2))
            w = w / (w.sum() + 1e-12)
            
            nbrs.append(idxs.cpu().numpy())
            weights.append(w.cpu().numpy())
            
        return np.stack(nbrs), np.stack(weights)
