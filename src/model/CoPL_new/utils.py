import numpy as np
import torch

from sklearn.model_selection import train_test_split

from src.utils.utils import _load_dataset_sequences
from src.model.CoPL_new.similarity import build_similarity


# =========================
# Low-level utilities
# =========================

def normalize_bipartite_adj(adj: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """D_u^-1/2 * A * D_i^-1/2 for (n_u x n_i) sparse COO."""
    adj = adj.coalesce()
    row_deg = torch.sparse.sum(adj, dim=1).to_dense()
    col_deg = torch.sparse.sum(adj, dim=0).to_dense()
    idx = adj.indices()
    val = adj.values()
    norm = torch.sqrt(row_deg[idx[0]] * col_deg[idx[1]] + eps)
    val = val / norm
    return torch.sparse_coo_tensor(idx, val, adj.size()).coalesce()

# =========================


# =========================
# CoPL Graph Dataset
# =========================
class CoPLGraphDataset:
    """
    CoPL 학습에 필요한 모든 전처리를 캡슐화하는 데이터셋 클래스.

    생성 시 다음을 수행:
      1. 드라이버(유저)별 시계열 데이터 로딩
      2. Train/Val 분할 (stratified)
      3. 이분 그래프 A_pos, A_neg 구축 및 정규화
      4. Item-Item 유사도 그래프 A_ii 구축 (PCA + kNN + RBF)

    인스턴스 속성으로 학습에 필요한 모든 텐서와 중간 상태를 보유.
    테스트 시 attach_test_items / adapt_test_user 메서드로 inductive 추론 지원.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        config = {"features": list(cfg.features)}

        # ----- 1. Load train drivers → global items + edges -----
        train_drivers = list(cfg.train_driver_names)
        self.train_drivers = train_drivers
        self.user_to_uid = {name: i for i, name in enumerate(train_drivers)}

        item_series_list = []
        item_owner_uid_list = []
        edge_u, edge_i, edge_y = [], [], []
        self.per_user_items = {}

        item_offset = 0
        for uname in train_drivers:
            X, y = _load_dataset_sequences(uname, cfg.time_range, cfg.downsample, config)
            uid = self.user_to_uid[uname]
            n = X.shape[0]

            item_ids = np.arange(item_offset, item_offset + n, dtype=np.int64)
            self.per_user_items[uid] = (item_ids, y.astype(np.int64))

            item_series_list.append(X.astype(np.float32))
            item_owner_uid_list.append(np.full((n,), uid, dtype=np.int64))

            edge_u.append(np.full((n,), uid, dtype=np.int64))
            edge_i.append(item_ids)
            edge_y.append(y.astype(np.int64))

            item_offset += n
            if cfg.verbose > 0:
                print(f"  - {uname}: items={n}, pos={int(y.sum())}, neg={int((1-y).sum())}")

        self.item_series = np.concatenate(item_series_list, axis=0)       # (n_items, T, d)
        self.item_owner_uid = np.concatenate(item_owner_uid_list, axis=0) # (n_items,)

        self.n_users = len(train_drivers)
        self.n_items = self.item_series.shape[0]
        self.T = self.item_series.shape[1]
        self.obs_dim = self.item_series.shape[2]

        if cfg.verbose > 0:
            print(f"Total train users={self.n_users}, train items={self.n_items}, "
                  f"obs_dim={self.obs_dim}, T={self.T}")

        # ----- 2. Per-user train/val split (stratified) -----
        tr_u, tr_i, tr_y = [], [], []
        va_u, va_i, va_y = [], [], []
        for uid, (item_ids, y) in self.per_user_items.items():
            if len(np.unique(y)) < 2:
                tr_u.append(np.full_like(item_ids, uid))
                tr_i.append(item_ids)
                tr_y.append(y)
                continue
            it_tr, it_va, y_tr, y_va = train_test_split(
                item_ids, y, test_size=cfg.val_size, random_state=cfg.seed, stratify=y
            )
            tr_u.append(np.full_like(it_tr, uid)); tr_i.append(it_tr); tr_y.append(y_tr)
            va_u.append(np.full_like(it_va, uid)); va_i.append(it_va); va_y.append(y_va)

        self.tr_u = np.concatenate(tr_u)
        self.tr_i = np.concatenate(tr_i)
        self.tr_y = np.concatenate(tr_y)
        self.va_u = np.concatenate(va_u)
        self.va_i = np.concatenate(va_i)
        self.va_y = np.concatenate(va_y)

        # ----- 3. Build A_pos / A_neg from TRAIN edges only -----
        tr_pos = (self.tr_y == 1)
        tr_neg = (self.tr_y == 0)

        pos_idx_bin = torch.as_tensor(
            np.vstack((self.tr_u[tr_pos], self.tr_i[tr_pos])), dtype=torch.long)
        neg_idx_bin = torch.as_tensor(
            np.vstack((self.tr_u[tr_neg], self.tr_i[tr_neg])), dtype=torch.long)

        self.Apos_bin = torch.sparse_coo_tensor(
            pos_idx_bin, torch.ones(pos_idx_bin.size(1)),
            size=(self.n_users, self.n_items)).coalesce()
        self.Aneg_bin = torch.sparse_coo_tensor(
            neg_idx_bin, torch.ones(neg_idx_bin.size(1)),
            size=(self.n_users, self.n_items)).coalesce()

        self.Apos_norm = normalize_bipartite_adj(self.Apos_bin)
        self.Aneg_norm = normalize_bipartite_adj(self.Aneg_bin)

        # ----- 4. Build A_ii (item-item similarity graph) -----
        if cfg.verbose > 0:
            print("\n[Build] Item-item graph A_ii (train items only)...")

        sim_method = getattr(cfg, "similarity_method", "pca")
        self.sim_builder = build_similarity(sim_method)
        result = self.sim_builder.fit(self.item_series, cfg)

        self.Aii_norm = result["Aii_norm"]
        self.Z_train = result["Z_train"]
        self.gamma = result["gamma"]
        self.Aii_meta = result["meta"]

    # ------------------------------------------------------------------
    # Test-time: attach test items to train embedding space
    # ------------------------------------------------------------------
    def attach_test_items(
        self,
        X_test: np.ndarray,         # (n_test, T, d)
        E_i_train: torch.Tensor,    # (n_train_items, d_emb)
        topk: int,
        device: torch.device,
    ):
        """
        테스트 아이템을 학습 아이템 임베딩 공간에 매핑 (top-k 가중 평균).
        Returns:
          E_i_test:  (n_test, d_emb) torch tensor
          neigh_idx: (n_test, topk) np int
          neigh_w:   (n_test, topk) np float
        """
        # sim_builder 전략에 따라 Z_test 생성
        Zt = self.sim_builder.transform_test(X_test)

        # sim_builder 전략에 따라 affinity 계산 (RBF or Cosine)
        neigh_idx, neigh_w = self.sim_builder.get_affinity(Zt, self.Z_train, k=topk)

        # E_i_test = weighted average of train embeddings
        # neigh_w: (n_test, topk), neigh_idx: (n_test, topk)
        E_i_train_cpu = E_i_train.detach().cpu().numpy()
        E_i_test = []
        for i in range(Zt.shape[0]):
            # w[i]: (topk,), nbr[i]: (topk,)
            # E_i_train_cpu[nbr[i]]: (topk, d_emb)
            # Weighted average
            e = (neigh_w[i][:, None] * E_i_train_cpu[neigh_idx[i]]).sum(axis=0)
            E_i_test.append(e)
        E_i_test = torch.tensor(np.stack(E_i_test), dtype=torch.float32, device=device)
        return E_i_test, neigh_idx, neigh_w

    # ------------------------------------------------------------------
    # Test-time: adapt test user embedding via item-item bridge
    # ------------------------------------------------------------------
    def adapt_test_user(
        self,
        y_test: np.ndarray,          # (n_test,)
        neigh_idx: np.ndarray,       # (n_test, topk)
        neigh_w: np.ndarray,         # (n_test, topk)
        E_u_train: torch.Tensor,     # (n_users, d_emb)
        device: torch.device,
    ):
        """
        Item-Item Bridge를 통해 테스트 유저 임베딩 생성.
        Returns:
          e_u_test: (d_emb,) torch tensor
          w_u:      (n_users,) np array — 각 학습 유저의 가중치
        """
        cfg = self.cfg
        y_test = y_test.astype(np.int64)
        n_train_items = self.Apos_bin.size(1)

        # Accumulate item votes
        v = np.zeros((n_train_items,), dtype=np.float32)
        pos_mask = (y_test == 1)
        neg_mask = (y_test == 0)

        if pos_mask.any():
            idx_pos = neigh_idx[pos_mask].reshape(-1)
            w_pos = neigh_w[pos_mask].reshape(-1)
            np.add.at(v, idx_pos, w_pos)

        if cfg.adapt_use_neg and neg_mask.any():
            idx_neg = neigh_idx[neg_mask].reshape(-1)
            w_neg = neigh_w[neg_mask].reshape(-1)
            np.add.at(v, idx_neg, -cfg.adapt_neg_weight * w_neg)

        v_t = torch.tensor(v, dtype=torch.float32, device=device)

        # User scores from train edges
        c_u = torch.spmm(
            self.Apos_bin.to(device), v_t.unsqueeze(-1)).squeeze(-1)

        if cfg.adapt_use_neg and self.Aneg_bin is not None:
            c_u = c_u - torch.spmm(
                self.Aneg_bin.to(device), v_t.unsqueeze(-1)).squeeze(-1)

        # Softmax weighting
        temp = max(1e-6, cfg.adapt_user_softmax_temp)
        w_u = torch.softmax(c_u / temp, dim=0)

        if torch.isnan(w_u).any() or float(w_u.sum().item()) < 1e-6:
            w_u = torch.ones_like(w_u) / w_u.numel()

        e_u_test = (w_u.unsqueeze(-1) * E_u_train).sum(dim=0)
        return e_u_test, w_u.detach().cpu().numpy()

    # ------------------------------------------------------------------
    # Convenience: load test driver data
    # ------------------------------------------------------------------
    def load_test_driver(self, test_driver_name: str):
        """
        테스트 드라이버의 시계열 데이터 로딩.
        Returns: X_test (n_test, T, d), y_test (n_test,)
        """
        config = {"features": list(self.cfg.features)}
        X_test, y_test = _load_dataset_sequences(
            test_driver_name, self.cfg.time_range, self.cfg.downsample, config)
        return X_test, y_test.astype(np.int64)

    # ------------------------------------------------------------------
    # Move adjacencies to device
    # ------------------------------------------------------------------
    def to(self, device: torch.device):
        """인접행렬들을 지정한 디바이스로 이동."""
        self.Apos_norm = self.Apos_norm.to(device)
        self.Aneg_norm = self.Aneg_norm.to(device)
        self.Aii_norm = self.Aii_norm.to(device)
        return self