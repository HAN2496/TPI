import numpy as np
import torch
from sklearn.model_selection import train_test_split

from ...data.splits import load_sequences
from .similarity import build_similarity


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


class CoPLGraphDataset:
    def __init__(self, cfg, sim_builder=None):
        self.cfg = cfg
        train_drivers = list(cfg.train_driver_names)
        self.train_drivers = train_drivers
        self.user_to_uid = {name: i for i, name in enumerate(train_drivers)}

        item_series_list, item_owner_uid_list = [], []
        self.per_user_items = {}
        item_offset = 0

        for uname in train_drivers:
            X, y = load_sequences(uname, list(cfg.features), cfg.time_range, cfg.downsample,
                                   smooth=cfg.smooth, smooth_cutoff=cfg.smooth_cutoff, smooth_order=cfg.smooth_order)
            uid = self.user_to_uid[uname]
            n = X.shape[0]
            item_ids = np.arange(item_offset, item_offset + n, dtype=np.int64)
            self.per_user_items[uid] = (item_ids, y.astype(np.int64))
            item_series_list.append(X.astype(np.float32))
            item_owner_uid_list.append(np.full((n,), uid, dtype=np.int64))
            item_offset += n
            if cfg.verbose > 0:
                print(f"  - {uname}: items={n}, pos={int(y.sum())}, neg={int((1-y).sum())}")

        self.item_series = np.concatenate(item_series_list, axis=0)
        self.item_owner_uid = np.concatenate(item_owner_uid_list, axis=0)
        self.n_users = len(train_drivers)
        self.n_items = self.item_series.shape[0]
        self.T = self.item_series.shape[1]
        self.obs_dim = self.item_series.shape[2]

        self.norm_mean = self.norm_std = None
        if cfg.normalize:
            self.norm_mean = self.item_series.mean(axis=(0, 1), keepdims=True)
            self.norm_std  = self.item_series.std(axis=(0, 1), keepdims=True) + 1e-6
            self.item_series = (self.item_series - self.norm_mean) / self.norm_std

        if cfg.verbose > 0:
            print(f"Total train users={self.n_users}, train items={self.n_items}, "
                  f"obs_dim={self.obs_dim}, T={self.T}")

        tr_u, tr_i, tr_y = [], [], []
        va_u, va_i, va_y = [], [], []
        for uid, (item_ids, y) in self.per_user_items.items():
            if len(np.unique(y)) < 2:
                tr_u.append(np.full_like(item_ids, uid))
                tr_i.append(item_ids)
                tr_y.append(y)
                continue
            it_tr, it_va, y_tr, y_va = train_test_split(
                item_ids, y, test_size=cfg.val_size, random_state=cfg.seed, stratify=y)
            tr_u.append(np.full_like(it_tr, uid)); tr_i.append(it_tr); tr_y.append(y_tr)
            va_u.append(np.full_like(it_va, uid)); va_i.append(it_va); va_y.append(y_va)

        self.tr_u = np.concatenate(tr_u)
        self.tr_i = np.concatenate(tr_i)
        self.tr_y = np.concatenate(tr_y)
        self.va_u = np.concatenate(va_u)
        self.va_i = np.concatenate(va_i)
        self.va_y = np.concatenate(va_y)

        tr_pos = (self.tr_y == 1)
        tr_neg = (self.tr_y == 0)

        pos_idx_bin = torch.as_tensor(np.vstack((self.tr_u[tr_pos], self.tr_i[tr_pos])), dtype=torch.long)
        neg_idx_bin = torch.as_tensor(np.vstack((self.tr_u[tr_neg], self.tr_i[tr_neg])), dtype=torch.long)

        self.Apos_bin = torch.sparse_coo_tensor(
            pos_idx_bin, torch.ones(pos_idx_bin.size(1)),
            size=(self.n_users, self.n_items)).coalesce()
        self.Aneg_bin = torch.sparse_coo_tensor(
            neg_idx_bin, torch.ones(neg_idx_bin.size(1)),
            size=(self.n_users, self.n_items)).coalesce()

        self.Apos_norm = normalize_bipartite_adj(self.Apos_bin)
        self.Aneg_norm = normalize_bipartite_adj(self.Aneg_bin)

        if cfg.verbose > 0:
            print("\n[Build] Item-item graph A_ii (train items only)...")

        if sim_builder is not None:
            self.sim_builder = sim_builder
            result = self.sim_builder.build_graph(self.item_series, cfg)
        else:
            self.sim_builder = build_similarity(cfg.similarity_method)
            result = self.sim_builder.fit(self.item_series, cfg)

        self.Aii_norm = result["Aii_norm"]
        self.Z_train = result["Z_train"]
        self.gamma = result["gamma"]
        self.Aii_meta = result["meta"]

    def attach_test_items(self, X_test, E_i_train, topk, device):
        Zt = self.sim_builder.transform_test(X_test)
        neigh_idx, neigh_w = self.sim_builder.get_affinity(Zt, self.Z_train, k=topk)
        E_i_train_cpu = E_i_train.detach().cpu().numpy()
        E_i_test = np.stack([
            (neigh_w[i][:, None] * E_i_train_cpu[neigh_idx[i]]).sum(axis=0)
            for i in range(Zt.shape[0])
        ])
        return torch.tensor(E_i_test, dtype=torch.float32, device=device), neigh_idx, neigh_w

    def adapt_test_user(self, y_test, neigh_idx, neigh_w, E_u_train, device):
        cfg = self.cfg
        y_test = y_test.astype(np.int64)
        n_train_items = self.Apos_bin.size(1)
        v = np.zeros((n_train_items,), dtype=np.float32)

        pos_mask = (y_test == 1)
        neg_mask = (y_test == 0)
        if pos_mask.any():
            np.add.at(v, neigh_idx[pos_mask].reshape(-1), neigh_w[pos_mask].reshape(-1))
        if cfg.adapt_use_neg and neg_mask.any():
            np.add.at(v, neigh_idx[neg_mask].reshape(-1), -cfg.adapt_neg_weight * neigh_w[neg_mask].reshape(-1))

        v_t = torch.tensor(v, dtype=torch.float32, device=device)
        c_u = torch.spmm(self.Apos_bin.to(device), v_t.unsqueeze(-1)).squeeze(-1)
        if cfg.adapt_use_neg and self.Aneg_bin is not None:
            c_u = c_u - torch.spmm(self.Aneg_bin.to(device), v_t.unsqueeze(-1)).squeeze(-1)

        temp = max(1e-6, cfg.adapt_user_softmax_temp)
        w_u = torch.softmax(c_u / temp, dim=0)
        if torch.isnan(w_u).any() or float(w_u.sum().item()) < 1e-6:
            w_u = torch.ones_like(w_u) / w_u.numel()

        e_u_test = (w_u.unsqueeze(-1) * E_u_train).sum(dim=0)
        return e_u_test, w_u.detach().cpu().numpy()

    def load_test_driver(self, test_driver_name: str):
        X_test, y_test = load_sequences(
            test_driver_name, list(self.cfg.features), self.cfg.time_range, self.cfg.downsample,
            smooth=self.cfg.smooth, smooth_cutoff=self.cfg.smooth_cutoff, smooth_order=self.cfg.smooth_order)
        if self.norm_mean is not None:
            X_test = (X_test - self.norm_mean) / self.norm_std
        return X_test, y_test.astype(np.int64)

    def to(self, device):
        self.Apos_norm = self.Apos_norm.to(device)
        self.Aneg_norm = self.Aneg_norm.to(device)
        self.Aii_norm = self.Aii_norm.to(device)
        return self
