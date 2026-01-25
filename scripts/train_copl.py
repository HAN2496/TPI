# scripts/train_copl.py
import os
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from src.utils.utils import _load_dataset_sequences
from src.model.CoPL_new.gcf import CoPLGCF  # <-- 너가 수정해둔 CoPLGCF(아이템-아이템 + pointwise) 사용

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# =========================
# Config
# =========================
@dataclass
class CFG:
    # data
    features: tuple = ("IMU_VerAccelVal", "Bounce_rate_6D", "Pitch_rate_6D", "IMU_LongAccelVal")
    train_driver_names: tuple = ("김진명", "김태근", "조현석", "한규택", "박재일", "이지환")
    test_driver_name: str = "강신길"
    time_range: tuple = (5, 7)
    downsample: int = 5

    # item-item graph (검증 결과 반영)
    pca_dim: int = 2
    gamma_mul: float = 8.0              # gamma = median_gamma * gamma_mul
    knn_k: int = 5
    mutual: bool = False

    # GCF
    hidden_dim: int = 128
    gcf_layers: int = 2
    gcf_dropout: float = 0.2
    item_item_weight: float = 1.0
    gcf_lr: float = 5e-3
    gcf_weight_decay: float = 0.0
    gcf_epochs: int = 100
    gcf_lambda_reg: float = 1e-4

    # RM
    rm_hidden: int = 64
    rm_mlp_hidden: int = 64
    rm_lr: float = 3e-4
    rm_weight_decay: float = 0.0
    rm_epochs: int = 200
    rm_batch_size: int = 256
    rm_lambda_reg: float = 1e-6

    # split
    val_size: float = 0.1
    seed: int = 42

    # adaptation for test user
    adapt_use_neg: bool = True
    adapt_neg_weight: float = 1.0
    adapt_user_softmax_temp: float = 1.0
    attach_topk_items: int = 20  # test item -> train items topk for embedding

    # viz
    tsne_max_items_per_driver: int = 400
    tsne_perplexity: int = 30

    # system
    device: str = "cuda"
    save_root: str = "artifacts/copl"
    verbose: int = 1


# =========================
# Utilities
# =========================
def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def standardize_fit(X: np.ndarray):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-6
    return mu, sd


def standardize_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray):
    return (X - mu) / sd


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


def normalize_square_adj(adj: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """D^-1/2 * A * D^-1/2 for (n x n) sparse COO."""
    adj = adj.coalesce()
    deg = torch.sparse.sum(adj, dim=1).to_dense()
    idx = adj.indices()
    val = adj.values()
    norm = torch.sqrt(deg[idx[0]] * deg[idx[1]] + eps)
    val = val / norm
    return torch.sparse_coo_tensor(idx, val, adj.size()).coalesce()


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


def plot_roc(y_true, y_score, save_path: Path, title: str):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.0
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_reward_scatter(y_true, y_prob, save_path: Path, title: str):
    idx = np.arange(len(y_true))
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    m0 = (y_true == 0)
    m1 = (y_true == 1)

    plt.figure(figsize=(10, 5))
    plt.scatter(idx[m0], y_prob[m0], s=10, alpha=0.5, label="Bad(0)")
    plt.scatter(idx[m1], y_prob[m1], s=10, alpha=0.7, label="Good(1)")
    plt.axhline(0.5, linestyle="--", alpha=0.5)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Sample index")
    plt.ylabel("Predicted probability")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def tsne_plot(cfg: CFG, emb: np.ndarray, labels: list[str], save_path: Path, title: str, star_mask=None):
    # labels -> int
    uniq = sorted(list(set(labels)))
    lab2i = {l: i for i, l in enumerate(uniq)}
    y = np.array([lab2i[l] for l in labels], dtype=np.int64)

    n = emb.shape[0]
    perp = min(cfg.tsne_perplexity, max(5, (n - 1) // 3))
    Z2 = TSNE(n_components=2, random_state=cfg.seed, perplexity=perp).fit_transform(emb)

    plt.figure(figsize=(9, 7))
    for l in uniq:
        m = np.array([x == l for x in labels])
        if star_mask is not None and m.any() and star_mask[m].all():
            plt.scatter(Z2[m, 0], Z2[m, 1], marker="*", s=120, alpha=1.0, label=l, edgecolors="black")
        else:
            plt.scatter(Z2[m, 0], Z2[m, 1], s=18, alpha=0.7, label=l)

    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# =========================
# Item-item graph builder (train items)
# =========================
def build_train_similarity_space(cfg: CFG, item_series: np.ndarray):
    """
    item_series: (n_items, T, d)
    returns:
      mu, sd, pca, Z_train, gamma
      Aii_norm (train items only)
    """
    N, T, D = item_series.shape
    X = item_series.reshape(N, T * D).astype(np.float32)

    mu, sd = standardize_fit(X)
    Xs = standardize_apply(X, mu, sd)

    pca = PCA(n_components=min(cfg.pca_dim, Xs.shape[1]), random_state=cfg.seed)
    Z = pca.fit_transform(Xs)

    gamma_med = median_heuristic_gamma(Z, seed=cfg.seed)
    gamma = gamma_med * cfg.gamma_mul

    # kNN on Z to build Aii among train items
    nnbrs = NearestNeighbors(n_neighbors=min(cfg.knn_k + 1, N), metric="euclidean")
    nnbrs.fit(Z)
    dist, nbr = nnbrs.kneighbors(Z, return_distance=True)

    rows, cols, vals = [], [], []
    for i in range(N):
        for j, dij in zip(nbr[i], dist[i]):
            if j == i:
                continue
            w = float(np.exp(-gamma * (dij ** 2)))
            if w <= 0:
                continue
            rows.append(i); cols.append(int(j)); vals.append(w)

    if cfg.mutual:
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

    Aii_norm = normalize_square_adj(A)

    meta = {
        "pca_dim": cfg.pca_dim,
        "explained_var_sum": float(pca.explained_variance_ratio_.sum()),
        "gamma_med": gamma_med,
        "gamma_mul": cfg.gamma_mul,
        "gamma": gamma,
        "knn_k": cfg.knn_k,
        "mutual": cfg.mutual,
    }
    return mu, sd, pca, Z, gamma, Aii_norm, meta


def attach_test_items_to_train(
    X_test: np.ndarray,         # (n_test, T, d)
    mu: np.ndarray,
    sd: np.ndarray,
    pca: PCA,
    Z_train: np.ndarray,
    gamma: float,
    E_i_train: torch.Tensor,    # (n_train_items, d)
    topk: int,
    device: torch.device
):
    """
    Map each test item to a train-item embedding by similarity-weighted top-k neighbors in PCA space.
    returns:
      E_i_test: (n_test, d) torch tensor
      neigh_idx: (n_test, topk) np int
      neigh_w:   (n_test, topk) np float
    """
    n_test, T, D = X_test.shape
    Xf = X_test.reshape(n_test, T * D).astype(np.float32)
    Xs = standardize_apply(Xf, mu, sd)
    Zt = pca.transform(Xs)

    nnbrs = NearestNeighbors(n_neighbors=min(topk, Z_train.shape[0]), metric="euclidean")
    nnbrs.fit(Z_train)
    dist, nbr = nnbrs.kneighbors(Zt, return_distance=True)  # (n_test, topk)

    w = np.exp(-gamma * (dist ** 2)).astype(np.float32)
    w_sum = w.sum(axis=1, keepdims=True) + 1e-8
    w = w / w_sum

    # build E_i_test = sum_k w * E_i_train[nbr]
    E_i_test = []
    E_i_train_cpu = E_i_train.detach().cpu().numpy()
    for i in range(n_test):
        e = (w[i][:, None] * E_i_train_cpu[nbr[i]]).sum(axis=0)
        E_i_test.append(e)
    E_i_test = torch.tensor(np.stack(E_i_test), dtype=torch.float32, device=device)
    return E_i_test, nbr, w


def adapt_test_user_embedding(
    cfg: CFG,
    y_test: np.ndarray,              # (n_test,)
    neigh_idx: np.ndarray,           # (n_test, topk) neighbors in train items
    neigh_w: np.ndarray,             # (n_test, topk) normalized weights
    Apos_bin: torch.Tensor,          # (n_users, n_items) binary sparse
    Aneg_bin: torch.Tensor | None,   # optional
    E_u_train: torch.Tensor,         # (n_users, d)
    device: torch.device
):
    """
    Build test user embedding via item-item bridge:
      v_item[j] = sum_{t in pos} w(t->j)  -  eta * sum_{t in neg} w(t->j)
      c_u = sum_{j in N_u^+} v_item[j]    (and optionally neg term)
      w_u = softmax(c_u / temp)
      e_u_test = sum_u w_u * e_u
    """
    y_test = y_test.astype(np.int64)
    topk = neigh_idx.shape[1]
    n_train_items = Apos_bin.size(1)

    # accumulate v_item on train items (numpy -> torch)
    v = np.zeros((n_train_items,), dtype=np.float32)

    pos_mask = (y_test == 1)
    neg_mask = (y_test == 0)

    # pos contribution
    if pos_mask.any():
        idx_pos = neigh_idx[pos_mask].reshape(-1)
        w_pos = neigh_w[pos_mask].reshape(-1)
        np.add.at(v, idx_pos, w_pos)

    # neg contribution (optional)
    if cfg.adapt_use_neg and neg_mask.any():
        idx_neg = neigh_idx[neg_mask].reshape(-1)
        w_neg = neigh_w[neg_mask].reshape(-1)
        np.add.at(v, idx_neg, -cfg.adapt_neg_weight * w_neg)

    v_t = torch.tensor(v, dtype=torch.float32, device=device)

    # user scores from positive train edges
    c_u = torch.spmm(Apos_bin.to(device), v_t.unsqueeze(-1)).squeeze(-1)  # (n_users,)

    # optionally include neg edges too (push away)
    if cfg.adapt_use_neg and Aneg_bin is not None:
        c_u = c_u - torch.spmm(Aneg_bin.to(device), v_t.unsqueeze(-1)).squeeze(-1)

    # softmax weighting
    temp = max(1e-6, cfg.adapt_user_softmax_temp)
    w_u = torch.softmax(c_u / temp, dim=0)  # (n_users,)

    # if everything is flat -> fallback
    if torch.isnan(w_u).any() or float(w_u.sum().item()) < 1e-6:
        w_u = torch.ones_like(w_u) / w_u.numel()

    e_u_test = (w_u.unsqueeze(-1) * E_u_train).sum(dim=0)  # (d,)
    return e_u_test, w_u.detach().cpu().numpy()


# =========================
# Reward Model (time-series, Linear encoder)
# =========================
class TSRewardModel(nn.Module):
    """
    time-series RM:
      - per-timestep linear projection
      - user conditioning via linear projection
      - nonlinearity + mean pooling
      - head -> logit
    """
    def __init__(self, obs_dim: int, user_dim: int, hidden: int = 128, mlp_hidden: int = 128):
        super().__init__()
        self.obs_proj = nn.Linear(obs_dim, hidden)
        self.user_proj = nn.Linear(user_dim, hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, user_emb: torch.Tensor, obs: torch.Tensor):
        """
        user_emb: (B, user_dim)
        obs:      (B, T, obs_dim)
        """
        # project
        h_obs = self.obs_proj(obs)  # (B,T,H)
        h_u = self.user_proj(user_emb).unsqueeze(1)  # (B,1,H)
        h = torch.tanh(h_obs + h_u)  # (B,T,H)
        h = h.mean(dim=1)            # (B,H)
        h = self.mlp(h)              # (B,H)
        return self.head(h).squeeze(-1)  # (B,)


def weighted_bce_logits(logits, labels, pos_weight=None):
    labels = labels.float()
    if pos_weight is not None:
        return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
    return F.binary_cross_entropy_with_logits(logits, labels)


class RMEdgeDataset(torch.utils.data.Dataset):
    def __init__(self, uids, iids, labels, item_series):
        self.uids = uids.astype(np.int64)
        self.iids = iids.astype(np.int64)
        self.labels = labels.astype(np.int64)
        self.item_series = item_series.astype(np.float32)

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        u = self.uids[idx]
        i = self.iids[idx]
        y = self.labels[idx]
        obs = self.item_series[i]  # (T,d)
        return u, obs, y


def rm_collate(batch):
    u = torch.tensor([b[0] for b in batch], dtype=torch.long)
    obs = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.float32)
    y = torch.tensor([b[2] for b in batch], dtype=torch.float32)
    return u, obs, y


# =========================
# Main
# =========================
def run_copl_training(cfg: CFG):
    seed_all(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    log_dir = None
    if cfg.verbose > 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(cfg.save_root) / timestamp
        log_dir.mkdir(parents=True, exist_ok=True)

        with open(log_dir / "cfg.json", "w", encoding="utf-8") as f:
            json.dump(cfg.__dict__, f, ensure_ascii=False, indent=2)

        print("Log dir:", log_dir)
        print("\n[1] Loading train drivers...")

    # -------------------------
    # 1) Load train drivers -> global items + edges
    # -------------------------
    config = {"features": list(cfg.features)}

    train_drivers = list(cfg.train_driver_names)
    test_driver = cfg.test_driver_name

    user_to_uid = {name: i for i, name in enumerate(train_drivers)}

    item_series_list = []
    item_owner_uid = []  # item -> which train driver
    edge_u = []
    edge_i = []
    edge_y = []

    per_user_items = {}  # uid -> (item_ids, y)

    item_offset = 0
    for uname in train_drivers:
        X, y = _load_dataset_sequences(uname, cfg.time_range, cfg.downsample, config)
        uid = user_to_uid[uname]
        n = X.shape[0]

        item_ids = np.arange(item_offset, item_offset + n, dtype=np.int64)
        per_user_items[uid] = (item_ids, y.astype(np.int64))

        item_series_list.append(X.astype(np.float32))
        item_owner_uid.append(np.full((n,), uid, dtype=np.int64))

        edge_u.append(np.full((n,), uid, dtype=np.int64))
        edge_i.append(item_ids)
        edge_y.append(y.astype(np.int64))

        item_offset += n
        if cfg.verbose > 0:
            print(f"  - {uname}: items={n}, pos={int(y.sum())}, neg={int((1-y).sum())}")

    item_series = np.concatenate(item_series_list, axis=0)          # (n_items,T,d)
    item_owner_uid = np.concatenate(item_owner_uid, axis=0)         # (n_items,)
    edge_u = np.concatenate(edge_u, axis=0)
    edge_i = np.concatenate(edge_i, axis=0)
    edge_y = np.concatenate(edge_y, axis=0)

    n_users = len(train_drivers)
    n_items = item_series.shape[0]
    T, obs_dim = item_series.shape[1], item_series.shape[2]
    
    if cfg.verbose > 0:
        print(f"Total train users={n_users}, train items={n_items}, obs_dim={obs_dim}, T={T}")

    # -------------------------
    # 2) per-user train/val split (stratified)
    # -------------------------
    tr_u, tr_i, tr_y = [], [], []
    va_u, va_i, va_y = [], [], []
    for uid, (item_ids, y) in per_user_items.items():
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

    tr_u = np.concatenate(tr_u); tr_i = np.concatenate(tr_i); tr_y = np.concatenate(tr_y)
    va_u = np.concatenate(va_u); va_i = np.concatenate(va_i); va_y = np.concatenate(va_y)

    # -------------------------
    # 3) Build A_pos/A_neg from TRAIN edges only
    # -------------------------
    tr_pos = (tr_y == 1)
    tr_neg = (tr_y == 0)

    # binary adjacency for adaptation (NOT normalized)
    pos_idx_bin = torch.as_tensor(np.vstack((tr_u[tr_pos], tr_i[tr_pos])), dtype=torch.long)
    neg_idx_bin = torch.as_tensor(np.vstack((tr_u[tr_neg], tr_i[tr_neg])), dtype=torch.long)
    Apos_bin = torch.sparse_coo_tensor(pos_idx_bin, torch.ones(pos_idx_bin.size(1)), size=(n_users, n_items)).coalesce()
    Aneg_bin = torch.sparse_coo_tensor(neg_idx_bin, torch.ones(neg_idx_bin.size(1)), size=(n_users, n_items)).coalesce()

    # normalized adjacency for GCF propagation
    Apos_norm = normalize_bipartite_adj(Apos_bin).to(device)
    Aneg_norm = normalize_bipartite_adj(Aneg_bin).to(device)

    # -------------------------
    # 4) Build A_ii (검증 설정 그대로)
    # -------------------------
    if cfg.verbose > 0:
        print("\n[2] Building item-item graph A_ii (train items only)...")
    
    mu, sd, pca, Z_train, gamma, Aii_norm, Aii_meta = build_train_similarity_space(cfg, item_series)
    Aii_norm = Aii_norm.to(device)

    if cfg.verbose > 0:
        with open(log_dir / "Aii_meta.json", "w", encoding="utf-8") as f:
            json.dump(Aii_meta, f, ensure_ascii=False, indent=2)
        print("Aii meta:", Aii_meta)

    # -------------------------
    # 5) Train GCF (pointwise BCE, imbalance via pos_weight)
    # -------------------------
    if cfg.verbose > 0:
        print("\n[3] Training GCF (pointwise BCE)...")
    
    gcf = CoPLGCF(
        n_u=n_users,
        n_i=n_items,
        d=cfg.hidden_dim,
        pos_adj_norm=Apos_norm,
        neg_adj_norm=Aneg_norm,
        dropout=cfg.gcf_dropout,
        l=cfg.gcf_layers,
        item_item_adj_norm=Aii_norm,
        item_item_weight=cfg.item_item_weight,
    ).to(device)

    opt_gcf = torch.optim.AdamW(gcf.parameters(), lr=cfg.gcf_lr, weight_decay=cfg.gcf_weight_decay)

    tr_u_t = torch.tensor(tr_u, dtype=torch.long, device=device)
    tr_i_t = torch.tensor(tr_i, dtype=torch.long, device=device)
    tr_y_t = torch.tensor(tr_y, dtype=torch.float32, device=device)
    va_u_t = torch.tensor(va_u, dtype=torch.long, device=device)
    va_i_t = torch.tensor(va_i, dtype=torch.long, device=device)
    va_y_np = va_y.astype(np.int64)

    pos_cnt = int(tr_y.sum())
    neg_cnt = int((1 - tr_y).sum())
    pos_weight = torch.tensor([neg_cnt / max(1, pos_cnt)], dtype=torch.float32, device=device)

    best_auc = -1.0
    best_gcf_state_dict = None

    for epoch in range(cfg.gcf_epochs):
        gcf.train()
        opt_gcf.zero_grad()
        loss, _ = gcf.forward_pointwise(
            tr_u_t, tr_i_t, tr_y_t,
            pos_weight=pos_weight,
            sample_weight=None,
            test=False,
            lambda_reg=cfg.gcf_lambda_reg,
        )
        loss.backward()
        opt_gcf.step()

        gcf.eval()
        with torch.no_grad():
            _, val_logits = gcf.forward_pointwise(
                va_u_t, va_i_t, torch.tensor(va_y_np, dtype=torch.float32, device=device),
                pos_weight=None,
                sample_weight=None,
                test=True,
                lambda_reg=0.0,
            )
            val_prob = torch.sigmoid(val_logits).detach().cpu().numpy()
            val_auc = roc_auc_score(va_y_np, val_prob) if len(np.unique(va_y_np)) > 1 else 0.0

        if val_auc > best_auc:
            best_auc = val_auc
            best_gcf_state_dict = {k: v.cpu() for k, v in gcf.state_dict().items()}

        if cfg.verbose > 0 and (epoch % 10 == 0 or epoch == cfg.gcf_epochs - 1):
            print(f"  [GCF] epoch={epoch:03d} loss={float(loss.item()):.4f} val_auc={val_auc:.4f} best={best_auc:.4f}")

    if cfg.verbose > 0:
        print("Best GCF val AUC:", best_auc)
        if best_gcf_state_dict is not None:
            torch.save(best_gcf_state_dict, log_dir / "best_gcf.pt")
            print("Saved:", log_dir / "best_gcf.pt")

    # load best & extract embeddings
    if best_gcf_state_dict is not None:
        gcf.load_state_dict({k: v.to(device) for k, v in best_gcf_state_dict.items()})
    
    gcf.eval()
    with torch.no_grad():
        E_u_train, E_i_train = gcf.encode_graph(test=True)  # (n_users,d), (n_items,d)

    # -------------------------
    # 6) Train Reward Model (RM) using fixed user embeddings
    # -------------------------
    if cfg.verbose > 0:
        print("\n[4] Training RM (time-series, pointwise BCE)...")
    
    rm = TSRewardModel(obs_dim=obs_dim, user_dim=E_u_train.shape[1], hidden=cfg.rm_hidden, mlp_hidden=cfg.rm_mlp_hidden).to(device)
    opt_rm = torch.optim.AdamW(rm.parameters(), lr=cfg.rm_lr, weight_decay=cfg.rm_weight_decay)

    # RM datasets use same split (tr_u,tr_i,tr_y) and (va_u,va_i,va_y)
    tr_ds = RMEdgeDataset(tr_u, tr_i, tr_y, item_series)
    va_ds = RMEdgeDataset(va_u, va_i, va_y, item_series)

    tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=cfg.rm_batch_size, shuffle=True, collate_fn=rm_collate, drop_last=True)
    va_loader = torch.utils.data.DataLoader(va_ds, batch_size=cfg.rm_batch_size, shuffle=False, collate_fn=rm_collate, drop_last=False)

    pos_cnt_rm = int(tr_y.sum())
    neg_cnt_rm = int((1 - tr_y).sum())
    pos_weight_rm = torch.tensor([neg_cnt_rm / max(1, pos_cnt_rm)], dtype=torch.float32, device=device)

    best_rm_auc = -1.0
    best_rm_state_dict = None

    for epoch in range(cfg.rm_epochs):
        rm.train()
        tr_loss_accum = 0.0
        n_seen = 0

        for uids_b, obs_b, y_b in tr_loader:
            uids_b = uids_b.to(device)
            obs_b = obs_b.to(device)
            y_b = y_b.to(device)

            user_emb = E_u_train[uids_b]  # fixed user embedding

            logits = rm(user_emb, obs_b)
            loss_bce = weighted_bce_logits(logits, y_b, pos_weight=pos_weight_rm)
            loss_reg = (user_emb.norm(2).pow(2).mean())  # small reg (optional)
            loss = loss_bce + cfg.rm_lambda_reg * loss_reg

            opt_rm.zero_grad()
            loss.backward()
            opt_rm.step()

            tr_loss_accum += float(loss.item()) * len(uids_b)
            n_seen += len(uids_b)

        rm.eval()
        all_prob = []
        all_y = []
        with torch.no_grad():
            for uids_b, obs_b, y_b in va_loader:
                uids_b = uids_b.to(device)
                obs_b = obs_b.to(device)
                y_b = y_b.to(device)
                user_emb = E_u_train[uids_b]
                logits = rm(user_emb, obs_b)
                prob = torch.sigmoid(logits).detach().cpu().numpy()
                all_prob.append(prob)
                all_y.append(y_b.detach().cpu().numpy())

        all_prob = np.concatenate(all_prob) if all_prob else np.array([])
        all_y = np.concatenate(all_y).astype(np.int64) if all_y else np.array([])
        val_auc = roc_auc_score(all_y, all_prob) if len(np.unique(all_y)) > 1 else 0.0

        if val_auc > best_rm_auc:
            best_rm_auc = val_auc
            best_rm_state_dict = {k: v.cpu() for k, v in rm.state_dict().items()}

        if cfg.verbose > 0 and (epoch % 5 == 0 or epoch == cfg.rm_epochs - 1):
            print(f"  [RM ] epoch={epoch:03d} train_loss={tr_loss_accum/max(1,n_seen):.4f} val_auc={val_auc:.4f} best={best_rm_auc:.4f}")

    if cfg.verbose > 0:
        print("Best RM val AUC:", best_rm_auc)
        if best_rm_state_dict is not None:
            torch.save(best_rm_state_dict, log_dir / "best_rm.pt")
            print("Saved:", log_dir / "best_rm.pt")

    if best_rm_state_dict is not None:
        rm.load_state_dict({k: v.to(device) for k, v in best_rm_state_dict.items()})
    rm.eval()

    # -------------------------
    # 7) Test driver evaluation (AUROC + plots) using adaptation
    # -------------------------
    if cfg.verbose > 0:
        print("\n[5] Evaluating on test driver:", test_driver)
    
    X_test, y_test = _load_dataset_sequences(test_driver, cfg.time_range, cfg.downsample, config)
    y_test = y_test.astype(np.int64)

    # attach test items into train item embedding space
    E_i_test, neigh_idx, neigh_w = attach_test_items_to_train(
        X_test=X_test.astype(np.float32),
        mu=mu, sd=sd, pca=pca,
        Z_train=Z_train,
        gamma=gamma,
        E_i_train=E_i_train,
        topk=cfg.attach_topk_items,
        device=device,
    )

    # adapt test user embedding using item-item bridge + train pos edges
    e_u_test, w_u = adapt_test_user_embedding(
        cfg=cfg,
        y_test=y_test,
        neigh_idx=neigh_idx,
        neigh_w=neigh_w,
        Apos_bin=Apos_bin,
        Aneg_bin=Aneg_bin,
        E_u_train=E_u_train,
        device=device,
    )

    # RM prediction on test
    with torch.no_grad():
        user_emb_test = e_u_test.unsqueeze(0).repeat(X_test.shape[0], 1)  # (n_test, d)
        obs_test = torch.tensor(X_test.astype(np.float32), dtype=torch.float32, device=device)
        logits_test = rm(user_emb_test, obs_test)
        prob_test = torch.sigmoid(logits_test).detach().cpu().numpy()

    test_auc = roc_auc_score(y_test, prob_test) if len(np.unique(y_test)) > 1 else 0.0
    
    if cfg.verbose > 0:
        print(f"Test AUROC (RM): {test_auc:.4f}")

        plot_roc(
            y_true=y_test,
            y_score=prob_test,
            save_path=log_dir / f"roc_test_{test_driver}.png",
            title=f"ROC (RM) - Test Driver {test_driver} (AUC={test_auc:.4f})"
        )

        plot_reward_scatter(
            y_true=y_test,
            y_prob=prob_test,
            save_path=log_dir / f"scatter_test_{test_driver}.png",
            title=f"RM Prediction Scatter - Test Driver {test_driver}"
        )

    # -------------------------
    # 8) t-SNE visualization (users + items)
    # -------------------------
    if cfg.verbose > 0:
        print("\n[6] t-SNE visualization...")

        # users: train users + test user (star)
        user_emb_all = torch.cat([E_u_train, e_u_test.unsqueeze(0)], dim=0).detach().cpu().numpy()
        user_labels = train_drivers + [f"{test_driver} (Test)"]
        star_mask = np.array([False] * len(train_drivers) + [True])

        tsne_plot(
            cfg=cfg,
            emb=user_emb_all,
            labels=user_labels,
            save_path=log_dir / "tsne_users.png",
            title="t-SNE: User Embeddings (Train + Test)",
            star_mask=star_mask,
        )

        # items: sample train items per driver + all test items (or sample)
        rng = np.random.default_rng(cfg.seed)
        train_item_idx = []
        train_item_labels = []
        for uid, uname in enumerate(train_drivers):
            ids = np.where(item_owner_uid == uid)[0]
            if len(ids) == 0:
                continue
            take = min(cfg.tsne_max_items_per_driver, len(ids))
            picked = rng.choice(ids, size=take, replace=False)
            train_item_idx.append(picked)
            train_item_labels += [f"{uname} (TrainItem)"] * take

        train_item_idx = np.concatenate(train_item_idx) if train_item_idx else np.array([], dtype=np.int64)
        E_i_train_sample = E_i_train[torch.tensor(train_item_idx, dtype=torch.long, device=device)].detach().cpu().numpy()

        # test items: 샘플링(너무 많으면)
        n_test_items = X_test.shape[0]
        take_test = min(cfg.tsne_max_items_per_driver, n_test_items)
        test_pick = rng.choice(np.arange(n_test_items), size=take_test, replace=False) if n_test_items > take_test else np.arange(n_test_items)
        E_i_test_sample = E_i_test[torch.tensor(test_pick, dtype=torch.long, device=device)].detach().cpu().numpy()
        test_item_labels = [f"{test_driver} (TestItem)"] * len(test_pick)

        item_emb_all = np.concatenate([E_i_train_sample, E_i_test_sample], axis=0)
        item_labels_all = train_item_labels + test_item_labels

        star_mask_items = np.array([False] * len(train_item_labels) + [True] * len(test_item_labels))
        tsne_plot(
            cfg=cfg,
            emb=item_emb_all,
            labels=item_labels_all,
            save_path=log_dir / "tsne_items.png",
            title="t-SNE: Item Embeddings (Train Items + Attached Test Items)",
            star_mask=star_mask_items,
        )

        # -------------------------
        # Save summary
        # -------------------------
        summary = {
            "gcf_best_val_auc": float(best_auc),
            "rm_best_val_auc": float(best_rm_auc),
            "rm_test_auc": float(test_auc),
            "Aii_meta": Aii_meta,
            "test_user_weight_top5": sorted(
                [(train_drivers[i], float(w_u[i])) for i in range(len(train_drivers))],
                key=lambda x: -x[1]
            )[:5],
        }
        with open(log_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print("\nSaved artifacts to:", log_dir)
        print("Summary:", summary)

    return test_auc


if __name__ == "__main__":
    cfg = CFG()
    run_copl_training(cfg)
