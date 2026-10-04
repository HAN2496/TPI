"""Item-item graph sparsification rules (plan 01, Section 3) and graph statistics.

All rules produce the symmetric, degree-normalized sparse adjacency that CoPLGCF consumes.
Weights follow the encoder's metric: exp(sim / temperature) for cosine latents, exp(-gamma d^2)
otherwise.  `build_graph` works from the latent matrix only, so a fitted encoder can be reused
across rules and k without retraining.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ..similarity.base import ItemSimilarityBuilder
from .encoders import prep, _pairwise_sq

RULES = ("topk", "mutual", "epsilon", "dense", "cross_forced")


def _weights(D, metric, gamma, temperature):
    if metric == "cosine":                     # D is squared Euclidean of unit vectors: sim = 1 - D/2
        return np.exp((1.0 - 0.5 * D) / temperature)
    return np.exp(-gamma * D)


def build_graph(Z, owner, rule="topk", k=30, metric="euclidean", gamma=None, temperature=0.2,
                cross_min=0, edges_per_node=None):
    """Return (Aii_norm sparse tensor, stats dict)."""
    Zp = prep(Z, metric)
    N = Zp.shape[0]
    D = _pairwise_sq(Zp)
    np.fill_diagonal(D, np.inf)
    if gamma is None and metric != "cosine":
        d = np.sqrt(D[np.isfinite(D)])
        gamma = 1.0 / (2 * np.median(d) ** 2 + 1e-12)
    W = _weights(D, metric, gamma, temperature)
    W[~np.isfinite(D)] = 0.0
    k = min(k, N - 1)
    owner = np.asarray(owner)

    if rule in ("topk", "mutual", "cross_forced"):
        nbr = np.argpartition(D, k, axis=1)[:, :k]
        rows = np.repeat(np.arange(N), k); cols = nbr.reshape(-1)
        if rule == "cross_forced" and cross_min > 0:
            Dc = D.copy(); Dc[owner[:, None] == owner[None, :]] = np.inf
            kc = min(cross_min, N - 1)
            nbr_c = np.argpartition(Dc, kc, axis=1)[:, :kc]
            ok = np.isfinite(np.take_along_axis(Dc, nbr_c, axis=1)).reshape(-1)
            rows = np.concatenate([rows, np.repeat(np.arange(N), kc)[ok]])
            cols = np.concatenate([cols, nbr_c.reshape(-1)[ok]])
            pairs = np.unique(np.stack([rows, cols], axis=1), axis=0)      # coalesce() would sum duplicates
            rows, cols = pairs[:, 0], pairs[:, 1]
        mutual = rule == "mutual"
    elif rule == "epsilon":
        m = int(edges_per_node or k) * N              # match the edge count of top-k
        flat = D.reshape(-1)
        thr = np.partition(flat, m)[m]
        rows, cols = np.where(D <= thr)
        mutual = False
    elif rule == "dense":
        rows, cols = np.where(np.isfinite(D))
        mutual = False
    else:
        raise ValueError(f"unknown graph rule {rule!r}")

    vals = W[rows, cols]
    keep = vals > 1e-8
    rows, cols, vals = rows[keep], cols[keep], vals[keep]
    A = ItemSimilarityBuilder._make_symmetric_adj(rows.tolist(), cols.tolist(), vals.tolist(), N, mutual)
    return A, graph_stats(A, owner, gamma=gamma)


def graph_stats(A, owner, gamma=None):
    """Edge count, cross-user edge fraction, mean degree, connected components, isolated users."""
    A = A.coalesce()
    idx = A.indices().cpu().numpy()
    N = A.size(0)
    owner = np.asarray(owner)
    r, c = idx[0], idx[1]
    off = r != c
    r, c = r[off], c[off]
    n_edges = int(len(r) // 2)
    cross = float(np.mean(owner[r] != owner[c])) if len(r) else 0.0
    deg = np.bincount(r, minlength=N)
    S = coo_matrix((np.ones(len(r)), (r, c)), shape=(N, N))
    n_comp, labels = connected_components(S, directed=False)
    # a user is isolated if none of its items has a cross-user edge
    users = np.unique(owner)
    has_cross = {u: False for u in users}
    for a, b in zip(r, c):
        if owner[a] != owner[b]:
            has_cross[owner[a]] = True
    return dict(n_items=int(N), n_edges=n_edges, rho_cross=cross, mean_degree=float(deg.mean()),
                min_degree=int(deg.min()) if N else 0, n_components=int(n_comp),
                n_isolated_users=int(sum(not v for v in has_cross.values())), gamma=None if gamma is None else float(gamma))
