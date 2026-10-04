"""Additional item encoders (scattering, ISO feature bank, raw RBF) and intrinsic graph metrics.

The intrinsic metrics measure, without training GCF or a reward model, how much *cross-user*
signal an item-similarity graph carries (plan 01, Section 2):
  agree_cross   mean label agreement with the k nearest items owned by OTHER users
  rho_cross     fraction of kNN edges that connect items of different users
  vote_auroc    AUROC / MLPD of a k-NN label vote restricted to other users' items
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from ..similarity.base import ItemSimilarityBuilder, standardize_fit, standardize_apply, median_heuristic_gamma


# ----------------------------------------------------------------------------- encoders
class _EuclideanBuilder(ItemSimilarityBuilder):
    """Fixed (non-learned) feature map followed by an RBF kNN graph in Euclidean latent space."""
    name = "euclid"

    def _features(self, X):                       # (N, T, D) -> (N, F); fitted state in self
        raise NotImplementedError

    def _fit_features(self, X, cfg):
        raise NotImplementedError

    def fit(self, item_series, cfg):
        N, T, D = item_series.shape
        self._T, self._D = T, D
        F = self._fit_features(item_series, cfg)
        self.mu, self.sd = standardize_fit(F)
        Z = standardize_apply(F, self.mu, self.sd)
        self.pca = None
        pdim = int(getattr(cfg, f"{self.name}_pca_dim", 0) or 0)
        if pdim and pdim < Z.shape[1]:
            self.pca = PCA(n_components=pdim, random_state=cfg.seed).fit(Z)
            Z = self.pca.transform(Z)
        return self.build_graph_from_Z(Z, cfg)

    def build_graph_from_Z(self, Z, cfg):
        self.Z_train = Z.astype(np.float32)
        gamma_med = median_heuristic_gamma(Z, seed=cfg.seed)
        self.gamma = gamma_med * cfg.gamma_mul
        self.metric = "rbf"
        Aii_norm = self.build_knn_graph(Z, knn_k=cfg.knn_k, gamma=self.gamma, mutual=cfg.mutual)
        self.Aii_norm = Aii_norm
        meta = {"method": self.name, "dim": int(Z.shape[1]), "gamma": float(self.gamma),
                "knn_k": cfg.knn_k, "mutual": cfg.mutual}
        if cfg.verbose > 0:
            print(f"  [{self.name}] meta: {meta}")
        return {"Aii_norm": Aii_norm, "Z_train": self.Z_train, "gamma": self.gamma, "meta": meta}

    def build_graph(self, item_series, cfg):      # same signature as AESimilarity (load path)
        return self.build_graph_from_Z(self.transform_test(item_series), cfg)

    def transform_test(self, X_test):
        Z = standardize_apply(self._features(X_test), self.mu, self.sd)
        return (self.pca.transform(Z) if self.pca is not None else Z).astype(np.float32)

    def get_affinity(self, Z_query, Z_target, k):
        nn_ = NearestNeighbors(n_neighbors=min(k, Z_target.shape[0]), metric="euclidean").fit(Z_target)
        dist, nbr = nn_.kneighbors(Z_query, return_distance=True)
        return nbr, self._compute_rbf_affinity(dist ** 2, self.gamma)

    def save(self, path):
        import joblib
        joblib.dump(self, path)

    def load(self, path, device=None):
        import joblib
        self.__dict__.update(joblib.load(path).__dict__)


class GaborScattering:
    """First- and second-order Gabor scattering per channel with global average pooling.

    S1_j = mean_t |x * psi_j|,  S2_jk = mean_t ||x * psi_j| * psi_k| (k lower frequency than j).
    Pure numpy FFT; suited to short windows (T of a few hundred samples).
    """

    def __init__(self, T, n1=8, f_min=0.03, f_max=0.45, q=0.5, order2=True):
        self.T = T
        self.N = int(2 ** np.ceil(np.log2(T)))
        self.n1 = n1
        self.order2 = order2
        omega = np.fft.fftfreq(self.N)
        self.freqs = np.geomspace(f_max, f_min, n1)
        self.psi = np.stack([np.exp(-0.5 * ((omega - f) / (q * f)) ** 2) for f in self.freqs])

    def transform(self, X):
        B, T, D = X.shape
        x = np.zeros((B, D, self.N))
        x[:, :, :T] = X.transpose(0, 2, 1)
        Xf = np.fft.fft(x, axis=-1)
        U1 = np.abs(np.fft.ifft(Xf[:, :, None, :] * self.psi[None, None], axis=-1))   # (B, D, n1, N)
        feats = [np.abs(x).mean(axis=-1, keepdims=True), U1.mean(axis=-1)]
        if self.order2:
            U1f = np.fft.fft(U1, axis=-1)
            pairs = [(j, k) for j in range(self.n1) for k in range(j + 1, self.n1)]
            S2 = np.stack([np.abs(np.fft.ifft(U1f[:, :, j, :] * self.psi[k][None, None], axis=-1)).mean(axis=-1)
                           for j, k in pairs], axis=-1)
            feats.append(S2)
        return np.log1p(np.concatenate(feats, axis=-1)).reshape(B, -1)


class ScatterSimilarity(_EuclideanBuilder):
    name = "scatter"

    def _fit_features(self, X, cfg):
        self.sca = GaborScattering(X.shape[1], n1=cfg.scatter_n1)
        return self.sca.transform(X)

    def _features(self, X):
        return self.sca.transform(X)


class ISOBankSimilarity(_EuclideanBuilder):
    """The T-IV paper's pruned bank of ISO-inspired ride statistics as the item representation."""
    name = "isobank"

    def _fit_features(self, X, cfg):
        from reward.fully_bayesian.loeo import bank as B
        channels = list(cfg.graph_channel_names)
        fs = float(cfg.fs)
        bank = B.full_bank(channels)
        pruned, _ = B.prune_bank([X], channels, fs, bank, rho_max=cfg.isobank_rho_max)
        self.phi = B.make_pipeline(pruned, channels, fs).fit([X], None)
        self.feature_names = [f for f in self.phi.feature_names if f != "bias"]
        return self._features(X)

    def _features(self, X):
        Z = self.phi.transform(X)
        return Z[:, 1:] if self.phi.feature_names[0] == "bias" else Z


class RawRBFSimilarity(_EuclideanBuilder):
    """Flattened standardized window; the lower-bound encoder."""
    name = "raw_rbf"

    def _fit_features(self, X, cfg):
        return X.reshape(X.shape[0], -1)

    def _features(self, X):
        return X.reshape(X.shape[0], -1)


EXTRA_ENCODERS = {"scatter": ScatterSimilarity, "isobank": ISOBankSimilarity, "raw_rbf": RawRBFSimilarity}


def register_extra_encoders():
    from .. import similarity as S
    for k, v in EXTRA_ENCODERS.items():
        S.SIMILARITY_REGISTRY.setdefault(k, v)


def latent_metric(builder):
    """How to compare latents of this builder: 'cosine' or 'euclidean'."""
    return "cosine" if getattr(builder, "metric", "rbf") == "cosine" else "euclidean"


def prep(Z, metric):
    Z = np.asarray(Z, np.float64)
    if metric == "cosine":
        Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    return Z


# ----------------------------------------------------------------------------- intrinsic metrics
def _pairwise_sq(Z):
    sq = (Z * Z).sum(1)
    D = sq[:, None] + sq[None, :] - 2.0 * Z @ Z.T
    np.maximum(D, 0, out=D)
    return D


def cross_neighbors(Z, owner, k, metric="euclidean"):
    """Indices (N, k) of the k nearest items owned by a different user."""
    Z = prep(Z, metric)
    D = _pairwise_sq(Z)
    same = owner[:, None] == owner[None, :]
    D[same] = np.inf
    k = min(k, Z.shape[0] - 1)
    return np.argsort(D, axis=1)[:, :k], D


def all_neighbors(Z, k, metric="euclidean"):
    Z = prep(Z, metric)
    D = _pairwise_sq(Z)
    np.fill_diagonal(D, np.inf)
    k = min(k, Z.shape[0] - 1)
    return np.argsort(D, axis=1)[:, :k]


def intrinsic_metrics(Z, y, owner, ks, metric="euclidean", alpha=1.0, gamma=None, min_labels=10):
    """Per k: agree_cross, agree_cross_excess, rho_cross, vote_auroc, vote_mlpd.

    Agreement and the vote are evaluated *per target user* (neighbours drawn from the other users
    only, weighted like the test-time attachment) and macro-averaged over users with both classes
    and at least `min_labels` items.  Pooling items of users with different base rates would
    inflate or deflate AUROC for reasons unrelated to the similarity, so it is avoided.
    rho_cross is a property of the whole graph and stays global.
    """
    y = np.asarray(y, int); owner = np.asarray(owner)
    Zp = prep(Z, metric)
    D = _pairwise_sq(Zp)
    np.fill_diagonal(D, np.inf)
    if gamma is None:
        d = np.sqrt(D[np.isfinite(D)])
        gamma = 1.0 / (2 * np.median(d) ** 2 + 1e-12)
    same = owner[:, None] == owner[None, :]
    Dc = D.copy(); Dc[same] = np.inf
    kmax = min(max(ks), Zp.shape[0] - 1)
    nbr_cross = np.argsort(Dc, axis=1)[:, :kmax]
    nbr_all = np.argsort(D, axis=1)[:, :kmax]
    users = [u for u in np.unique(owner)
             if (owner == u).sum() >= min_labels and len(np.unique(y[owner == u])) == 2]
    out = {}
    for k in ks:
        k = min(int(k), kmax)
        nc = nbr_cross[:, :k]
        W = np.exp(-gamma * np.take_along_axis(Dc, nc, axis=1))           # same weighting as heldout_vote
        rho = float(np.mean(owner[nbr_all[:, :k]] != owner[:, None]))
        agrees, excs, aucs, mlpds = [], [], [], []
        for u in users:
            m = owner == u
            yu = y[m]; pi_u = float(yu.mean())
            ag = float(np.mean((y[nc[m]] == yu[:, None]).mean(axis=1)))
            p = ((W[m] * y[nc[m]]).sum(1) + alpha * pi_u) / (W[m].sum(1) + alpha)
            p = np.clip(p, 1e-6, 1 - 1e-6)
            agrees.append(ag); excs.append(ag - max(pi_u, 1 - pi_u))
            aucs.append(float(roc_auc_score(yu, p)))
            mlpds.append(float(np.mean(yu * np.log(p) + (1 - yu) * np.log(1 - p))))
        out[int(k)] = dict(agree_cross=float(np.mean(agrees)), agree_cross_excess=float(np.mean(excs)),
                           rho_cross=rho, vote_auroc=float(np.mean(aucs)), vote_mlpd=float(np.mean(mlpds)),
                           n_users=len(users))
    return out


def heldout_vote(Z_pop, y_pop, Z_q, k, metric="euclidean", alpha=1.0, weights=True, gamma=None):
    """k-NN label vote for query items from population items (used by the KNN-vote baseline)."""
    Zp, Zq = prep(Z_pop, metric), prep(Z_q, metric)
    nn_ = NearestNeighbors(n_neighbors=min(k, Zp.shape[0]), metric="euclidean").fit(Zp)
    dist, nbr = nn_.kneighbors(Zq, return_distance=True)
    y_pop = np.asarray(y_pop, float)
    pi = float(y_pop.mean())
    if weights:
        g = gamma if gamma is not None else 1.0 / (2 * np.median(dist) ** 2 + 1e-12)
        w = np.exp(-g * dist ** 2)
    else:
        w = np.ones_like(dist)
    p = ((w * y_pop[nbr]).sum(1) + alpha * pi) / (w.sum(1) + alpha)
    return np.clip(p, 1e-6, 1 - 1e-6)
