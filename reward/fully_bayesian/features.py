import numpy as np
from sklearn.decomposition import PCA


def _deriv(x, fs):                           # time derivative along axis 1, length preserved (rate -> accel)
    return np.gradient(x, 1.0 / fs, axis=1)


def _wk(f):  # ISO 2631-1 W_k (vertical z), |H(jw)| shape; overall gain irrelevant (z-scored)
    s = 1j * 2 * np.pi * f
    w1, w2 = 2 * np.pi * 0.4, 2 * np.pi * 100.0           # band-limit hi-pass / lo-pass
    w3 = 2 * np.pi * 12.5                                 # a-v transition
    w4, Q4 = 2 * np.pi * 12.5, 0.63
    w5, Q5 = 2 * np.pi * 2.37, 0.91                       # upward step
    w6, Q6 = 2 * np.pi * 3.35, 0.91
    Hh = s ** 2 / (s ** 2 + (w1 / 0.7071) * s + w1 ** 2)
    Hl = w2 ** 2 / (s ** 2 + (w2 / 0.7071) * s + w2 ** 2)
    Ht = (1 + s / w3) / (1 + s / (Q4 * w4) + (s / w4) ** 2)
    Hs = ((w6 / w5) ** 2) * (1 + s / (Q5 * w5) + (s / w5) ** 2) / (1 + s / (Q6 * w6) + (s / w6) ** 2)
    return np.abs(Hh * Hl * Ht * Hs)


def _wd(f):                                  # ISO 2631-1 W_d (horizontal x/y): a-v at 2.0Hz, no upward step
    s = 1j * 2 * np.pi * f
    w1, w2 = 2 * np.pi * 0.4, 2 * np.pi * 100.0
    w4, Q4 = 2 * np.pi * 2.0, 0.63
    Hh = s ** 2 / (s ** 2 + (w1 / 0.7071) * s + w1 ** 2)
    Hl = w2 ** 2 / (s ** 2 + (w2 / 0.7071) * s + w2 ** 2)
    Ht = (1 + s / w4) / (1 + s / (Q4 * w4) + (s / w4) ** 2)
    return np.abs(Hh * Hl * Ht)


def _we(f):                                  # ISO 2631-1 W_e (rotational): a-v at 1.0Hz, no upward step
    s = 1j * 2 * np.pi * f
    w1, w2 = 2 * np.pi * 0.4, 2 * np.pi * 100.0
    w4, Q4 = 2 * np.pi * 1.0, 0.63
    Hh = s ** 2 / (s ** 2 + (w1 / 0.7071) * s + w1 ** 2)
    Hl = w2 ** 2 / (s ** 2 + (w2 / 0.7071) * s + w2 ** 2)
    Ht = (1 + s / w4) / (1 + s / (Q4 * w4) + (s / w4) ** 2)
    return np.abs(Hh * Hl * Ht)


def _wrms(x, fs, weight):                    # frequency-weighted RMS, weight = _wk / _wd / _we
    X = np.fft.rfft(x, axis=1)
    f = np.fft.rfftfreq(x.shape[1], 1.0 / fs)
    aw = np.fft.irfft(X * weight(f), n=x.shape[1], axis=1)
    return np.sqrt((aw ** 2).mean(1))


def _band_rms(x, fs, lo, hi):                # sqrt of PSD integral in [lo, hi)
    P = np.abs(np.fft.rfft(x, axis=1)) ** 2
    f = np.fft.rfftfreq(x.shape[1], 1.0 / fs)
    mask = (f >= lo) & (f < hi)
    return np.sqrt(P[:, mask].sum(1))


def _band_rel(x, fs, lo, hi):                # relative PSD energy in [lo, hi)
    P = np.abs(np.fft.rfft(x, axis=1)) ** 2
    f = np.fft.rfftfreq(x.shape[1], 1.0 / fs)
    mask = (f >= lo) & (f < hi)
    return P[:, mask].sum(1) / (P.sum(1) + 1e-12)


def _mtvv(x, fs):                            # max running 1s RMS (ISO 2631 MTVV)
    w = int(round(fs))
    if w >= x.shape[1]:
        return np.sqrt((x ** 2).mean(1))
    c = np.cumsum(x ** 2, axis=1)
    return np.sqrt(((c[:, w:] - c[:, :-w]) / w).max(1))


def _third_pos_peak(x):                      # value of 3rd positive local max per row, 0 if fewer
    m = (x[:, 1:-1] > x[:, :-2]) & (x[:, 1:-1] > x[:, 2:])
    out = np.zeros(len(x))
    for i in range(len(x)):
        pk = x[i, 1:-1][m[i]]
        pk = pk[pk > 0]
        out[i] = pk[2] if len(pk) >= 3 else 0.0
    return out


# 통계 메뉴. 전부 (x [N, T], fs) -> [N].  manual_stats = {채널: [이름...]}으로 선택.
FNS = {
    # 시간영역 기본
    "mean":           lambda x, fs: x.mean(1),
    "std":            lambda x, fs: x.std(1),
    "abs_peak":       lambda x, fs: np.abs(x).max(1),
    "energy":         lambda x, fs: (x ** 2).mean(1),
    "rms":            lambda x, fs: np.sqrt((x ** 2).mean(1)),
    "p95_abs":        lambda x, fs: np.percentile(np.abs(x), 95, axis=1),
    "p2p":            lambda x, fs: x.max(1) - x.min(1),
    "impulse_abs":    lambda x, fs: np.abs(x).sum(1) / fs,
    "crest":          lambda x, fs: np.abs(x).max(1) / (np.sqrt((x ** 2).mean(1)) + 1e-12),
    "z_nod":          lambda x, fs: _third_pos_peak(x),                     # 3rd positive peak
    # 미분(rate -> accel) 계열
    "mean_deriv":     lambda x, fs: _deriv(x, fs).mean(1),
    "rms_deriv":      lambda x, fs: np.sqrt((_deriv(x, fs) ** 2).mean(1)),
    "p2p_deriv":      lambda x, fs: _deriv(x, fs).max(1) - _deriv(x, fs).min(1),
    "abs_peak_deriv": lambda x, fs: np.abs(_deriv(x, fs)).max(1),
    # ISO 2631 계열
    "vdv":            lambda x, fs: ((x ** 4).sum(1) / fs) ** 0.25,         # 4th-power, peak-sensitive
    "vdv_deriv":      lambda x, fs: ((_deriv(x, fs) ** 4).sum(1) / fs) ** 0.25,
    "mtvv":           lambda x, fs: _mtvv(x, fs),                           # max transient vibration
    "wrms_z":         lambda x, fs: _wrms(x, fs, _wk),                      # W_k vertical accel
    "wrms_xy":        lambda x, fs: _wrms(x, fs, _wd),                      # W_d horizontal accel
    "wrms_rot":       lambda x, fs: _wrms(_deriv(x, fs), fs, _we),          # W_e rotation (rate -> ang accel)
    "wrms_z_deriv":   lambda x, fs: _wrms(_deriv(x, fs), fs, _wk),          # W_k vertical (heave rate -> accel)
    # 주파수 대역
    "sigma_sd":       lambda x, fs: _band_rms(x, fs, 0.5, 5.0),             # primary ride band
    "band_low":       lambda x, fs: _band_rel(x, fs, 1.0, 2.0),
    "band_mid":       lambda x, fs: _band_rel(x, fs, 4.0, 8.0),
    "rms_dfz":        lambda x, fs: np.sqrt((((x - x.mean(1, keepdims=True)) / (x.mean(1, keepdims=True) + 1e-12)) ** 2).mean(1)),
}


def _group_contributions(theta, Phi, groups, cov=None, top_k=12):
    theta = np.asarray(theta)
    phi = np.asarray(Phi)
    contrib = np.mean(phi * theta[None, :], axis=0) if phi.ndim == 2 else phi * theta
    by_group = {}
    for idx, group in enumerate(groups):
        if group == "bias":
            continue
        rec = by_group.setdefault(group, {"signed": 0.0, "abs": 0.0, "var": 0.0})
        rec["signed"] += float(contrib[idx])
        rec["abs"] += float(abs(contrib[idx]))
        if cov is not None:
            rec["var"] += float(cov[idx, idx])
    rows = [(g, r["signed"], r["abs"], float(np.sqrt(max(r["var"], 0.0))) if cov is not None else None)
            for g, r in by_group.items()]
    rows.sort(key=lambda r: r[2], reverse=True)
    return rows[:top_k]


class Features:
    """z = φ(τ): cfg.manual_stats의 (채널, 통계) 쌍을 FNS로 추출 + 표준화 (+bias).

    pca_dim을 주면 표준화된 feature를 PCA로 압축 (feature_method="manual_pca").
    """

    def __init__(self, cfg, pca_dim=None):
        self.channels = cfg.view.cols
        self.fs = cfg.view.fs
        self.standardize = cfg.standardize
        self.include_bias = cfg.include_bias
        # 순서 유지하며 중복 (채널, 통계) 쌍 제거 — 중복 열은 feature 단위 selection을
        # ill-defined하게 만들고 projected Hessian을 singular로 만든다.
        seen = set()
        self.pairs = []
        for ch, ms in cfg.manual_stats.items():
            for m in ms:
                if (ch, m) not in seen:
                    if m not in FNS:
                        raise KeyError(f"Unknown manual statistic: {m}")
                    self.pairs.append((ch, m))
                    seen.add((ch, m))
        self.pca_dim = pca_dim
        self.pca = None

    def fit(self, X_by_user, y_by_user):
        F = np.concatenate([self._extract(X) for X in X_by_user], axis=0)
        self.mu = F.mean(0)
        self.sd = F.std(0) + 1e-12
        names = [f"{ch}__{m}" for ch, m in self.pairs]
        groups = [ch for ch, _ in self.pairs]
        if self.pca_dim:
            Z = (F - self.mu) / self.sd if self.standardize else F
            n_comp = min(self.pca_dim, Z.shape[1], Z.shape[0] - 1)
            self.pca = PCA(n_components=n_comp, random_state=0).fit(Z)
            names = [f"pc_{i}" for i in range(n_comp)]
            groups = list(names)
        bias = ["bias"] if self.include_bias else []
        self.feature_names = bias + names
        self.groups = bias + groups
        return self

    def transform(self, X):
        Z = self._extract(X)
        if self.standardize:
            Z = (Z - self.mu) / self.sd
        if self.pca is not None:
            Z = self.pca.transform(Z)
        if self.include_bias:
            Z = np.concatenate([np.ones((Z.shape[0], 1)), Z], axis=1)
        return Z

    def _extract(self, X):                       # (N, T, D) -> (N, n_features)
        idx = {ch: j for j, ch in enumerate(self.channels)}
        cols = [FNS[m](X[:, :, idx[ch]], self.fs) for ch, m in self.pairs]
        return np.stack(cols, axis=1)

    def group_contributions(self, theta, Phi, cov=None, top_k=12):
        return _group_contributions(theta, Phi, self.groups, cov=cov, top_k=top_k)


class AEPipeline:
    """Conv1d autoencoder latent code."""

    def __init__(self, cfg):
        self.cfg = cfg

    def fit(self, X_by_user, y_by_user):
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
        from ..copl.similarity.ae import Conv1dAE

        cfg = self.cfg
        X = np.concatenate(X_by_user, axis=0)
        N, T, D = X.shape
        flat = X.reshape(N * T, D)
        self.mu = flat.mean(0)
        self.sd = flat.std(0) + 1e-6
        Xs = (X - self.mu) / self.sd

        device = torch.device(cfg.ae_device)
        latent_dim = cfg.ae_latent_dim
        self.ae = Conv1dAE(D, T, latent_dim, hidden_channels=cfg.ae_hidden_channels).to(device)
        opt = torch.optim.Adam(self.ae.parameters(), lr=cfg.ae_lr)
        loader = DataLoader(TensorDataset(torch.tensor(Xs.transpose(0, 2, 1), dtype=torch.float32)),
                            batch_size=cfg.ae_batch_size, shuffle=True)
        self.history = {"train/loss": []}
        for _ in range(cfg.ae_epochs):
            self.ae.train()
            ep_loss = 0.0
            for (xb,) in loader:
                xb = xb.to(device)
                x_hat, _ = self.ae(xb)
                loss = F.mse_loss(x_hat, xb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                ep_loss += loss.item() * xb.size(0)
            self.history["train/loss"].append(ep_loss / N)
        self.ae.eval()
        self.feature_names = ["bias"] + [f"z_{i}" for i in range(latent_dim)]
        self.groups = list(self.feature_names)
        return self

    def transform(self, X):
        import torch
        Xs = (X - self.mu) / self.sd
        device = next(self.ae.parameters()).device
        xb = torch.tensor(Xs.transpose(0, 2, 1), dtype=torch.float32).to(device)
        with torch.no_grad():
            z = self.ae.encode(xb).cpu().numpy()
        return np.concatenate([np.ones((z.shape[0], 1)), z], axis=1)

    def group_contributions(self, theta, Phi, cov=None, top_k=12):
        return _group_contributions(theta, Phi, self.groups, cov=cov, top_k=top_k)


def build_feature_pipeline(cfg):
    return {
        "manual": lambda: Features(cfg),
        "manual_pca": lambda: Features(cfg, pca_dim=cfg.manual_pca_dim),
        "ae": lambda: AEPipeline(cfg),
    }[cfg.feature_method]()
