import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, filtfilt
from scipy.optimize import minimize


def filt(x, fc, btype, fs):
    b, a = butter(2, fc / (fs / 2), btype)
    return filtfilt(b, a, x).astype(np.float32)


def physics(x, fs):
    roll, az, ax = x[:, 0], x[:, 1], x[:, 4]
    vw = x[:, 5:9].mean(1) / 3.6
    bounce = filt(np.cumsum(az - az.mean()) / fs, 0.3, "high", fs)
    pitch = -filt(np.gradient((ax - np.gradient(vw) * fs) / 9.81) * fs, 2.0, "low", fs)
    return np.stack([bounce, roll, pitch], 1)


def batch_corr(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))


class Kalman:
    # half-car 정상상태 칼만. 상태 [z, z_dot, theta, theta_dot, eta(유색 노면 여기), eta_t(유색 pitch 여기),
    # az바이어스, 노면경사], 입력 [v_dot, 토크, 앞뒤 v_dot 차], 측정 [az, ax - v_dot].
    # p[:10] 물리 파라미터, p[10:] log 노이즈 분산.
    # kappa: 종방향 힘의 수직 성분(anti-lift), czt: 앞뒤 서스펜션 비대칭의 pitch-heave 커플링.
    # fit 옵션(기본 off): warmup=N이면 초기 N스텝 시변 게인, eta_t=True면 유색 pitch 여기 활성.
    def __init__(self, p, fs, warmup=0):
        self.p = np.asarray(p)
        self.fs = fs
        self.warmup = warmup
        wz, zz, wt, zt, lever, g1, g2, kappa, czt, g3 = self.p[:10]
        qz, qt, qe, qet, qb, qg, rz, rx = np.exp(self.p[10:])
        dt = 1 / fs
        self.A = np.eye(8)
        self.A[0, 1] = dt
        self.A[1, 0] = -wz * wz * dt
        self.A[1, 1] = 1 - 2 * zz * wz * dt
        self.A[1, 4] = dt
        self.A[2, 3] = dt
        self.A[3, 1] = czt * dt
        self.A[3, 2] = -wt * wt * dt
        self.A[3, 3] = 1 - 2 * zt * wt * dt
        self.A[3, 5] = dt
        self.B = np.zeros((8, 3))
        self.B[1, 0] = kappa * dt
        self.B[3] = g1 * dt, g2 * dt, g3 * dt
        self.H = np.array([[-wz * wz, -2 * zz * wz + lever * czt, -lever * wt * wt, -2 * lever * zt * wt,
                            1, lever, 1, 0],
                           [0, 0, 9.81, 0, 0, 0, 0, 9.81]])
        self.D = np.array([[kappa + lever * g1, lever * g2, lever * g3], [0.0, 0.0, 0.0]])
        Q = np.diag([0, qz, 0, qt, qe, qet, qb, qg]) * dt
        R = np.diag([rz, rx])
        P = np.eye(8) * (100.0 if warmup else 1.0)
        self.Ks = np.empty((300, 8, 2))
        for i in range(300):
            P = self.A @ P @ self.A.T + Q
            self.Ks[i] = P @ self.H.T @ np.linalg.inv(self.H @ P @ self.H.T + R)
            P = (np.eye(8) - self.Ks[i] @ self.H) @ P
        self.K = self.Ks[-1]

    def predict(self, xr):
        fs = self.fs
        az = (xr[:, :, 1] - 1.0) * 9.81
        whl = xr[:, :, 5:9] / 3.6
        vdot = filt(np.gradient(whl.mean(2), axis=1) * fs, 5.0, "low", fs)
        dvfr = filt(np.gradient(whl[:, :, :2].mean(2) - whl[:, :, 2:].mean(2), axis=1) * fs, 5.0, "low", fs)
        u = np.stack([vdot, xr[:, :, 9] + xr[:, :, 10], dvfr], 2)
        ym = np.stack([az, xr[:, :, 4] * 9.81 - vdot], 2)
        s = np.zeros((len(xr), 8))
        out = np.empty((len(xr), 2, xr.shape[1]))
        for t in range(xr.shape[1]):
            s = s @ self.A.T + u[:, t] @ self.B.T
            K = self.Ks[t] if t < self.warmup else self.K
            s += (ym[:, t] - s @ self.H.T - u[:, t] @ self.D.T) @ K.T
            out[:, :, t] = s[:, [1, 3]]
        return np.stack([out[:, 0], xr[:, :, 0], -out[:, 1]], 1)

    @classmethod
    def fit(cls, xr, yr, fs, warmup=0, eta_t=False):
        sub = np.linspace(0, len(xr) - 1, 400).astype(int)
        xs, yb, yp = xr[sub], yr[sub, :, 0], yr[sub, :, 2]

        def loss(p):
            k = cls(p, fs, warmup).predict(xs)
            cost = 3 - batch_corr(k[:, 0], yb).mean() - 2 * batch_corr(k[:, 2], yp).mean()
            return cost if np.isfinite(cost) else 10.0

        p0 = np.array([8.0, 0.3, 7.8, 0.3, 0.5, 0.3, 0.0, 0.0, 0.0, 0.0]
                      + list(np.log([1.0, 1.0, 1.0, 1.0 if eta_t else 1e-35, 1e-4, 1e-4, 0.1, 0.1])))
        res = minimize(loss, p0, method="Powell", options={"maxfev": 600})
        print(f"  fit loss {res.fun:.3f}  wz {res.x[0]:.1f}  wt {res.x[2]:.1f}  lever {res.x[4]:.2f}"
              f"  czt {res.x[8]:.2f}  g3 {res.x[9]:.2f}")
        return cls(res.x, fs, warmup)


class FIR(nn.Module):
    def __init__(self, ch, k=101):
        super().__init__()
        self.conv = nn.Conv1d(ch, 3, k, padding=k // 2)

    def forward(self, x):
        return self.conv(x)


class LSTM(nn.Module):
    def __init__(self, ch, out_ch, hidden=96, layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(ch, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.out = nn.Linear(hidden, out_ch)

    def forward(self, x):
        z, _ = self.lstm(x.transpose(1, 2))
        return self.out(z).transpose(1, 2)


class UNet(nn.Module):
    def __init__(self, ch, w=32):
        super().__init__()
        self.e1 = nn.Sequential(nn.Conv1d(ch, w, 9, padding=4), nn.ReLU(),
                                nn.Conv1d(w, w, 9, padding=4), nn.ReLU())
        self.e2 = nn.Sequential(nn.Conv1d(w, 2 * w, 9, padding=4), nn.ReLU(),
                                nn.Conv1d(2 * w, 2 * w, 9, padding=4), nn.ReLU())
        self.e3 = nn.Sequential(nn.Conv1d(2 * w, 4 * w, 9, padding=4), nn.ReLU(),
                                nn.Conv1d(4 * w, 4 * w, 9, padding=4), nn.ReLU())
        self.d2 = nn.Sequential(nn.Conv1d(6 * w, 2 * w, 9, padding=4), nn.ReLU(),
                                nn.Conv1d(2 * w, 2 * w, 9, padding=4), nn.ReLU())
        self.d1 = nn.Sequential(nn.Conv1d(3 * w, w, 9, padding=4), nn.ReLU(),
                                nn.Conv1d(w, w, 9, padding=4), nn.ReLU())
        self.out = nn.Conv1d(w, 3, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.avg_pool1d(e1, 2))
        e3 = self.e3(F.avg_pool1d(e2, 2))
        d2 = self.d2(torch.cat([F.interpolate(e3, scale_factor=2.0), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, scale_factor=2.0), e1], 1))
        return self.out(d1)


def netout(model, x, p=None):
    """Run either a direct model or a physics-residual model."""
    if p is None:
        return model(x)
    return p + model(torch.cat([x, p], 1))


def fit(model, x, y, p, epochs, device, bs=64, lr=1e-3,
        weight_decay=0.0, grad_clip=None):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for ep in range(epochs):
        tot = 0.0
        for b in torch.randperm(len(x)).split(bs):
            xb, yb = x[b].to(device), y[b].to(device)
            pb = None if p is None else p[b].to(device)
            loss = F.mse_loss(netout(model, xb, pb), yb)
            opt.zero_grad()
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            tot += loss.item() * len(b)
        if ep % 10 == 9:
            print(f"  epoch {ep + 1}  mse {tot / len(x):.4f}")
    return model


@torch.no_grad()
def predict(model, x, p, device, bs=64):
    model.to(device).eval()
    return torch.cat([
        netout(model, x[b].to(device), None if p is None else p[b].to(device))
        for b in torch.arange(len(x)).split(bs)
    ]).cpu().numpy()
