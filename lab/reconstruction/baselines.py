from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, filtfilt
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs" / "lab" / "reconstruction"

from loader import Dataset
from models.fully_bayesian.features import FNS

BOUNCE_STATS = ["impulse_abs", "abs_peak_deriv", "p2p_deriv", "wrms_z_deriv"]

# 배포 가용 신호 : 거동 = IMU + WHL만, 토크 입력 사용 가능. SAS/페달/브레이크 불가.
X_CH = ["IMU_RollRtVal", "IMU_VerAccelVal", "IMU_YawRtVal", "IMU_LatAccelVal", "IMU_LongAccelVal",
        "WHL_SpdFLVal", "WHL_SpdFRVal", "WHL_SpdRLVal", "WHL_SpdRRVal",
        "MCU_Mg1EstTqVal", "MCU_Mg2EstTqVal", "VCU_MotTqCmdFrntVal", "VCU_MotTqCmdRearVal"]
Y_CH = ["Bounce_rate_6D", "Roll_rate_6D", "Pitch_rate_6D"]
TEST = {"김재호", "김진명", "김태근", "신민철", "이강근"}
FS, T = 100.0, 1000
DEV = "cuda"


def filt(x, fc, btype):
    b, a = butter(2, fc / (FS / 2), btype)
    return filtfilt(b, a, x).astype(np.float32)


def physics(x):
    roll, az, ax = x[:, 0], x[:, 1], x[:, 4]
    vw = x[:, 5:9].mean(1) / 3.6
    bounce = filt(np.cumsum(az - az.mean()) / FS, 0.3, "high")
    pitch = -filt(np.gradient((ax - np.gradient(vw) * FS) / 9.81) * FS, 2.0, "low")
    return np.stack([bounce, roll, pitch], 1)


def batch_corr(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))


class Kalman:
    # half-car 정상상태 칼만. 상태 [z, z_dot, theta, theta_dot, eta(유색 heave 여기), eta_t(유색 pitch 여기),
    # az바이어스, 노면경사], 입력 [v_dot, 토크, 앞뒤 v_dot 차], 측정 [az, ax - v_dot].
    # p[:10] 물리 파라미터, p[10:] log 노이즈 분산.
    # kappa: 종방향 힘의 수직 성분(anti-lift), czt: 앞뒤 서스펜션 비대칭의 pitch-heave 커플링.
    # fit 옵션(기본 off): warmup=N이면 초기 N스텝 시변 게인, eta_t=True면 유색 pitch 여기 활성.
    # control=False면 입력 u를 완전 차단 (B=D=0, 측정만으로 추정. 측정식의 v_dot은 유지).
    def __init__(self, p, warmup=0, control=True):
        self.p = np.asarray(p)
        self.warmup = warmup
        self.control = control
        wz, zz, wt, zt, lever, g1, g2, kappa, czt, g3 = self.p[:10]
        qz, qt, qe, qet, qb, qg, rz, rx = np.exp(self.p[10:])
        dt = 1 / FS
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
        if not control:
            self.B[:] = 0.0
            self.D[:] = 0.0
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
        az = (xr[:, :, 1] - 1.0) * 9.81
        whl = xr[:, :, 5:9] / 3.6
        vdot = filt(np.gradient(whl.mean(2), axis=1) * FS, 5.0, "low")
        dvfr = filt(np.gradient(whl[:, :, :2].mean(2) - whl[:, :, 2:].mean(2), axis=1) * FS, 5.0, "low")
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
    def fit(cls, xr, yr, warmup=0, eta_t=False, control=True):
        sub = np.linspace(0, len(xr) - 1, 400).astype(int)
        xs, yb, yp = xr[sub], yr[sub, :, 0], yr[sub, :, 2]

        def loss(p):
            k = cls(p, warmup, control).predict(xs)
            cost = 3 - batch_corr(k[:, 0], yb).mean() - 2 * batch_corr(k[:, 2], yp).mean()
            return cost if np.isfinite(cost) else 10.0

        p0 = np.array([8.0, 0.3, 7.8, 0.3, 0.5, 0.3, 0.0, 0.0, 0.0, 0.0]
                      + list(np.log([1.0, 1.0, 1.0, 1.0 if eta_t else 1e-35, 1e-4, 1e-4, 0.1, 0.1])))
        res = minimize(loss, p0, method="Powell", options={"maxfev": 600})
        print(f"  fit loss {res.fun:.3f}  wz {res.x[0]:.1f}  wt {res.x[2]:.1f}  lever {res.x[4]:.2f}"
              f"  czt {res.x[8]:.2f}  g3 {res.x[9]:.2f}")
        return cls(res.x, warmup, control)


def load():
    xs, ys, ps, ids = [], [], [], []
    for ep in Dataset(ROOT / "datasets").episodes:
        df = ep.signals.df
        if len(df) < T:
            continue
        x = df[X_CH].to_numpy(np.float32)[:T]
        xs.append(x)
        ys.append(df[Y_CH].to_numpy(np.float32)[:T])
        ps.append(physics(x))
        ids.append(f"{ep.driver} {ep.id}")
    x, y, p = np.stack(xs), np.stack(ys), np.stack(ps)
    te = np.array([i.split()[0] in TEST for i in ids])
    std = lambda a: (a - a[~te].reshape(-1, a.shape[2]).mean(0)) / (a[~te].reshape(-1, a.shape[2]).std(0) + 1e-8)
    to = lambda a: torch.tensor(std(a).transpose(0, 2, 1))
    return to(x), to(y), to(p), torch.tensor(te), np.array(ids), x, y


class FIR(nn.Module):
    def __init__(self, ch, k=101):
        super().__init__()
        self.conv = nn.Conv1d(ch, 3, k, padding=k // 2)

    def forward(self, x):
        return self.conv(x)


def block(ci, co):
    return nn.Sequential(nn.Conv1d(ci, co, 9, padding=4), nn.ReLU(),
                         nn.Conv1d(co, co, 9, padding=4), nn.ReLU())


class UNet(nn.Module):
    def __init__(self, ch, w=32):
        super().__init__()
        self.e1, self.e2, self.e3 = block(ch, w), block(w, 2 * w), block(2 * w, 4 * w)
        self.d2, self.d1 = block(6 * w, 2 * w), block(3 * w, w)
        self.out = nn.Conv1d(w, 3, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.avg_pool1d(e1, 2))
        e3 = self.e3(F.avg_pool1d(e2, 2))
        d2 = self.d2(torch.cat([F.interpolate(e3, scale_factor=2.0), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, scale_factor=2.0), e1], 1))
        return self.out(d1)


class WaveUNet(nn.Module):
    # Wave-U-Net M1: conv-decimate encoder, linear-upsample-concat decoder, raw-input output skip.
    def __init__(self, ch, w=24, depth=12):
        super().__init__()
        cs = [w * (i + 1) for i in range(depth)]
        conv = lambda ci, co, k: nn.Sequential(nn.Conv1d(ci, co, k, padding=k // 2), nn.LeakyReLU(0.2))
        self.down = nn.ModuleList([conv(ch if i == 0 else cs[i - 1], c, 15) for i, c in enumerate(cs)])
        self.mid = conv(cs[-1], w * (depth + 1), 15)
        cur = w * (depth + 1)
        self.up = nn.ModuleList()
        for c in reversed(cs):
            self.up.append(conv(cur + c, c, 5))
            cur = c
        self.out = nn.Conv1d(w + ch, 3, 1)

    def forward(self, x):
        z, skip = x, []
        for down in self.down:
            z = down(z)
            skip.append(z)
            z = z[:, :, ::2]
        z = self.mid(z)
        for up, s in zip(self.up, reversed(skip)):
            z = F.interpolate(z, size=s.shape[-1], mode="linear", align_corners=False)
            z = up(torch.cat([s, z], 1))
        return self.out(torch.cat([x, z], 1))


def netout(model, x, p):
    return model(x) if p is None else p + model(torch.cat([x, p], 1))


def fit(model, x, y, p, epochs, bs=64, lr=1e-3):
    model.to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr)
    for ep in range(epochs):
        tot = 0.0
        for b in torch.randperm(len(x)).split(bs):
            xb, yb = x[b].to(DEV), y[b].to(DEV)
            pb = None if p is None else p[b].to(DEV)
            loss = F.mse_loss(netout(model, xb, pb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        if ep % 10 == 9:
            print(f"  epoch {ep + 1}  mse {tot / len(x):.4f}")
    return model


@torch.no_grad()
def predict(model, x, p, bs=64):
    model.eval()
    return torch.cat([netout(model, x[b].to(DEV), None if p is None else p[b].to(DEV))
                      for b in torch.arange(len(x)).split(bs)]).cpu().numpy()


def score(pred, y):
    r = np.array([[np.corrcoef(pred[i, j], y[i, j])[0, 1] for j in range(3)] for i in range(len(y))])
    return np.nanmedian(r, 0), np.nanpercentile(r, 25, 0), np.nanpercentile(r, 75, 0)


if __name__ == "__main__":
    x, y, p, te, ids, xr, yr = load()
    print(f"{len(x)} episodes  train {(~te).sum()}  test {te.sum()}")
    xtr, ytr, ptr = x[~te], y[~te], p[~te]
    xte, yte, pte = x[te], y[te], p[te]

    preds = {"physics": pte.numpy()}
    tn = te.numpy()
    kalman_params = {}
    for name, control in (("kalman", True), ("kalman_nc", False)):
        print(name)
        km = Kalman.fit(xr[~tn], yr[~tn], control=control)
        kalman_params[name] = km.p
        kal = km.predict(xr)
        mk, sk = kal[~tn].mean((0, 2), keepdims=True), kal[~tn].std((0, 2), keepdims=True)
        preds[name] = ((kal - mk) / sk)[tn]
    ch = len(X_CH) + 3
    models = {"fir": (FIR(ch), 20, True), "unet": (UNet(ch), 40, True),
              "waveunet": (WaveUNet(len(X_CH)), 40, False)}
    for name, (model, epochs, residual) in models.items():
        print(name)
        preds[name] = predict(fit(model, xtr, ytr, ptr if residual else None, epochs),
                              xte, pte if residual else None)

    print(f"\n{'':8s}" + "".join(f"{c:>28s}" for c in Y_CH))
    for name, pr in preds.items():
        med, lo, hi = score(pr, yte.numpy())
        print(f"{name:8s}" + "".join(f"     {m:.3f} [{l:.3f}, {h:.3f}]" for m, l, h in zip(med, lo, hi)))

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "reconstruction_predictions.npz", y=yte.numpy(), ids=ids[te.numpy()],
             kalman_params=kalman_params["kalman"], kalman_nc_params=kalman_params["kalman_nc"], **preds)
    torch.save({name: m.state_dict() for name, (m, _, _) in models.items()}, OUT / "models.pt")
    true_bounce = yte.numpy()[:, 0]
    print(f"\nbounce stats corr (test episodes)\n{'':8s}" + "".join(f"{s:>16s}" for s in BOUNCE_STATS))
    for name, pr in preds.items():
        rs = [np.corrcoef(FNS[s](true_bounce, FS), FNS[s](pr[:, 0], FS))[0, 1] for s in BOUNCE_STATS]
        print(f"{name:8s}" + "".join(f"{r:16.3f}" for r in rs))
