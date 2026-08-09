from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, filtfilt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs" / "lab" / "reconstruction"

from loader import Dataset
from lab.reconstruction.kalman_filters import KALMAN_FILTERS
from reward.fully_bayesian.features import FNS

BOUNCE_STATS = ["impulse_abs", "abs_peak_deriv", "p2p_deriv", "wrms_z_deriv"]

# 배포 가용 신호 : 거동 = IMU + WHL만, 토크 입력 사용 가능. SAS/페달/브레이크 불가.
X_CH = ["IMU_RollRtVal", "IMU_VerAccelVal", "IMU_YawRtVal", "IMU_LatAccelVal", "IMU_LongAccelVal",
        "WHL_SpdFLVal", "WHL_SpdFRVal", "WHL_SpdRLVal", "WHL_SpdRRVal",
        "MCU_Mg1EstTqVal", "MCU_Mg2EstTqVal", "VCU_MotTqCmdFrntVal", "VCU_MotTqCmdRearVal"]
Y_CH = ["Bounce_rate_6D", "Roll_rate_6D", "Pitch_rate_6D"]
TEST = {"김재호", "김진명", "김태근", "신민철", "이강근"}
FS, T = 100.0, 1000
DEV = "cuda"

KALMAN_SPECS = {
    **{name: (kf_class, True) for name, kf_class in KALMAN_FILTERS.items()},
    "kalman": (KALMAN_FILTERS["delayed_road_kf"], True),
    "kalman_nc": (KALMAN_FILTERS["delayed_road_kf"], False),
    "delayed_road_kf_nc": (KALMAN_FILTERS["delayed_road_kf"], False),
}

# 실행할 baseline만 선택. 전체 실행 예시:
# ("physics", "reduced_kf", "delayed_road_kf", "quarter_car_kf",
#  "parametric_halfcar_kf", "parametric_halfcar_rts",
#  "linear_halfcar_kf", "linear_halfcar_rts", "nonlinear_halfcar_ekf",
#  "fir", "lstm", "unet", "unet1d", "waveunet")
RUN_METHODS = ("physics", "unet", "lstm")


def filt(x, fc, btype):
    b, a = butter(2, fc / (FS / 2), btype)
    return filtfilt(b, a, x).astype(np.float32)


def physics(x):
    roll, az, ax = x[:, 0], x[:, 1], x[:, 4]
    vw = x[:, 5:9].mean(1) / 3.6
    bounce = filt(np.cumsum(az - az.mean()) / FS, 0.3, "high")
    pitch = -filt(np.gradient((ax - np.gradient(vw) * FS) / 9.81) * FS, 2.0, "low")
    return np.stack([bounce, roll, pitch], 1)


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


class LSTM(nn.Module):
    """Causal sequence-to-sequence LSTM baseline without a physics residual."""

    def __init__(self, ch, out_ch, hidden=96, layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(ch, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.out = nn.Linear(hidden, out_ch)

    def forward(self, x):
        z, _ = self.lstm(x.transpose(1, 2))
        return self.out(z).transpose(1, 2)


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


class UNet1D(nn.Module):
    """Axis-only 1D adaptation of Ronneberger et al.'s original U-Net.

    Reference: Ronneberger, Fischer, and Brox, "U-Net: Convolutional
    Networks for Biomedical Image Segmentation," MICCAI 2015.

    The original topology is preserved: four down/up levels, feature widths
    64-128-256-512-1024, two unpadded kernel-3 convolutions per level,
    max-pooling by 2, kernel-2 transposed convolutions, crop-and-concatenate
    skips, and a final point-wise convolution.  Only the spatial operators and
    task-specific input/output channel counts differ from the 2D paper model.

    As in the paper's overlap-tile strategy, reflection context is added outside
    the valid-convolution core and the central output is cropped back to the
    requested sequence length.
    """

    def __init__(self, ch, out_ch=3):
        super().__init__()

        def conv_block(ci, co):
            return nn.Sequential(nn.Conv1d(ci, co, 3), nn.ReLU(inplace=True),
                                 nn.Conv1d(co, co, 3), nn.ReLU(inplace=True))

        self.pool = nn.MaxPool1d(2, 2)
        self.e1 = conv_block(ch, 64)
        self.e2 = conv_block(64, 128)
        self.e3 = conv_block(128, 256)
        self.e4 = conv_block(256, 512)
        self.mid = conv_block(512, 1024)

        self.u4 = nn.ConvTranspose1d(1024, 512, 2, stride=2)
        self.d4 = conv_block(1024, 512)
        self.u3 = nn.ConvTranspose1d(512, 256, 2, stride=2)
        self.d3 = conv_block(512, 256)
        self.u2 = nn.ConvTranspose1d(256, 128, 2, stride=2)
        self.d2 = conv_block(256, 128)
        self.u1 = nn.ConvTranspose1d(128, 64, 2, stride=2)
        self.d1 = conv_block(128, 64)
        self.out = nn.Conv1d(64, out_ch, 1)
        self.apply(self._init_original)

    @staticmethod
    def _init_original(module):
        if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
            n = module.in_channels * module.kernel_size[0]
            nn.init.normal_(module.weight, mean=0.0, std=np.sqrt(2.0 / n))
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def _center_crop(x, length):
        extra = x.shape[-1] - length
        if extra < 0:
            raise ValueError(f"cannot crop length {x.shape[-1]} to {length}")
        start = extra // 2
        return x[..., start:start + length]

    @staticmethod
    def _core_output_length(length):
        n = int(length)
        for _ in range(4):
            n -= 4
            if n < 2:
                return -1
            n //= 2
        n -= 4
        if n < 1:
            return -1
        for _ in range(4):
            n = 2 * n - 4
        return n

    @classmethod
    def _padded_input_length(cls, output_length):
        n = int(output_length)
        while cls._core_output_length(n) < output_length:
            n += 1
        return n

    def _forward_core(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        z = self.mid(self.pool(e4))

        z = self.u4(z)
        z = self.d4(torch.cat([self._center_crop(e4, z.shape[-1]), z], 1))
        z = self.u3(z)
        z = self.d3(torch.cat([self._center_crop(e3, z.shape[-1]), z], 1))
        z = self.u2(z)
        z = self.d2(torch.cat([self._center_crop(e2, z.shape[-1]), z], 1))
        z = self.u1(z)
        z = self.d1(torch.cat([self._center_crop(e1, z.shape[-1]), z], 1))
        return self.out(z)

    def forward(self, x):
        length = x.shape[-1]
        padded = self._padded_input_length(length)
        extra = padded - length
        left, right = extra // 2, extra - extra // 2
        if max(left, right) >= length:
            raise ValueError(f"sequence length {length} is too short for reflection padding")
        if extra:
            x = F.pad(x, (left, right), mode="reflect")
        return self._center_crop(self._forward_core(x), length)


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


def fit(model, x, y, p, epochs, bs=64, lr=1e-3,
        weight_decay=0.0, grad_clip=None):
    model.to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for ep in range(epochs):
        tot = 0.0
        for b in torch.randperm(len(x)).split(bs):
            xb, yb = x[b].to(DEV), y[b].to(DEV)
            pb = None if p is None else p[b].to(DEV)
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
def predict(model, x, p, bs=64):
    model.eval()
    return torch.cat([netout(model, x[b].to(DEV), None if p is None else p[b].to(DEV))
                      for b in torch.arange(len(x)).split(bs)]).cpu().numpy()


def score(pred, y):
    r = np.array([[np.corrcoef(pred[i, j], y[i, j])[0, 1] for j in range(3)] for i in range(len(y))])
    return np.nanmedian(r, 0), np.nanpercentile(r, 25, 0), np.nanpercentile(r, 75, 0)


if __name__ == "__main__":
    known = {"physics", "fir", "lstm", "unet", "unet1d", "waveunet"} | set(KALMAN_SPECS)
    unknown = set(RUN_METHODS) - known
    if unknown:
        raise ValueError(f"unknown RUN_METHODS: {sorted(unknown)}")

    x, y, p, te, ids, xr, yr = load()
    print(f"{len(x)} episodes  train {(~te).sum()}  test {te.sum()}")
    xtr, ytr, ptr = x[~te], y[~te], p[~te]
    xte, yte, pte = x[te], y[te], p[te]

    preds = {}
    if "physics" in RUN_METHODS:
        preds["physics"] = pte.numpy()
    tn = te.numpy()
    kalman_params = {}
    for name, (kf_class, control) in KALMAN_SPECS.items():
        if name not in RUN_METHODS:
            continue
        print(name)
        km = kf_class.fit(xr[~tn], yr[~tn], fs=FS, control=control)
        kalman_params[name] = km.p
        kal = km.predict(xr)
        mk, sk = kal[~tn].mean((0, 2), keepdims=True), kal[~tn].std((0, 2), keepdims=True)
        preds[name] = ((kal - mk) / sk)[tn]
    ch = len(X_CH) + len(Y_CH)
    model_specs = {
        "fir": (lambda: FIR(ch), 20, True, {}),
        "lstm": (lambda: LSTM(len(X_CH), len(Y_CH)), 45, False,
                 {"bs": 32, "lr": 5e-3, "weight_decay": 0.0, "grad_clip": 1.0}),
        "unet": (lambda: UNet(ch), 40, True, {}),
        "unet1d": (lambda: UNet1D(len(X_CH)), 40, False, {}),
        "waveunet": (lambda: WaveUNet(len(X_CH)), 40, False, {}),
    }
    models = {}
    for name, (make_model, epochs, residual, fit_kwargs) in model_specs.items():
        if name not in RUN_METHODS:
            continue
        model = make_model()
        models[name] = (model, epochs, residual)
        print(name)
        preds[name] = predict(fit(model, xtr, ytr, ptr if residual else None,
                                  epochs, **fit_kwargs),
                              xte, pte if residual else None)

    print(f"\n{'':8s}" + "".join(f"{c:>28s}" for c in Y_CH))
    for name, pr in preds.items():
        med, lo, hi = score(pr, yte.numpy())
        print(f"{name:8s}" + "".join(f"     {m:.3f} [{l:.3f}, {h:.3f}]" for m, l, h in zip(med, lo, hi)))

    OUT.mkdir(parents=True, exist_ok=True)
    save_data = {"y": yte.numpy(), "ids": ids[te.numpy()], **preds}
    save_data.update({f"{name}_params": params for name, params in kalman_params.items()})
    np.savez(OUT / "reconstruction_predictions.npz", **save_data)
    if models:
        torch.save({name: m.state_dict() for name, (m, _, _) in models.items()}, OUT / "models.pt")
    true_bounce = yte.numpy()[:, 0]
    print(f"\nbounce stats corr (test episodes)\n{'':8s}" + "".join(f"{s:>16s}" for s in BOUNCE_STATS))
    for name, pr in preds.items():
        rs = [np.corrcoef(FNS[s](true_bounce, FS), FNS[s](pr[:, 0], FS))[0, 1] for s in BOUNCE_STATS]
        print(f"{name:8s}" + "".join(f"{r:16.3f}" for r in rs))
