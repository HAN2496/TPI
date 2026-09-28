"""센서 체인 진단 그림 3장 (Times New Roman, 영문 라벨, 제목 포함).

  diag1_imu_lag.png     휠속 미분과 IMU 종방향 가속도의 지연 상호상관
  diag2_imu_lever.png   pitch 각에 대한 IMU 종방향 응답 기울기 (중력만이면 +9.8)
  diag3_wheel_diff.png  앞뒤 wheel speed 차와 pitch rate 의 회귀

색: 기준선 검정, method 는 red, scatter 는 blue. 선 두께는 기준선 1.0, method 1.5.
"""
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt

from .pitch_staged_reconstruction import observations
from .run import OUTPUT, data, training_split

RED, BLUE, BLACK = "#d62728", "#3b7dd8", "#000000"
THIN, MAIN = 1.0, 1.5
mpl.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 12, "axes.labelsize": 12, "axes.titlesize": 13,
                     "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
                     "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
                     "xtick.direction": "in", "ytick.direction": "in", "axes.unicode_minus": False,
                     "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.03})
FIG = OUTPUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
SIZE = (4.3, 3.3)


def save(fig, name):  # 뷰어가 파일을 잡고 있을 때를 대비한 재시도
    for _ in range(6):
        try:
            fig.tight_layout(), fig.savefig(FIG / name, dpi=300), plt.close(fig)
            return print(name, "saved", flush=True)
        except OSError:
            time.sleep(1.0)
    raise RuntimeError(f"{name} 저장 실패")


cfg, x, y, ids, test = data()
fs = cfg.fs
train = training_split(ids, test)[0]
idx = train[np.linspace(0, len(train) - 1, 400).astype(int)]
obs = observations(x)
label = y[:, :, 2] / (180 / np.pi)
band = lambda s, lo, hi: sosfiltfilt(butter(2, (lo, hi), btype="bandpass", fs=fs, output="sos"), s, axis=1)
vdot = np.gradient(obs["vbar"][idx], 1 / fs, axis=1)
ax_imu, q, dvw = obs["ax"][idx], label[idx], obs["dvw"][idx]
sub = slice(None, None, 37)

# (1) IMU lag
a, b = band(vdot, 0.3, 5.0), band(ax_imu, 0.3, 5.0)
lags = np.arange(-15, 16)
cc = []
for shift in lags:
    u, v = (a[:, :a.shape[1] - shift], b[:, shift:]) if shift >= 0 else (a[:, -shift:], b[:, :b.shape[1] + shift])
    u, v = u - u.mean(1, keepdims=True), v - v.mean(1, keepdims=True)
    cc.append(np.median(np.sum(u * v, 1) / np.sqrt(np.sum(u * u, 1) * np.sum(v * v, 1))))
cc = np.array(cc)
peak = lags[int(np.argmax(cc))] * 1000 / fs
fig, ax = plt.subplots(figsize=SIZE)
ax.axvline(0, color=BLACK, lw=THIN, ls="--", zorder=1)
ax.plot(lags * 1000 / fs, cc, color=RED, lw=MAIN, zorder=3)
ax.plot([peak], [cc.max()], "o", color=RED, ms=4.0, zorder=4)
ax.annotate(f"{peak:.0f} ms", (peak, cc.max()), textcoords="offset points", xytext=(7, -1), color=RED)
ax.set(title="IMU lags the wheel-derived acceleration", xlabel="IMU lag [ms]", ylabel="Correlation", xlim=(-150, 150))
ax.grid(alpha=.25)
save(fig, "diag1_imu_lag.png")
print(f"(1) peak {cc.max():.3f} at {peak:.0f} ms")

# (2) IMU response to pitch angle
th = band(np.cumsum(q - q.mean(1, keepdims=True), 1) / fs, 0.2, 3.0)
res = band(ax_imu - vdot, 0.2, 3.0)
slope = np.sum(th * res) / np.sum(th * th)
fig, ax = plt.subplots(figsize=SIZE)
xs = np.array([-0.05, 0.05])
ax.scatter(th[:, sub].ravel(), res[:, sub].ravel(), s=1.6, color=BLUE, alpha=.18, linewidths=0, zorder=2)
ax.plot(xs, 9.81 * xs, color=BLACK, lw=THIN, zorder=3, label="Gravity only:  +9.8")
ax.plot(xs, slope * xs, color=RED, lw=MAIN, zorder=4, label=f"Measured:  {slope:.1f}")
ax.set(title="IMU sits off the pitch axis", xlabel=r"Pitch angle $\theta$ [rad]",
       ylabel=r"$a_{x,\mathrm{IMU}}-\dot{\bar{v}}_w$ [m/s$^2$]", xlim=(-.05, .05), ylim=(-1.4, 1.4))
ax.grid(alpha=.25), ax.legend(loc="upper left", handlelength=1.6)
save(fig, "diag2_imu_lever.png")
print(f"(2) slope {slope:.2f}")

# (3) Wheel speed difference vs pitch rate
qb, vb, db = band(q, 0.2, 3.0), band(vdot, 0.2, 3.0), band(dvw, 0.2, 3.0)
design = np.stack([qb.ravel(), vb.ravel(), np.ones(qb.size)], 1)
ell, kappa, offset = np.linalg.lstsq(design, db.ravel(), rcond=None)[0]
part = db - kappa * vb - offset
only_q = np.stack([qb.ravel(), np.ones(qb.size)], 1)
r2 = 1 - np.var(db.ravel() - only_q @ np.linalg.lstsq(only_q, db.ravel(), rcond=None)[0]) / np.var(db.ravel())
fig, ax = plt.subplots(figsize=SIZE)
xs = np.array([-0.6, 0.6])
ax.axhline(0, color=BLACK, lw=THIN, ls="--", zorder=1)
ax.scatter(qb[:, sub].ravel(), part[:, sub].ravel(), s=1.6, color=BLUE, alpha=.18, linewidths=0, zorder=2)
ax.plot(xs, ell * xs, color=RED, lw=MAIN, zorder=4, label=f"Slope:  {ell:.2f} m")
ax.set(title="Front-rear wheel speed difference carries pitch rate", xlabel=r"Pitch rate $q$ [rad/s]",
       ylabel=r"$\Delta v_w$ [m/s]", xlim=(-.6, .6), ylim=(-.26, .26))
ax.grid(alpha=.25), ax.legend(loc="upper left", handlelength=1.6)
save(fig, "diag3_wheel_diff.png")
print(f"(3) ell={ell:.3f} m, kappa={kappa:.4f} s, R2(q)={r2:.3f}")
