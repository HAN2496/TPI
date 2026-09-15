import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, correlate, correlation_lags, csd, sosfiltfilt, welch

from ..pitch_staged_reconstruction import SEALED, fit, signed_lag_ms, unpack
from ..run import data, training_split, write_csv
from ..state_space import GRAVITY, StateSpace, calibrate, discretize, discretize_input, highpass, innovation_metrics, metrics
from ..viz import plot_waveforms

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "outputs"
DEG = 180 / np.pi
WHEELBASE = 2.95  # 기존 lab 가정 (l_f 1.45 + l_r 1.50). 데이터에서 식별 실패 → 고정 가정 (md 참조)
PARAMS = {  # name: (start, bounds); log_ 접두는 exp 변환
    "log_qv": (0.0, (-8.0, 6.0)), "log_qb": (-6.0, (-16.0, 2.0)), "log_rv": (-2.0, (-10.0, 4.0)),
    "log_rz": (np.log(0.1), (-10.0, 4.0)),
    "log_qp": (np.log(0.1), (-8.0, 4.0)), "log_qx": (-9.0, (-16.0, 0.0)),
    "log_lam_a": (np.log(5.0), (np.log(0.5), np.log(50.0))), "log_qa": (np.log(0.5), (-6.0, 4.0)),
    "log_rx": (np.log(0.01), (-10.0, 2.0)), "log_rw": (np.log(0.05), (-10.0, 2.0)), "log_rd": (np.log(1e-3), (-14.0, 0.0)),
    "hi": (0.3, (-1.0, 1.0)), "ell": (-0.3, (-1.0, 1.0)), "kappa": (-0.05, (-0.5, 0.5)),
    "fh": (1.3, (0.6, 3.0)), "zh": (0.3, (0.05, 0.9)), "eps": (0.0, (-0.8, 0.8)), "log_j": (0.0, (np.log(0.3), np.log(3.0))),
    "gu": (0.0, (-5.0, 5.0)), "log_lam_r": (np.log(2.0), (np.log(0.1), np.log(50.0))), "log_qr": (-4.0, (-14.0, 4.0)),
    "log_qbody": (-6.0, (-16.0, 2.0)),
}
GRAV_NAMES = ["log_qp", "log_qx", "log_lam_a", "log_qa", "log_rx", "log_rw"]
HALF_NAMES = ["fh", "zh", "eps", "log_j", "gu", "log_lam_r", "log_qr", "log_qbody", "log_rz"]
plt.rcParams["axes.unicode_minus"] = False


def zero_phase(a, low, high, fs):
    return sosfiltfilt(butter(2, (low, high), btype="bandpass", fs=fs, output="sos"), a, axis=1)


def deriv(a, fs):
    return np.diff(a, axis=1, prepend=a[:, :1]) * fs


def delay(a, k):
    out = np.roll(a, k, axis=1)
    if k > 0:
        out[:, :k] = a[:, :1]
    if k < 0:
        out[:, k:] = a[:, -1:]
    return out


def channels(x, y):
    return dict(az=(x[:, :, 1] - 1) * GRAVITY, ax=x[:, :, 4] * GRAVITY, roll_imu=x[:, :, 0], yaw=x[:, :, 2],
                vf=x[:, :, 5:7].mean(2) / 3.6, vr=x[:, :, 7:9].mean(2) / 3.6, v=x[:, :, 5:9].mean(2) / 3.6,
                vside=(x[:, :, [6, 8]].mean(2) - x[:, :, [5, 7]].mean(2)) / 3.6,
                tf=x[:, :, 10], tr=x[:, :, 9], bounce=y[:, :, 0], roll6=y[:, :, 1], q=y[:, :, 2] / DEG)


# ---------------------------------------------------------------- stage 0: data facts
def lags_ms(a, b, fs, max_shift=60):  # 양수 = b 가 a 보다 늦다
    a, b = zero_phase(a, .3, 8, fs), zero_phase(b, .3, 8, fs)
    lag = correlation_lags(a.shape[1], b.shape[1])
    keep = np.abs(lag) <= max_shift
    return np.array([-lag[keep][np.argmax(correlate(u, w, method="fft")[keep])] for u, w in zip(a, b)]) * 1000 / fs


def transfer(a, b, fs, nperseg=256):
    f, saa = welch(a, fs, nperseg=nperseg, axis=1)
    sbb, sab = welch(b, fs, nperseg=nperseg, axis=1)[1], csd(a, b, fs, nperseg=nperseg, axis=1)[1]
    saa, sbb, sab = saa.mean(0), sbb.mean(0), sab.mean(0)
    return f, sab / saa, np.abs(sab) ** 2 / (saa * sbb)


def chip_filter(f, H, coh):  # bounce_6d ≈ sign·K·HP_fc(∫az)·e^{-j2πfτ}
    G = H * 2j * np.pi * f
    mid = (f >= 1) & (f <= 4)
    sign = np.sign(np.real(G[mid]).mean())
    ref = np.median(np.abs(G[(f >= 2) & (f <= 5)]))
    band = (f >= .3) & (f <= 5) & (coh > .5)
    grid = np.arange(.05, 2, .01)
    fc = grid[np.argmin([np.sum((np.abs(G[band]) / ref - f[band] / np.hypot(f[band], c)) ** 2) for c in grid])]
    sub = f >= .3
    hp = 1j * f[sub] / (fc + 1j * f[sub])
    phase = np.unwrap(np.angle(sign * G[sub] / hp))
    fit_band = (f[sub] >= 1) & (f[sub] <= 5)
    slope = np.polyfit(f[sub][fit_band], phase[fit_band], 1)[0]
    return sign, fc, -slope / (2 * np.pi) * 1000, sign * ref


def regress(y, X):
    M = np.stack([x.ravel() for x in X], 1)
    coef = np.linalg.lstsq(M, y.ravel(), rcond=None)[0]
    return coef, 1 - np.sum((y.ravel() - M @ coef) ** 2) / np.sum((y - y.mean()) ** 2)


def bump_timing(c, fs):  # 앞바퀴 휠속 스파이크 시각; 축거 추정 시도 = 앞→뒤 휠속 교차상관 지연 × 속도 (실패, md 참조)
    hf, hr = zero_phase(c["vf"], 1, 10, fs), zero_phase(c["vr"], 1, 10, fs)
    N, T = hf.shape
    t0 = np.clip(np.abs(c["az"]).argmax(1), 100, T - 200)
    win = t0[:, None] + np.arange(-100, 150)
    wf, wr = np.take_along_axis(hf, win, 1), np.take_along_axis(hr, win, 1)
    tf = win[np.arange(N), np.abs(wf).argmax(1)]
    lag = correlation_lags(win.shape[1], win.shape[1])
    keep = (lag <= -5) & (lag >= -80)
    dt = np.array([-lag[keep][np.argmax(correlate(u, w, method="fft")[keep])] for u, w in zip(wf, wr)]) / fs
    return tf, c["v"][np.arange(N), tf] * dt


def decay_fit(seg, fs):  # (f, zeta) 격자 × 선형 최소제곱 [cos, sin, 1] — free decay 로그 감쇠의 격자판
    t = np.arange(seg.shape[1]) / fs
    best, out = np.full(len(seg), np.inf), np.zeros((len(seg), 2))
    for fq in np.arange(.5, 4.01, .05):
        for z in np.arange(.02, .81, .02):
            w = 2 * np.pi * fq
            e = np.exp(-z * w * t)
            M = np.stack([e * np.cos(w * np.sqrt(1 - z * z) * t), e * np.sin(w * np.sqrt(1 - z * z) * t), np.ones_like(t)], 1)
            err = np.sum((seg.T - M @ np.linalg.lstsq(M, seg.T, rcond=None)[0]) ** 2, 0)
            better = err < best
            best[better], out[better] = err[better], (fq, z)
    return out


def psd_peak(a, fs):
    f, p = welch(a, fs, nperseg=512, axis=1)
    p = p.mean(0)
    band = (f >= .5) & (f <= 5)
    return f, p, f[band][np.argmax(p[band])]


def stage0(c, fs, train, ids):
    tr_ = train[np.linspace(0, len(train) - 1, 300).astype(int)]
    vdot = deriv(c["v"], fs)
    pairs = {"roll_imu->roll_6d": (c["roll_imu"], c["roll6"]), "wheel_side_diff->yaw_imu": (c["vside"], c["yaw"]),
             "wheel_accel->ax_imu": (vdot, c["ax"]), "torque->wheel_accel": (c["tf"] + c["tr"], vdot),
             "az_imu->d_bounce_6d": (c["az"], deriv(c["bounce"], fs))}
    rows, med = [], {}
    for name, (a, b) in pairs.items():
        p = np.percentile(lags_ms(a[train], b[train], fs), [10, 50, 90])
        rows.append(dict(pair=name, lag_p10_ms=p[0], lag_median_ms=p[1], lag_p90_ms=p[2]))
        med[name] = p[1]
    k = dict(k_wheel=int(round(med["wheel_side_diff->yaw_imu"] * fs / 1000)),
             k_label=-int(round(med["roll_imu->roll_6d"] * fs / 1000)))
    k["k_torque"] = k["k_wheel"] + int(round(med["torque->wheel_accel"] * fs / 1000))
    write_csv(OUTPUT / "stage0_alignment_lags.csv", rows)

    bp = lambda a: zero_phase(a, .3, 5, fs)
    qd, theta = deriv(c["q"], fs), np.cumsum(c["q"], 1) / fs
    lab = lambda a: bp(delay(a, k["k_label"]))[train]
    wheel_accel = bp(delay(vdot, k["k_wheel"]))[train]
    coef_x, r2_x = regress(bp(c["ax"])[train], [wheel_accel, lab(qd), lab(theta)])
    coef_x0, r2_x0 = regress(bp(c["ax"])[train], [wheel_accel])
    geometry = [dict(regression="ax ~ wheel_accel + qdot_label + theta_label (unconstrained; theta·qdot collinear in pitch band)",
                     coef=coef_x.tolist(), r2=r2_x, derived="unreliable", value=np.nan),
                dict(regression="ax ~ wheel_accel", coef=coef_x0.tolist(), r2=r2_x0, derived="-", value=np.nan)]
    for sign in (1, -1):  # 첫 실행에 쓴 방법 (g 고정, 부호 가설별 회귀) — theta·qdot 공선성으로 h_imu 가 틀림, 기록용
        coef, r2 = regress(bp(c["ax"])[train] - GRAVITY * sign * lab(theta), [wheel_accel, lab(qd)])
        geometry.append(dict(regression=f"[collinear, not used] ax - 9.81·({sign:+d}·theta_label) ~ wheel_accel + qdot_label", coef=coef.tolist(),
                             r2=r2, derived="h_imu [m] if this sign", value=-coef[1] * sign))
    low = lambda a: zero_phase(a, .05, .5, fs)[train]  # 저주파: 레버암 h·ω² ≪ g 라 중력누설만 남음 → 부호
    ax_body = c["ax"] - delay(vdot, k["k_wheel"])
    coef_low, r2_low = regress(low(ax_body), [low(delay(theta, k["k_label"]))])
    k["pitch_sign"] = int(np.sign(coef_low[0]))
    torque_corr = np.corrcoef(low(delay(theta, k["k_label"])).ravel(), low(c["tf"] + c["tr"]).ravel())[0, 1]
    geometry.append(dict(regression="[sign] ax - wheel_accel ~ theta_label, 0.05-0.5 Hz (expect ±g)", coef=coef_low.tolist(), r2=r2_low,
                         derived="pitch_sign = sign(coef); corr(theta_label, torque) low band", value=[k["pitch_sign"], torque_corr]))
    f, H, coh = transfer(delay(theta, k["k_label"])[train], ax_body[train], fs, nperseg=512)  # Re H = s_p(g + h ω²)
    band = (f >= .3) & (f <= 3) & (coh > .5)
    slope, intercept = np.polyfit((2 * np.pi * f[band]) ** 2, np.real(H[band]), 1)
    k["h_imu"], k["g_est"] = float(slope * k["pitch_sign"]), float(intercept * k["pitch_sign"])
    geometry.append(dict(regression="[used] Re H(theta_label -> ax - wheel_accel) = a + b·ω², 0.3-3 Hz, coh>0.5", coef=[intercept, slope], r2=np.nan,
                         derived="g_est = a·s_p, h_imu = b·s_p [m]", value=[k["g_est"], k["h_imu"]]))
    for name, target in (("ax", c["ax"]), ("ax - 0.6·wheel_accel", c["ax"] - .6 * delay(vdot, k["k_wheel"]))):
        f2, H2, coh2 = transfer(delay(theta, k["k_label"])[train], target[train], fs, nperseg=512)
        band2 = (f2 >= .3) & (f2 <= 3) & (coh2 > .5)
        slope2, intercept2 = np.polyfit((2 * np.pi * f2[band2]) ** 2, np.real(H2[band2]), 1)
        geometry.append(dict(regression=f"[sensitivity] Re H(theta_label -> {name}) = a + b·ω²", coef=[intercept2, slope2], r2=np.nan,
                             derived="g_est, h_imu [m]", value=[intercept2 * k["pitch_sign"], slope2 * k["pitch_sign"]]))
    coef_b, r2_b = regress(lab(deriv(c["bounce"], fs)), [bp(c["az"])[train], lab(qd)])
    k["x_total"] = float(-coef_b[1] / coef_b[0] / k["pitch_sign"])
    geometry.append(dict(regression="d_bounce ~ az + qdot_label", coef=coef_b.tolist(), r2=r2_b, derived="x_imu + x_6d [m]", value=k["x_total"]))
    write_csv(OUTPUT / "stage0_geometry_regression.csv", geometry)

    f, H, coh = transfer(c["az"][train], delay(c["bounce"], k["k_label"])[train], fs)
    az_corr = c["az"] - k["x_total"] * k["pitch_sign"] * delay(qd, k["k_label"])
    f, H2, coh2 = transfer(az_corr[train], delay(c["bounce"], k["k_label"])[train], fs)
    sign, fc, tau, gain = chip_filter(f, H2, coh2)
    k["bounce_sign"], k["chip_fc"], k["chip_tau_ms"], k["chip_gain"] = float(sign), float(fc), float(tau), float(gain)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for h, co, name in ((H, coh, "az_imu"), (H2, coh2, "az_imu - x_total·qdot")):
        axes[0].semilogx(f[1:], np.abs(h[1:]) * 2 * np.pi * f[1:] / abs(gain), label=name)
        axes[1].semilogx(f[1:], np.unwrap(np.angle(sign * h[1:] * 2j * np.pi * f[1:])) * DEG, label=name)
        axes[2].semilogx(f[1:], co[1:], label=name)
    axes[0].semilogx(f[1:], f[1:] / np.hypot(f[1:], fc), "k--", label=f"HP fc={fc:.2f} Hz")
    axes[1].semilogx(f[1:], (np.arctan(fc / f[1:]) - 2 * np.pi * f[1:] * tau / 1000) * DEG, "k--", label=f"HP + delay {tau:.0f} ms")
    for ax, title in zip(axes, ("|H|·2πf / K  (integrator removed)", "phase of H·j2πf [deg]", "coherence")):
        ax.set_title(title), ax.set_xlim(.2, 20), ax.grid(alpha=.3), ax.legend(fontsize=9)
    axes[2].set_ylim(0, 1)
    fig.suptitle("Stage 0: az_imu -> Bounce_rate_6D transfer function (train)")
    fig.tight_layout(), fig.savefig(OUTPUT / "stage0_transfer_function.png", dpi=150), plt.close(fig)

    tf, L = bump_timing(c, fs)
    speed = c["v"][np.arange(len(L)), tf]
    start = tf[tr_] + int(.6 * fs)  # 앞바퀴 통과 + 0.6 s ≈ 뒷바퀴 통과 후
    seg = lambda a: np.stack([a[i, s:s + int(2 * fs)] for i, s in zip(tr_, start) if s + int(2 * fs) <= a.shape[1]])
    modal_q, modal_z = decay_fit(seg(bp(c["q"] * DEG)), fs), decay_fit(seg(bp(c["az"])), fs)
    fq, pq, peak_q = psd_peak(c["q"][train] * DEG, fs)
    fz, pz, peak_z = psd_peak(c["az"][train], fs)
    k |= dict(L=WHEELBASE, L_attempt_median=float(np.median(L)), fp=float(np.median(modal_q[:, 0])), zp=float(np.median(modal_q[:, 1])),
              fh=float(np.median(modal_z[:, 0])), zh=float(np.median(modal_z[:, 1])), psd_peak_pitch=float(peak_q),
              psd_peak_az=float(peak_z), speed_median=float(np.median(speed)))
    write_csv(OUTPUT / "stage0_bump_modal.csv", [
        dict(quantity="wheelbase attempt [m] (front->rear wheel-speed lag × v; NOT used, L fixed 2.95)", p10=np.percentile(L, 10),
             median=k["L_attempt_median"], p90=np.percentile(L, 90), n=len(L)),
        dict(quantity="pitch free-decay f [Hz]", p10=np.percentile(modal_q[:, 0], 10), median=k["fp"], p90=np.percentile(modal_q[:, 0], 90), n=len(modal_q)),
        dict(quantity="pitch free-decay zeta", p10=np.percentile(modal_q[:, 1], 10), median=k["zp"], p90=np.percentile(modal_q[:, 1], 90), n=len(modal_q)),
        dict(quantity="heave (az) free-decay f [Hz]", p10=np.percentile(modal_z[:, 0], 10), median=k["fh"], p90=np.percentile(modal_z[:, 0], 90), n=len(modal_z)),
        dict(quantity="heave (az) free-decay zeta", p10=np.percentile(modal_z[:, 1], 10), median=k["zh"], p90=np.percentile(modal_z[:, 1], 90), n=len(modal_z)),
        dict(quantity="PSD peak pitch rate [Hz]", p10=np.nan, median=peak_q, p90=np.nan, n=len(train)),
        dict(quantity="PSD peak az [Hz]", p10=np.nan, median=peak_z, p90=np.nan, n=len(train)),
        dict(quantity="speed at bump [m/s]", p10=np.percentile(speed, 10), median=k["speed_median"], p90=np.percentile(speed, 90), n=len(L))])
    i = tr_[len(tr_) // 2]
    t = np.arange(c["az"].shape[1]) / fs
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, (name, a) in zip(axes[0], (("az_imu [m/s2]", c["az"][i]), ("Pitch_rate_6D [deg/s]", c["q"][i] * DEG),
                                       ("wheel speed HP 1-10 Hz [m/s]", zero_phase(c["vf"], 1, 10, fs)[i]))):
        ax.plot(t, a, lw=1, label=name)
        if "wheel" in name:
            ax.plot(t, zero_phase(c["vr"], 1, 10, fs)[i], lw=1, label="rear")
        ax.axvline(tf[i] / fs, color="r", ls="--", label="front hit (wheel-speed spike)")
        ax.axvspan(start[len(tr_) // 2] / fs, start[len(tr_) // 2] / fs + 2, color="gray", alpha=.15, label="decay window")
        ax.set_title(f"{ids[i]}  v={speed[i]:.1f} m/s"), ax.legend(fontsize=8), ax.grid(alpha=.3)
    axes[1, 0].hist(np.clip(L, -1, 5), 60), axes[1, 0].set_title(f"wheelbase attempt (median {k['L_attempt_median']:.2f} m), not used")
    axes[1, 1].hist(modal_q[:, 0], 30, alpha=.6, label="pitch"), axes[1, 1].hist(modal_z[:, 0], 30, alpha=.6, label="heave")
    axes[1, 1].set_title(f"free-decay f: pitch {k['fp']:.2f}, heave {k['fh']:.2f} Hz"), axes[1, 1].legend()
    axes[1, 2].semilogy(fq, pq / pq.max(), label="pitch rate"), axes[1, 2].semilogy(fz, pz / pz.max(), label="az")
    axes[1, 2].set_xlim(0, 15), axes[1, 2].set_title(f"PSD peaks: pitch {peak_q:.2f}, az {peak_z:.2f} Hz"), axes[1, 2].legend()
    fig.tight_layout(), fig.savefig(OUTPUT / "stage0_bump_and_modes.png", dpi=150), plt.close(fig)
    (OUTPUT / "stage0_constants.json").write_text(json.dumps(k, indent=1), encoding="utf-8")
    print(json.dumps(k, indent=1), flush=True)
    return k


# ---------------------------------------------------------------- state-space builders
def build_kinematic(d, fs, position):
    n = 3 if position else 2
    f, b = np.zeros((n, n)), np.zeros((n, 1))
    f[-2, -1], b[-2, 0] = -1, 1
    if position:
        f[0, 1] = 1
    qc = np.diag(([0] if position else []) + [d["qv"], d["qb"]])
    A, Q = discretize(f, qc, 1 / fs)
    R = np.diag(([d["rz"]] if position else []) + [d["rv"]])
    return StateSpace(A, np.eye(n)[:-1], Q, R, np.eye(n), B=discretize_input(f, b, 1 / fs)[1]), dict(v=n - 2)


def build_pitch(d, fs, wheel):  # [theta, q, bias_x(+g·grade), a_body, v_x]; y = [a_x_imu, v_wheel(, Δv_wheel)]
    wp, n = 2 * np.pi * d["fp"], 5
    f = np.zeros((n, n))
    f[0, 1], f[1, 0], f[1, 1], f[3, 3], f[4, 3] = 1, -wp * wp, -2 * d["zp"] * wp, -d["lam_a"], 1
    qc = np.diag([0, d["qp"], d["qx"], 2 * d["qa"] * d["lam_a"], 0])
    ax = np.zeros(n)
    ax[[0, 2, 3]] = GRAVITY, 1, 1
    ax[:2] -= d["hi"] * f[1, :2]
    h, R = [ax, np.eye(n)[4]], [d["rx"], d["rw"]]
    D = None
    if wheel:
        row = np.zeros(n)
        row[1], row[3] = d["ell"], d["kappa"]
        h, R = h + [row], R + [d["rd"]]
        D = np.zeros((3, 2))
        D[2] = d["sf"], d["sr"]
    A, Q = discretize(f, qc, 1 / fs)
    P = np.eye(n)
    P[2, 2], P[3, 3] = .01, d["qa"]
    return StateSpace(A, np.stack(h), Q, np.diag(R), P, B=np.zeros((n, 2)) if wheel else None, D=D), dict(q=1, vx=4)


def build_halfcar(d, fs, indep):  # [z, zd, theta, q, rf, rfd, rr, rrd, a_body, bias_x, v_x]
    wh, n = 2 * np.pi * d["fh"], 11
    af, ar = wh * wh * (1 + d["eps"]) / 2, wh * wh * (1 - d["eps"]) / 2
    bf, br = d["zh"] * wh * (1 + d["eps"]), d["zh"] * wh * (1 - d["eps"])
    lf = lr = d["L"] / 2
    jr = d["j"] * lf * lr
    front, rear = np.zeros(n), np.zeros(n)
    front[:6] = af, bf, af * lf, bf * lf, -af, -bf
    rear[[0, 1, 2, 3, 6, 7]] = ar, br, -ar * lr, -br * lr, -ar, -br
    f = np.zeros((n, n))
    f[0, 1] = f[2, 3] = f[4, 5] = f[6, 7] = 1
    f[1] = -(front + rear)
    f[3] = (-lf * front + lr * rear) / jr
    f[3, 8] += d["gu"]
    f[5, 5], f[8, 8], f[10, 8] = -d["lam_r"], -d["lam_a"], 1
    qc = np.zeros((n, n))
    qc[1, 1] = qc[3, 3] = d["qbody"]
    qc[5, 5], qc[8, 8], qc[9, 9] = 2 * d["qr"] * d["lam_r"], 2 * d["qa"] * d["lam_a"], d["qx"]
    P = np.eye(n)
    P[4:8, 4:8] *= 1e-4
    P[8, 8], P[9, 9] = d["qa"], .01
    if indep:
        f[7, 7], qc[7, 7] = -d["lam_r"], 2 * d["qr"] * d["lam_r"]
    else:
        P[6:8, 6:8] = 1e-8 * np.eye(2)
    ax = np.zeros(n)
    ax[[2, 8, 9]] = GRAVITY, 1, 1
    ax -= d["hi"] * f[3]
    dvw = np.zeros(n)
    dvw[3], dvw[8] = d["ell"], d["kappa"]
    h = np.stack([f[1] + d["x_imu"] * f[3], ax, np.eye(n)[10], dvw])
    D = np.zeros((4, 2))
    D[3] = d["sf"], d["sr"]
    A, Q = discretize(f, qc, 1 / fs)
    return StateSpace(A, h, Q, np.diag([d["rz"], d["rx"], d["rw"], d["rd"]]), P, B=np.zeros((n, 2)), D=D), dict(q=3, zd=1, rf=4, rr=6, vx=10)


def rear_replay(v, n, out, fs, L):
    distance = np.cumsum(np.maximum(v, 0), 1) / fs
    index = np.stack([np.searchsorted(dd, dd - L) for dd in distance])
    index[distance < L] = -1
    rows, ifr, ir = np.arange(len(v)), out["rf"], out["rr"]

    def replay(t, history):
        idx = index[:, t]
        valid = (idx >= 0) & (idx < t)
        target = np.zeros((len(v), 2))
        target[valid] = history[rows[valid], idx[valid], ifr:ifr + 2]
        add = np.zeros((len(v), n))
        add[:, ir:ir + 2] = target - (history[:, t - 1, ir:ir + 2] if t else 0)
        return add

    return replay


# ---------------------------------------------------------------- fitting / evaluation
def run_variant(spec, obs, index, p, fs):
    y = np.stack([obs[key][index] for key in spec["y"]], -1)
    u = np.stack([obs[key][index] for key in spec["u"]], -1) if spec["u"] else None
    state_space, out = spec["build"](unpack(spec["names"], p, spec["fixed"]), fs)
    extra = rear_replay(obs["v"][index], len(state_space.A), out, fs, spec["fixed"]["L"]) if spec["delay"] else None
    x0 = np.zeros((len(index), len(state_space.A)))
    if "vx" in out:
        x0[:, out["vx"]] = obs["vw"][index][:, 0]
    state, nu, cov = state_space.filter(y, u, x0, extra=extra, innovations=True)
    return state, nu, cov, out


def evaluate(pred, label, train, dev, fs, k_label, gain):
    g, c = calibrate(pred[train], label[train], gain)
    p = g * pred + c
    corr, rmse, _ = metrics(label[dev], p[dev], fs)
    aligned = metrics(delay(label, k_label)[dev], p[dev], fs)[0]
    cp = np.percentile(corr, [10, 50, 90])
    return dict(pred=p[dev], corr=corr, corr_p10=cp[0], corr_median=cp[1], corr_p90=cp[2], corr_aligned_median=np.median(aligned),
                rmse_median=np.median(rmse), signed_lag_ms=signed_lag_ms(label[dev], p[dev], fs),
                free_gain=calibrate(pred[train], label[train])[0])


def fit_and_evaluate(name, spec, obs, labels, split, fs, objectives, maxiter, restarts, rows, results, params):
    train, dev, fit_index, k_label = split
    target = labels[spec["target"]]
    for objective in objectives:
        started, key = time.perf_counter(), f"{name}_{objective}"

        def cost(p):
            state, nu, cov, out = run_variant(spec, obs, fit_index, p, fs)
            if objective == "ml":
                return innovation_metrics(nu, cov)["energy"]
            pred = state[..., out[spec["out"]]]
            gain, offset = calibrate(pred, target[fit_index], spec["gain"])
            return np.sqrt(np.mean((gain * pred + offset - target[fit_index]) ** 2)) / target[fit_index].std()

        start = [PARAMS[key2][0] for key2 in spec["names"]]
        bounds = [PARAMS[key2][1] for key2 in spec["names"]]
        loss, p = fit(cost, start, bounds, maxiter, restarts)
        state, nu, cov, out = run_variant(spec, obs, np.arange(len(target)), p, fs)
        white = innovation_metrics(nu[dev], cov)
        d = unpack(spec["names"], p, spec["fixed"])
        params[key] = {key2: float(d[key2]) for key2 in d}
        outputs = {spec["target"]: state[..., out[spec["out"]]]}
        if "bounce_out" in spec:
            outputs["bounce"] = highpass(state[..., out["zd"]] - spec["fixed"]["x_6d"] * state[..., out["q"]], spec["fixed"]["chip_fc"], fs)
        if "oracle" in spec:
            outputs["bounce_oracle_q"] = outputs["bounce"] - spec["fixed"]["x_total"] * spec["fixed"]["pitch_sign"] * delay(labels["q"], k_label)
        for target_name, raw in outputs.items():
            label = labels["bounce" if target_name.startswith("bounce") else target_name]
            gain = spec["gain"] if target_name == "pitch" else None
            r = evaluate(raw, label, train, dev, fs, k_label, gain)
            results[(target_name, key)] = r
            rows.append(dict(model=name, objective=objective, target=target_name, loss=loss, energy_dev=white["energy"],
                             nis=white["nis"], acf1=white["acf1"], free_gain=r["free_gain"], corr_p10=r["corr_p10"],
                             corr_median=r["corr_median"], corr_p90=r["corr_p90"], corr_aligned_median=r["corr_aligned_median"],
                             rmse_median=r["rmse_median"], signed_lag_ms=r["signed_lag_ms"],
                             fit_seconds=time.perf_counter() - started,
                             parameters=" ".join(f"{key2}={d[key2]:.5g}" for key2 in d)))
            print(f"{key:24s} {target_name:16s} corr={r['corr_median']:.3f} aligned={r['corr_aligned_median']:.3f} "
                  f"rmse={r['rmse_median']:.3f} lag={r['signed_lag_ms']:+.0f}ms free_gain={r['free_gain']:+.2f} "
                  f"nis={white['nis']:.2f} acf1={white['acf1']:+.2f} loss={loss:.4g} ({rows[-1]['fit_seconds']:.0f}s)", flush=True)


def waveform_plots(labels, results, ids, dev, fs, stage, reference):
    for target_name in ("bounce", "pitch"):
        picked = {key[1]: r for key, r in results.items() if key[0] == target_name}
        if picked:
            ref = reference if reference in picked else tuple(picked)[-1]
            plot_waveforms(labels[target_name][dev], picked, ids[dev], fs, OUTPUT / f"stage{stage}_{target_name}_waveforms.png", ref)


def observations(c, k, fs):  # 휠속·토크를 측정 지연만큼 늦춰 IMU 시간축에 정렬
    v = delay(c["v"], k["k_wheel"])
    return dict(az=c["az"], ax=c["ax"], vdot=delay(deriv(c["v"], fs), k["k_wheel"]), v=v, vw=v,
                dvw=delay(c["vf"] - c["vr"], k["k_wheel"]), tf=delay(c["tf"], k["k_torque"]), tr=delay(c["tr"], k["k_torque"]),
                zero=np.zeros_like(c["az"]))


def geometry_split(k):
    x_imu = float(np.clip(k["x_total"] / 4, 0, .5))
    return dict(x_imu=x_imu, x_6d=k["x_total"] - x_imu, x_total=k["x_total"], hi=k["h_imu"], L=k["L"], chip_fc=k["chip_fc"],
                pitch_sign=k["pitch_sign"])


# ---------------------------------------------------------------- stages 1-4
def stage1(obs, labels, split, fs, k, args):
    rows, results, params = [], {}, {}
    common, gain = dict(fp=k["fp"], zp=k["zp"]), k["pitch_sign"] * DEG
    specs = {
        "bounce_kin_v": dict(build=lambda d, fs: build_kinematic(d, fs, False), names=["log_qv", "log_qb", "log_rv"], fixed={},
                             y=["zero"], u=["az"], target="bounce", out="v", gain=None, delay=False),
        "bounce_kin_zv": dict(build=lambda d, fs: build_kinematic(d, fs, True), names=["log_qv", "log_qb", "log_rz", "log_rv"],
                              fixed=dict(x_total=k["x_total"], pitch_sign=k["pitch_sign"]), y=["zero", "zero"], u=["az"], target="bounce",
                              out="v", gain=None, delay=False, oracle=True),
        "pitch_grav_h0": dict(build=lambda d, fs: build_pitch(d, fs, False), names=GRAV_NAMES, fixed=common | dict(hi=0.0),
                              y=["ax", "vw"], u=None, target="pitch", out="q", gain=gain, delay=False),
        "pitch_grav_hi": dict(build=lambda d, fs: build_pitch(d, fs, False), names=GRAV_NAMES, fixed=common | dict(hi=k["h_imu"]),
                              y=["ax", "vw"], u=None, target="pitch", out="q", gain=gain, delay=False),
        "pitch_grav_hfree": dict(build=lambda d, fs: build_pitch(d, fs, False), names=GRAV_NAMES + ["hi"], fixed=common,
                                 y=["ax", "vw"], u=None, target="pitch", out="q", gain=gain, delay=False),
    }
    for name, spec in specs.items():
        fit_and_evaluate(name, spec, obs, labels, split, fs, args.objectives, args.maxiter, args.restarts, rows, results, params)
    write_csv(OUTPUT / "stage1_metrics.csv", rows)
    (OUTPUT / "stage1_parameters.json").write_text(json.dumps(params, indent=1), encoding="utf-8")
    waveform_plots(labels, results, args.ids, split[1], fs, 1, "pitch_grav_hi_ml")


def stage2(obs, labels, split, fs, k, args):
    train, dev, fit_index, k_label = split
    bp = lambda a: zero_phase(a, .3, 5, fs)
    q, v = k["pitch_sign"] * delay(labels["q"], k_label), obs["v"]  # q: nose-up 양수 물리 부호
    X = dict(q=bp(q), wheel_accel=bp(obs["vdot"]), torque_front=bp(obs["tf"]), torque_rear=bp(obs["tr"]),
             v_x_az=bp(v * obs["az"]), v_x_qdot=bp(v * deriv(q, fs)))
    y = bp(obs["dvw"])
    speed = v.mean(1)
    edges = np.percentile(speed[train], [0, 33, 66, 100])
    rows = []
    for name, index in [("all train", train)] + [(f"speed {edges[i]:.1f}-{edges[i + 1]:.1f} m/s",
                                                    train[(speed[train] >= edges[i]) & (speed[train] <= edges[i + 1])]) for i in range(3)]:
        coef, r2 = regress(y[index], [X[key][index] for key in X])
        coef_q, r2_q = regress(y[index], [X["q"][index]])
        rows.append(dict(subset=name, n=len(index), r2=r2, r2_q_only=r2_q, ell_q_only_m=coef_q[0],
                         **{f"coef_{key}": value for key, value in zip(X, coef)}))
    write_csv(OUTPUT / "stage2_wheel_speed_regression.csv", rows)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for key in ("coef_q", "coef_wheel_accel", "coef_torque_front", "coef_torque_rear"):
        ax[0].plot(range(1, 4), [row[key] for row in rows[1:]], "o-", label=key)
    ax[0].set_xticks(range(1, 4), [row["subset"] for row in rows[1:]], fontsize=8), ax[0].legend(fontsize=8), ax[0].grid(alpha=.3)
    ax[0].set_title("Δv_w regression coefficients by speed tercile")
    i = train[np.argsort(metrics(y[train], X["q"][train] * rows[0]["coef_q"], fs)[0])[len(train) // 2]]
    t = np.arange(y.shape[1]) / fs
    ax[1].plot(t, y[i], "k", lw=1, label="Δv_w (BP 0.3-5 Hz)")
    ax[1].plot(t, sum(X[key][i] * rows[0][f"coef_{key}"] for key in X), "r", lw=1, label=f"regression (R²={rows[0]['r2']:.2f})")
    ax[1].plot(t, X["q"][i] * rows[0]["coef_q"], "b--", lw=1, label="ℓ·q term only")
    ax[1].legend(fontsize=8), ax[1].set_title(f"{args.ids[i]}"), ax[1].grid(alpha=.3)
    fig.tight_layout(), fig.savefig(OUTPUT / "stage2_wheel_speed_regression.png", dpi=150), plt.close(fig)

    wheel = dict(ell=rows[0]["coef_q"], kappa=rows[0]["coef_wheel_accel"], sf=rows[0]["coef_torque_front"], sr=rows[0]["coef_torque_rear"])
    common, gain = dict(fp=k["fp"], zp=k["zp"], hi=k["h_imu"]), k["pitch_sign"] * DEG
    results, params, metric_rows = {}, {}, []
    specs = {
        "pitch_wheel": dict(build=lambda d, fs: build_pitch(d, fs, True), names=GRAV_NAMES + ["log_rd"], fixed=common | wheel,
                            y=["ax", "vw", "dvw"], u=["tf", "tr"], target="pitch", out="q", gain=gain, delay=False),
        "pitch_wheel_free": dict(build=lambda d, fs: build_pitch(d, fs, True), names=GRAV_NAMES + ["log_rd", "ell", "kappa"],
                                 fixed=common | dict(sf=wheel["sf"], sr=wheel["sr"]), y=["ax", "vw", "dvw"], u=["tf", "tr"],
                                 target="pitch", out="q", gain=gain, delay=False),
    }
    for name, spec in specs.items():
        fit_and_evaluate(name, spec, obs, labels, split, fs, args.objectives, args.maxiter, args.restarts, metric_rows, results, params)
    write_csv(OUTPUT / "stage2_metrics.csv", metric_rows)
    (OUTPUT / "stage2_parameters.json").write_text(json.dumps(params, indent=1), encoding="utf-8")
    waveform_plots(labels, results, args.ids, dev, fs, 2, "pitch_wheel_ml")


def halfcar_fixed(k, stage2_params):
    base = stage2_params["pitch_wheel_ml"]
    keep = ("qx", "lam_a", "qa", "rx", "rw", "rd", "ell", "kappa", "sf", "sr")
    return {key: base[key] for key in keep} | geometry_split(k)


def stage3(obs, labels, split, fs, k, args):
    fixed = halfcar_fixed(k, json.loads((OUTPUT / "stage2_parameters.json").read_text(encoding="utf-8")))
    PARAMS["fh"], PARAMS["zh"] = (k["fh"], PARAMS["fh"][1]), (k["zh"], PARAMS["zh"][1])
    rows, results, params, gain = [], {}, {}, k["pitch_sign"] * DEG
    specs = {
        "halfcar_delay": dict(build=lambda d, fs: build_halfcar(d, fs, False), names=HALF_NAMES, fixed=fixed,
                              y=["az", "ax", "vw", "dvw"], u=["tf", "tr"], target="pitch", out="q", gain=gain, delay=True, bounce_out=True),
        "halfcar_indep": dict(build=lambda d, fs: build_halfcar(d, fs, True), names=HALF_NAMES, fixed=fixed,
                              y=["az", "ax", "vw", "dvw"], u=["tf", "tr"], target="pitch", out="q", gain=gain, delay=False, bounce_out=True),
    }
    for name, spec in specs.items():
        fit_and_evaluate(name, spec, obs, labels, split, fs, args.objectives, args.maxiter, args.restarts, rows, results, params)
    write_csv(OUTPUT / "stage3_metrics.csv", rows)
    (OUTPUT / "stage3_parameters.json").write_text(json.dumps(params, indent=1), encoding="utf-8")
    waveform_plots(labels, results, args.ids, split[1], fs, 3, "halfcar_delay_ml")
    key = "halfcar_delay_ml" if "halfcar_delay_ml" in params else tuple(params)[0]
    spec = specs[key.rsplit("_", 1)[0]]
    p = [np.log(params[key][n[4:]]) if n.startswith("log_") else params[key][n] for n in spec["names"]]
    dev = split[1]
    order = np.argsort(results[("pitch", key)]["corr"])
    i = np.flatnonzero(dev)[order[len(order) // 2]]
    state, nu, cov, out = run_variant(spec, obs, np.array([i]), p, fs)
    s, t = state[0], np.arange(state.shape[1]) / fs
    fig, axes = plt.subplots(3, 2, figsize=(14, 9))
    panels = [("pitch rate [deg/s, label sign]", k["pitch_sign"] * DEG * s[:, out["q"]], labels["pitch"][i]),
              ("bounce at 6D (chip HP) [calibrated]", results[("bounce", key)]["pred"][order[len(order) // 2]], labels["bounce"][i]),
              ("road front / rear [m]", s[:, out["rf"]], s[:, out["rr"]]), ("theta [deg], z [cm]", DEG * s[:, 2], 100 * s[:, 0]),
              ("a_body [m/s2] vs wheel accel", s[:, 8], obs["vdot"][i]), ("az fit [m/s2]", (state[0] @ spec["build"](unpack(spec["names"], p, spec["fixed"]), fs)[0].H.T)[:, 0], obs["az"][i])]
    for ax, (title, a, b) in zip(axes.flat, panels):
        ax.plot(t, b, "k", lw=1, label="recorded / second"), ax.plot(t, a, "r", lw=1, label="estimate / first")
        ax.set_title(title), ax.grid(alpha=.3), ax.legend(fontsize=8)
    fig.suptitle(f"Stage 3 {key} states, {args.ids[i]}"), fig.tight_layout()
    fig.savefig(OUTPUT / "stage3_halfcar_states_median.png", dpi=150), plt.close(fig)


def simulate(state_space, u, replay, x0, rng):
    N, T = u.shape[:2]
    n, m = len(state_space.A), state_space.H.shape[0]
    Lq, Lr = np.linalg.cholesky(state_space.Q + 1e-14 * np.eye(n)), np.linalg.cholesky(state_space.R)
    x, y, s = np.zeros((N, T, n)), np.zeros((N, T, m)), x0.copy()
    for t in range(T):
        pred = s @ state_space.A.T + u[:, t] @ state_space.B.T
        if replay is not None:
            pred = pred + replay(t, x)
        s = pred + rng.standard_normal((N, n)) @ Lq.T
        x[:, t] = s
        y[:, t] = s @ state_space.H.T + u[:, t] @ state_space.D.T + rng.standard_normal((N, m)) @ Lr.T
    return x, y


def stage4(obs, labels, split, fs, k, args):
    train, dev, fit_index, k_label = split
    fixed = halfcar_fixed(k, json.loads((OUTPUT / "stage2_parameters.json").read_text(encoding="utf-8")))
    truth = json.loads((OUTPUT / "stage3_parameters.json").read_text(encoding="utf-8"))["halfcar_delay_ml"]
    spec = dict(build=lambda d, fs: build_halfcar(d, fs, False), names=HALF_NAMES, fixed=fixed, y=["az", "ax", "vw", "dvw"],
                u=["tf", "tr"], target="pitch", out="q", gain=k["pitch_sign"] * DEG, delay=True)
    p_true = np.array([np.log(truth[n[4:]]) if n.startswith("log_") else truth[n] for n in HALF_NAMES])
    state_space, out = build_halfcar(truth, fs, False)
    u = np.stack([obs[key][fit_index] for key in spec["u"]], -1)
    x0 = np.zeros((len(fit_index), len(state_space.A)))
    x0[:, out["vx"]] = obs["vw"][fit_index][:, 0]
    x, y = simulate(state_space, u, rear_replay(obs["v"][fit_index], len(state_space.A), out, fs, fixed["L"]), x0, np.random.default_rng(0))
    synthetic = {key: y[..., i] for i, key in enumerate(spec["y"])} | dict(v=obs["v"][fit_index], tf=u[..., 0], tr=u[..., 1])
    local = np.arange(len(fit_index))

    def cost(p):
        return innovation_metrics(*run_variant(spec, synthetic, local, p, fs)[1:3])["energy"]

    started = time.perf_counter()
    loss, p_hat = fit(cost, [PARAMS[n][0] for n in HALF_NAMES], [PARAMS[n][1] for n in HALF_NAMES], args.maxiter, args.restarts)
    q_hat = run_variant(spec, synthetic, local, p_hat, fs)[0][..., out["q"]]
    corr = np.median(metrics(x[..., out["q"]], q_hat, fs)[0])
    rows = [dict(parameter=n, truth=truth[n[4:]] if n.startswith("log_") else truth[n], start=np.exp(PARAMS[n][0]) if n.startswith("log_") else PARAMS[n][0],
                 recovered=np.exp(v) if n.startswith("log_") else v, energy_truth=cost(p_true), energy_recovered=loss,
                 corr_q_true_vs_hat=corr, fit_seconds=time.perf_counter() - started) for n, v in zip(HALF_NAMES, p_hat)]
    write_csv(OUTPUT / "stage4_synthetic_recovery.csv", rows)
    for row in rows:
        print(f"recovery {row['parameter']:10s} truth={row['truth']:.4g} recovered={row['recovered']:.4g}", flush=True)

    profile = []
    base = [np.log(truth[n[4:]]) if n.startswith("log_") else truth[n] for n in HALF_NAMES]
    free = ["log_qbody", "log_rz"]
    for x_total in (0.0, 0.5, 1.0, 1.5):
        for hi in (-0.2, 0.0, 0.2, 0.4):
            geometry = geometry_split(k | dict(x_total=x_total)) | dict(hi=hi)
            spec_p = spec | dict(names=free, fixed=fixed | geometry | {n[4:] if n.startswith("log_") else n: (np.exp(v) if n.startswith("log_") else v)
                                                                       for n, v in zip(HALF_NAMES, base) if n not in free})

            def cost_p(p):
                return innovation_metrics(*run_variant(spec_p, obs, fit_index, p, fs)[1:3])["energy"]

            loss_p, p_p = fit(cost_p, [base[HALF_NAMES.index(n)] for n in free], [PARAMS[n][1] for n in free], 4, 1)
            state, nu, cov, out_p = run_variant(spec_p, obs, np.arange(len(labels["pitch"])), p_p, fs)
            r = evaluate(state[..., out_p["q"]], labels["pitch"], train, dev, fs, k_label, k["pitch_sign"] * DEG)
            b = evaluate(highpass(state[..., out_p["zd"]] - geometry["x_6d"] * state[..., out_p["q"]], k["chip_fc"], fs), labels["bounce"], train, dev, fs, k_label, None)
            profile.append(dict(x_total=x_total, h_imu=hi, energy_fit=loss_p, energy_dev=innovation_metrics(nu[dev], cov)["energy"],
                                pitch_corr=r["corr_median"], pitch_free_gain=r["free_gain"], bounce_corr=b["corr_median"]))
            print(f"profile x_total={x_total} h_imu={hi} energy={loss_p:.4f} pitch={r['corr_median']:.3f} bounce={b['corr_median']:.3f}", flush=True)
    write_csv(OUTPUT / "stage4_geometry_profile.csv", profile)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for hi in (-0.2, 0.0, 0.2, 0.4):
        sub = [row for row in profile if row["h_imu"] == hi]
        for ax, key in zip(axes, ("energy_dev", "pitch_corr", "bounce_corr")):
            ax.plot([row["x_total"] for row in sub], [row[key] for row in sub], "o-", label=f"h_imu={hi}")
    for ax, key in zip(axes, ("innovation energy (dev, lower=better)", "pitch corr median", "bounce corr median")):
        ax.set_title(key), ax.set_xlabel("x_imu + x_6d [m]"), ax.grid(alpha=.3), ax.legend(fontsize=8)
    fig.suptitle("Stage 4: profile over sensor geometry (noise refit only)"), fig.tight_layout()
    fig.savefig(OUTPUT / "stage4_geometry_profile.png", dpi=150), plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("0", "1", "2", "3", "4", "all"))
    parser.add_argument("--objectives", default="ml,sup")
    parser.add_argument("--maxiter", type=int, default=30)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--fit-episodes", type=int, default=200)
    args = parser.parse_args()
    args.objectives = tuple(value.strip() for value in args.objectives.split(","))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cfg, x, y, ids, test = data()
    args.ids = ids
    c = channels(x, y)
    drivers = np.array([value.split()[0] for value in ids])
    dev = test & ~np.isin(drivers, SEALED)
    train, validation, validation_driver = training_split(ids, test)
    fit_index = train[np.linspace(0, len(train) - 1, min(args.fit_episodes, len(train))).astype(int)]
    print(f"train={len(train)} fit={len(fit_index)} dev-test={dev.sum()} sealed={(test & np.isin(drivers, SEALED)).sum()} "
          f"validation={validation_driver}", flush=True)
    stages = ("0", "1", "2", "3", "4") if args.stage == "all" else (args.stage,)
    k = stage0(c, cfg.fs, train, ids) if "0" in stages else json.loads((OUTPUT / "stage0_constants.json").read_text(encoding="utf-8"))
    obs = observations(c, k, cfg.fs)
    labels = dict(bounce=c["bounce"], pitch=c["q"] * DEG, q=c["q"])
    split = (train, dev, fit_index, k["k_label"])
    for stage in stages:
        if stage != "0":
            {"1": stage1, "2": stage2, "3": stage3, "4": stage4}[stage](obs, labels, split, cfg.fs, k, args)
    print(f"outputs={OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
