"""Per-signal time + frequency view. Hardcoded test script.
Col 1: time-domain (x=time). Col 2: Welch PSD linear-y.
Smoothing + downsampling applied (mirrors loader.View: smooth -> downsample).
Ride bands shaded: primary 0.5-5Hz, choppiness 5-10Hz, shake 10-25Hz."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from loader import Dataset

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["mathtext.default"] = "regular"

# ---------------- hardcoded config ----------------
DATA_ROOT = ROOT / "datasets"
DRIVER = "박재일" # 강신길, 박재일, 조현석, 한규택
EP_INDEX = 1

DOWNSAMPLE = 1            # 1 = no downsample. raw logging 100Hz -> fs = 100/DOWNSAMPLE
SMOOTH = False            # Butterworth low-pass before downsample (anti-alias)
SMOOTH_CUTOFF = 10.0     # Hz
SMOOTH_ORDER = 2

# IMU_LatAccelVal, IMU_LongAccelVal, IMU_RollRtVal, IMU_VerAccelVal, IMU_YawRtVal
# Pitch_rate_6D, Bounce_rate_6D, Roll_rate_6D
# "_dot" suffix = time derivative (np.gradient) of the base column.
# Pitch/Bounce/Roll_rate_6D are rates -> their _dot is angular acceleration (ddot theta).
CHANNELS = [
    "IMU_LongAccelVal", "IMU_LongAccelVal",
    "IMU_VerAccelVal", "IMU_VerAccelVal",
    "IMU_RollRtVal", "IMU_RollRtVal",
    "Pitch_rate_6D", "Pitch_rate_6D",
    "Bounce_rate_6D", "Bounce_rate_6D",
    "Roll_rate_6D", "Roll_rate_6D",
    "Pitch_rate_6D_dot", "Pitch_rate_6D_dot",
    "Bounce_rate_6D_dot", "Bounce_rate_6D_dot",
    "Roll_rate_6D_dot", "Roll_rate_6D_dot",

    "WHL_SpdRRVal", "WHL_SpdRRVal",
    "WHL_SpdFLVal", "WHL_SpdFLVal",
    "WHL_SpdFRVal", "WHL_SpdFRVal",
    "WHL_SpdRLVal", "WHL_SpdRLVal",

    "VCU_AccPedDepVal", "VCU_AccPedDepVal",
    "VCU_MotTqCmdRearVal", "VCU_MotTqCmdRearVal",
    "VCU_MotTqCmdFrntVal", "VCU_MotTqCmdFrntVal",

    "MCU_Mg1EstTqVal", "MCU_Mg1EstTqVal",
    "MCU_Mg2EstTqVal", "MCU_Mg2EstTqVal",
]
# per-row window in seconds. None = whole signal, (lo, hi) = slice only that range.
TIME_RANGE = [
    None, (4, 7),
    None, (4, 7),
    None, (4, 7),
    None, (4, 7),
    None, (4, 7),
    None, (4, 7),
    None, (4, 7),
    None, (4, 7),
    None, (4, 7),

    None, (4, 7),
    None, (4, 7),
    None, (4, 7),
    None, (4, 7),

    None, (4, 7),
    None, (4, 7),
    None, (4, 7),

    None, (4, 7),
    None, (4, 7),
]

DERIV = "_dot"
BANDS = [
    (0.5, 5.0, "primary", "C0"),
    (5.0, 10.0, "choppiness", "C1"),
    (10.0, 25.0, "shake", "C3"),
]
FREQ_XLIM = 30.0
NPERSEG = 512
# --------------------------------------------------

assert len(CHANNELS) == len(TIME_RANGE), "CHANNELS and TIME_RANGE length mismatch"
episodes = [e for e in Dataset(DATA_ROOT)[DRIVER] if e.label is not None]
assert episodes, f"no episodes for {DRIVER}"

base_of = {ch: ch[: -len(DERIV)] if ch.endswith(DERIV) else ch for ch in set(CHANNELS)}
keep = ["Time"] + sorted(set(base_of.values()))
series = {ch: [] for ch in CHANNELS}
times = []
fs = None
for ep in episodes:
    sig = ep.signals
    if SMOOTH:
        sig = sig.smoothed(SMOOTH_CUTOFF, SMOOTH_ORDER, keep[1:])
    df = sig.df[keep]
    if DOWNSAMPLE > 1:
        df = df.iloc[::DOWNSAMPLE].reset_index(drop=True)
    t = df["Time"].to_numpy(float)
    times.append(t)
    fs = 1.0 / np.median(np.diff(t))
    for ch in set(CHANNELS):
        s = df[base_of[ch]].to_numpy(float)
        series[ch].append(np.gradient(s, t) if ch.endswith(DERIV) else s)

ep_shown = episodes[EP_INDEX]
print(f"time-domain ep#{EP_INDEX}: {ep_shown.state}  (label={ep_shown.label})")

nyq = fs / 2.0
FGRID = np.linspace(0.0, nyq, 401)              # common grid for averaging, capped at Nyquist
xlim = min(FREQ_XLIM, nyq)


def slice_range(t, s, tr):                       # tr = None (whole) or (lo, hi) seconds
    if tr is None:
        return t, s
    m = (t >= tr[0]) & (t < tr[1])
    return t[m], s[m]


n = len(CHANNELS)
band_rows = []
fig, axes = plt.subplots(n, 2, figsize=(14, 2.4 * n))
for i, (ch, tr) in enumerate(zip(CHANNELS, TIME_RANGE)):
    ax_t, ax_flin = axes[i, 0], axes[i, 1]
    tag = "all" if tr is None else f"{tr[0]}-{tr[1]}s"

    # ---- time domain (representative episode = first) ----
    t0, sig = slice_range(times[EP_INDEX], series[ch][EP_INDEX], tr)
    ax_t.plot(t0, sig, lw=0.7, color="0.2")
    ax_t.set_ylabel(f"{ch}\n[{tag}]", fontsize=9)
    ax_t.set_xlabel("time [s]")
    ax_t.grid(alpha=0.3)

    # ---- frequency domain (mean Welch PSD over episodes) ----
    psds = []
    for tk, s in zip(times, series[ch]):
        _, seg = slice_range(tk, s, tr)
        f, p = welch(seg, fs=fs, nperseg=min(NPERSEG, len(seg)), detrend="constant")
        psds.append(np.interp(FGRID, f, p))
    psd = np.mean(psds, axis=0)

    # ---- band energy fraction (= variance share per ride band) ----
    e_total = np.trapezoid(psd, FGRID)
    frac = {name: np.trapezoid(psd[(FGRID >= lo) & (FGRID < hi)], FGRID[(FGRID >= lo) & (FGRID < hi)]) / e_total
            for lo, hi, name, _ in BANDS}
    band_rows.append((f"{ch} [{tag}]", frac))

    ax_flin.plot(FGRID, psd, lw=0.9, color="C2")
    ax_flin.set_ylim(0, None)
    ax_flin.set_xlim(0, xlim)
    ax_flin.set_xlabel("frequency [Hz]")
    ax_flin.set_ylabel("PSD")
    ax_flin.grid(alpha=0.3, which="both")
    for lo, hi, name, c in BANDS:
        ax_flin.axvspan(lo, hi, alpha=0.12, color=c, label=name if i == 0 else None)
    if i == 0:
        ax_flin.legend(fontsize=8, loc="upper right")

axes[0, 0].set_title(f"time  (driver={DRIVER}, ep#{EP_INDEX})")
axes[0, 1].set_title(f"PSD linear-y  (mean over {len(episodes)} eps, fs={fs:.0f}Hz)")
fig.tight_layout()
out = Path(__file__).resolve().parents[2] / "outputs/lab/dataset/signal_analysis.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=120)
print(f"saved {out}  (fs={fs:.1f}Hz, ds={DOWNSAMPLE}, smooth={SMOOTH}@{SMOOTH_CUTOFF}Hz, {len(episodes)} episodes)")

names = [b[2] for b in BANDS]
print("\nband energy fraction (share of total variance):")
print(f"  {'signal':24s}" + "".join(f"{nm:>12s}" for nm in names) + f"{'sum':>10s}")
for label, frac in band_rows:
    print(f"  {label:24s}" + "".join(f"{frac[nm]:>12.3f}" for nm in names) + f"{sum(frac.values()):>10.3f}")
