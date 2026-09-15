"""Results 슬라이드용 그림 3장: pb2_basic (2.5 결합 모델) 의 플랜트를 sup2 (discriminative) 로 고정하고
Q, R 만 EM / discriminative 로 각각 정했을 때의 파형·수렴궤적·상관계수 분포.

  result_waveform_*.png          median / worst episode 의 pitch rate / bounce rate 실제 출력 파형, _bump 는 4.5–6 s 확대
  result_em_convergence.png      EM 반복에 따른 sensor 우도와 pitch 상관계수
  result_corr_distribution.png   episode 별 상관계수 분포 (boxplot + 산점)

색·선: GT 검정 점선, EM 파랑 점쇄선, Discriminative 빨강 실선 (1.5 배 굵기). Times New Roman, 영문 라벨.
`--redraw` 를 주면 parameter_estimation_results.npz 와 _em_track.json 을 읽어 EM 재실행 없이 그림만 다시 그린다.
"""
import csv
import json
import sys
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .pitch_staged_reconstruction import DEG, SEALED, VARIANTS, observations
from .run import OUTPUT, data, training_split
from .run_em_noise_covariance import em
from .state_space import calibrate, innovation_metrics, metrics

RED, BLUE, BLACK, GREY = "#d62728", "#3b7dd8", "#000000", "#bbbbbb"
THIN, MAIN = 1.0, 1.5
mpl.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 12, "axes.labelsize": 12, "axes.titlesize": 13,
                     "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
                     "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
                     "xtick.direction": "in", "ytick.direction": "in", "axes.unicode_minus": False,
                     "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.03})
FIG = OUTPUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
MODEL, SOURCE, CHUNK, CHUNKS, FS = "pb2_basic", "sup2", 100, 26, 100.0
KEYS = ("pitch", "bounce", "corr", "bcorr")
NPZ, TRACK = OUTPUT / "parameter_estimation_results.npz", OUTPUT / "parameter_estimation_em_track.json"


def save(fig, name):  # 뷰어가 파일을 잡고 있을 때를 대비한 재시도
    for _ in range(6):
        try:
            fig.tight_layout(), fig.savefig(FIG / name, dpi=300), plt.close(fig)
            return print(name, "saved", flush=True)
        except OSError:
            time.sleep(1.0)
    raise RuntimeError(f"{name} 저장 실패")


if "--redraw" in sys.argv:
    z = np.load(NPZ)
    label, bounce = z["label"], z["bounce"]
    disc = {k: z[f"disc_{k}"] for k in KEYS}
    emr = {k: z[f"em_{k}"] for k in KEYS}
    track = json.loads(TRACK.read_text(encoding="utf-8"))
else:
    cfg, x, y, ids, test = data()
    drivers = np.array([value.split()[0] for value in ids])
    dev = test & ~np.isin(drivers, SEALED)
    train = training_split(ids, test)[0]
    fit_index = train[np.linspace(0, len(train) - 1, 200).astype(int)]
    full_label, full_bounce, obs = y[:, :, 2], y[:, :, 0], observations(x)
    spec = VARIANTS[MODEL]
    bidx = spec["bounce_index"]
    with (OUTPUT / "pitch_staged_metrics.csv").open(encoding="utf-8-sig") as stream:
        fits = {r["objective"]: r["parameters"] for r in csv.DictReader(stream) if r["model"] == MODEL}
    d = dict(spec["fixed"]) | {i.split("=")[0]: float(i.split("=")[1]) for i in fits[SOURCE].split()}
    ss0 = spec["build"]({(k[4:] if k.startswith("log_") else k): v for k, v in d.items()}, cfg.fs)
    yv = np.stack([obs[channel] for channel in spec["channels"]], -1)
    x0 = np.zeros((len(yv), len(ss0.A)))
    x0[:, 0] = yv[:, 0, 0]

    def evaluate(ss):
        state, nu, cov = ss.filter(yv, None, x0, innovations=True)
        q = state[..., 3]
        pitch = DEG * q + full_label[train].mean() - DEG * q[train].mean()
        gb, ob = calibrate(state[train][..., bidx], full_bounce[train])
        bnc = gb * state[..., bidx] + ob
        corr, rmse, _ = metrics(full_label[dev], pitch[dev], cfg.fs)
        return dict(pitch=pitch[dev], bounce=bnc[dev], corr=corr, rmse=rmse,
                    bcorr=metrics(full_bounce[dev], bnc[dev], cfg.fs)[0],
                    gain=calibrate(q[train], full_label[train])[0], **innovation_metrics(nu[dev], cov))

    report = lambda tag, r: print(f"{tag}: corr={np.median(r['corr']):.3f} bounce={np.median(r['bcorr']):.3f} "
                                  f"rmse={np.median(r['rmse']):.2f} nis={r['nis']:.2f} energy={r['energy']:.3f} "
                                  f"gain={r['gain']:.1f}", flush=True)
    disc = evaluate(ss0)
    report("discriminative", disc)
    start = innovation_metrics(*ss0.filter(yv[fit_index], None, x0[fit_index], innovations=True)[1:])["energy"]
    ss, track = ss0, [dict(iters=0, energy=float(start), corr=float(np.median(disc["corr"])),
                           bcorr=float(np.median(disc["bcorr"])))]
    for chunk in range(CHUNKS):
        started = time.perf_counter()
        ss, history = em(ss, yv[fit_index], None, x0[fit_index], CHUNK, 0.0, False)
        state = ss.filter(yv[dev], None, x0[dev])
        gb, ob = calibrate(state[..., bidx], full_bounce[dev])
        track.append(dict(iters=CHUNK * (chunk + 1), energy=float(history[-1]),
                          corr=float(np.median(metrics(full_label[dev], DEG * state[..., 3], cfg.fs)[0])),
                          bcorr=float(np.median(metrics(full_bounce[dev], gb * state[..., bidx] + ob, cfg.fs)[0]))))
        print(f"  EM {track[-1]['iters']:5d} it: energy={track[-1]['energy']:+.4f} corr={track[-1]['corr']:.3f} "
              f"bounce={track[-1]['bcorr']:.3f} ({time.perf_counter() - started:.0f}s)", flush=True)
        if len(history) < CHUNK or abs(history[-1] - history[-2]) < 1e-6 * abs(history[-2]):
            break
    emr = evaluate(ss)
    report("EM", emr)
    label, bounce = full_label[dev], full_bounce[dev]
    TRACK.write_text(json.dumps(track, indent=1), encoding="utf-8")
    np.savez_compressed(NPZ, label=label, bounce=bounce,
                        **{f"disc_{k}": disc[k] for k in KEYS}, **{f"em_{k}": emr[k] for k in KEYS})

# ── 그림 1: 파형 (pitch, bounce). episode 는 discriminative pitch corr 순위로 고름
#    mode: full = 10 s 전체, zoom = bump 구간만, inset = 10 s 전체 + pitch 패널에 bump 구간 확대 inset
order = np.argsort(disc["corr"])
t = np.arange(label.shape[1]) / FS
panels = ((label, "pitch", "corr", "Pitch rate [deg/s]", "deg/s"), (bounce, "bounce", "bcorr", "Bounce rate [-]", ""))
mid, worst, bump = order[len(order) // 2], order[0], (4.5, 6.0)
figures = (("result_waveform_mid.png", mid, "median", "full"), ("result_waveform_worst.png", worst, "worst", "full"),
           ("result_waveform_mid_bump.png", mid, "median", "zoom"), ("result_waveform_worst_bump.png", worst, "worst", "zoom"),
           ("result_waveform_mid_inset.png", mid, "median", "inset"), ("result_waveform_worst_inset.png", worst, "worst", "inset"))
for name, ep, rank, mode in figures:  # 실제 필터 출력 그대로 (pitch: 단위 변환 + train offset, bounce: train 이득)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6))
    span = f", {bump[0]}–{bump[1]} s" if mode == "zoom" else ""
    titles = (f"Pitch rate: {rank} episode{span}", f"Bounce rate: same episode{span}")
    for ax, title, (truth, key, ck, ylab, unit) in zip(axes, titles, panels):
        zoom = ax if mode == "zoom" else ax.inset_axes([.64, .17, .34, .47]) if mode == "inset" and key == "pitch" else None
        for target in (ax,) if zoom in (None, ax) else (ax, zoom):
            target.plot(t, truth[ep], color=BLACK, ls="--", lw=THIN, label="6D chip label")
            for res, color, ls, lw, lab in ((emr, BLUE, "-.", THIN, "EM"), (disc, RED, "-", MAIN, "Discriminative")):
                text = lab if mode == "zoom" else f"{lab} ({res[ck][ep]:.2f})"
                target.plot(t, res[key][ep], color=color, ls=ls, lw=lw, label=text)
        if zoom is not None:  # 확대 축 범위는 라벨과 discriminative 기준. 범위를 벗어난 EM 은 제목에 표기
            inside = (t >= bump[0]) & (t <= bump[1])
            ref, em_in = np.concatenate([truth[ep][inside], disc[key][ep][inside]]), emr[key][ep][inside]
            pad = .1 * (ref.max() - ref.min())
            zoom.set(xlim=bump, ylim=(ref.min() - pad, ref.max() + pad))
            clipped = em_in.min() < ref.min() - pad or em_in.max() > ref.max() + pad
            if zoom is not ax:
                zoom.tick_params(labelsize=9), zoom.grid(alpha=.25), ax.indicate_inset_zoom(zoom, edgecolor="0.35")
            elif clipped:
                title = f"{title}  (EM clipped, range {em_in.min():.0f} to {em_in.max():.0f} {unit})"
        low, high = ax.get_ylim()
        ax.set(title=title, xlabel="Time [s]", ylabel=ylab, ylim=(low, high + .32 * (high - low)))
        ax.grid(alpha=.25), ax.legend(ncol=3, loc="upper center")
    save(fig, name)

# ── 그림 2: EM 수렴 궤적 (우도는 위로 갈수록 좋게, 상관계수는 pitch·bounce 둘 다)
fig, ax = plt.subplots(figsize=(7.6, 4.4))
it = [r["iters"] for r in track]
ax.plot(it, [-r["energy"] for r in track], color=BLUE, ls="-.", lw=THIN, marker="o", ms=3.0)
ax.set(title="EM iteration: sensor likelihood rises while pitch rate degrades",
       xlabel="EM iteration", ylabel="Sensor log-likelihood per sample")
ax.grid(alpha=.25)
twin = ax.twinx()
twin.plot(it, [r["bcorr"] for r in track], color=BLACK, ls="--", lw=THIN)
twin.plot(it, [r["corr"] for r in track], color=RED, lw=MAIN, marker="s", ms=3.0)
twin.set(ylabel="Waveform correlation", ylim=(0.3, 1.0))
twin.tick_params(axis="y", direction="in")
ax.plot([], [], color=BLUE, ls="-.", lw=THIN, marker="o", ms=3.0, label="Sensor log-likelihood (left, higher = better)")
ax.plot([], [], color=RED, lw=MAIN, marker="s", ms=3.0, label="Pitch rate correlation (right)")
ax.plot([], [], color=BLACK, ls="--", lw=THIN, label="Bounce rate correlation (right)")
ax.legend(loc="center right")
save(fig, "result_em_convergence.png")

# ── 그림 3: episode 별 상관계수 분포 (예전 episode waveform correlation 판과 같은 boxplot)
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
axes[0].boxplot([emr["corr"], disc["corr"]], tick_labels=["EM", "Discriminative"], showfliers=False, widths=.5)
axes[0].set(title="Episode waveform correlation", ylabel="Pitch rate correlation")
axes[0].grid(axis="y", alpha=.25)
axes[1].scatter(emr["corr"], disc["corr"], s=14, color=BLUE, alpha=.8, linewidths=0)
low = min(emr["corr"].min(), disc["corr"].min()) - .03
axes[1].plot([low, 1], [low, 1], color=GREY, ls="--", lw=THIN)
axes[1].set(title=f"Discriminative better in {100 * np.mean(disc['corr'] > emr['corr']):.0f}% of episodes",
            xlabel="EM correlation", ylabel="Discriminative correlation")
axes[1].grid(alpha=.25)
save(fig, "result_corr_distribution.png")
