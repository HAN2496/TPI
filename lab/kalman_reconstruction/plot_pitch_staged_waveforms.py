"""pitch_staged_metrics.csv 의 파라미터로 예측을 재계산해 그린다 (fit 없음, EM 은 재실행 ~10 s).

  모델별:   --models d_torque,f_fixlever --objectives sup,ml,joint:3,em:joint:3,em:sup
            pitch_staged_compare_<model>.png      라벨 vs 추정 파형 (worst/median/best) + episode corr boxplot
            pitch_staged_diagnostics_<model>.png  필터 일관성: innovation 자기상관, NIS 분포, innovation ±2σ 대역
  모델 비교: --compare --models f_fixlever,g_physical --objectives sup
            pitch_staged_compare_<m1>_vs_<m2>_<objective>.png  같은 목적함수로 모델들을 한 장에
            (파형 worst/median/best, corr 분포, 에피소드별 corr 산점, innovation 자기상관)
objective 표기: sup | ml | joint:<mu> | em:<source>  (em:<source> = 그 fit 의 플랜트 + Q,R 만 EM full)
"""
import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np

from .pitch_staged_reconstruction import DEG, SEALED, VARIANTS, observations
from .run import OUTPUT, data, training_split
from .run_em_noise_covariance import em
from .state_space import metrics
from .viz import SUFFIX_STYLES, plot_waveforms

plt.rcParams["axes.unicode_minus"] = False
COMPARE_COLORS = ("#d62728", "#ff7f0e", "#2ca02c", "#9467bd")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="d_torque,f_fixlever")
    parser.add_argument("--objectives", default="sup,ml,joint:3,em:joint:3,em:sup")
    parser.add_argument("--fit-episodes", type=int, default=200)
    parser.add_argument("--compare", action="store_true", help="모델들을 같은 목적함수로 한 장에 비교")
    parser.add_argument("--em-iters", type=int, default=2000, help="em:<source> 재실행 반복 수 (60회는 미수렴)")
    args = parser.parse_args()
    cfg, x, y, ids, test = data()
    drivers = np.array([value.split()[0] for value in ids])
    dev = test & ~np.isin(drivers, SEALED)
    train, _, _ = training_split(ids, test)
    fit_index = train[np.linspace(0, len(train) - 1, min(args.fit_episodes, len(train))).astype(int)]
    label, obs, fs = y[:, :, 2], observations(x), cfg.fs
    with (OUTPUT / "pitch_staged_metrics.csv").open(encoding="utf-8-sig") as stream:
        fits = {(row["model"], row["objective"]): row["parameters"] for row in csv.DictReader(stream)}
    style = lambda key: next((s for k, s in SUFFIX_STYLES.items() if k in key), {})
    models, objectives = args.models.split(","), args.objectives.split(",")
    results, runs = {}, {}
    for model in models:
        spec = VARIANTS[model]
        yv = np.stack([obs[channel] for channel in spec["channels"]], -1)
        x0 = np.zeros((len(yv), 9))
        x0[:, 0] = yv[:, 0, 0]
        for objective in objectives:
            source = objective.split(":", 1)[1] if objective.startswith("em:") else objective
            if (model, source) not in fits:
                continue
            d = dict(spec["fixed"]) | {item.split("=")[0]: float(item.split("=")[1]) for item in fits[model, source].split()}
            d = {(key[4:] if key.startswith("log_") else key): value for key, value in d.items()}
            ss = spec["build"](d, fs)
            u = obs["torque"] if ss.B is not None else None
            if objective.startswith("em:"):
                ss = em(ss, yv[fit_index], None if u is None else u[fit_index], x0[fit_index], args.em_iters, 1e-6, False)[0]
            state, nu, cov = ss.filter(yv, u, x0, innovations=True)
            q = state[..., 3]
            pred = DEG * q + label[train].mean() - DEG * q[train].mean()
            key = f"{model}_{objective}"
            results[key] = dict(pred=pred[dev], corr=metrics(label[dev], pred[dev], fs)[0])
            runs[key] = (nu[dev][:, 100:], cov[100:])
    acf = lambda nu, cov, lags: [np.mean(np.sum(w[:, k:] * w[:, :w.shape[1] - k], 1) / np.sum(w * w, 1))
                                for w in [(lambda z: z - z.mean(1, keepdims=True))(nu / np.sqrt(np.einsum("tii->ti", cov)))]
                                for k in lags]
    nis = lambda nu, cov: np.einsum("nti,tij,ntj->nt", nu, np.linalg.inv(cov), nu).mean(1) / cov.shape[-1]

    if args.compare:
        keys, lags = list(results), np.arange(51)
        picks = np.argsort(results[keys[-1]]["corr"])
        picks = (picks[0], picks[len(picks) // 2], picks[-1])
        t = np.arange(label.shape[1]) / fs
        fig, axes = plt.subplots(3, 2, figsize=(14, 13))
        for ax, name, index in zip(axes.flat[:3], ("worst", "median", "best"), picks):
            ax.plot(t, label[dev][index], color="black", lw=1.1, label="recorded")
            for color, key in zip(COMPARE_COLORS, keys):
                ax.plot(t, results[key]["pred"][index], color=color, lw=1.0, label=key)
            ax.set(title=f"{name} (by {keys[-1]}): {ids[dev][index]}", xlabel="time [s]"), ax.legend(fontsize=9)
        axes[1, 1].boxplot([results[k]["corr"] for k in keys], tick_labels=keys, showfliers=False)
        axes[1, 1].set(title="episode waveform correlation")
        base = results[keys[0]]["corr"]
        for color, key in zip(COMPARE_COLORS[1:], keys[1:]):
            other = results[key]["corr"]
            axes[2, 0].scatter(base, other, s=14, color=color, alpha=.8,
                               label=f"{key} better in {100 * np.mean(other > base):.0f}% of episodes")
        lo, hi = min(map(np.min, (results[k]["corr"] for k in keys))), 1.0
        axes[2, 0].plot([lo, hi], [lo, hi], "k--", lw=.8)
        axes[2, 0].set(xlabel=f"{keys[0]} corr", ylabel="corr", title="per-episode correlation (above diagonal = better)")
        axes[2, 0].legend(fontsize=9)
        for color, key in zip(COMPARE_COLORS, keys):
            axes[2, 1].plot(lags, acf(*runs[key], lags), color=color, lw=1.2,
                            label=f"{key}  (NIS median {np.median(nis(*runs[key])):.2f})")
        axes[2, 1].axhline(0, color="k", lw=.8)
        axes[2, 1].set(title="normalized innovation autocorrelation (white = 0)", xlabel="lag [samples]"), axes[2, 1].legend(fontsize=9)
        for ax in axes.flat:
            ax.grid(alpha=.25)
        fig.suptitle(f"{' vs '.join(models)} — objective {args.objectives} (dev-test)")
        path = OUTPUT / f"pitch_staged_compare_{'_vs_'.join(models)}_{args.objectives.replace(':', '')}.png"
        fig.tight_layout(), fig.savefig(path, dpi=150), plt.close(fig)
        print(path.name, {key: round(float(np.median(r["corr"])), 3) for key, r in results.items()}, flush=True)
        return

    for model in models:
        keys = [key for key in results if key.startswith(model + "_")]
        sub = {key: results[key] for key in keys}
        reference = f"{model}_sup"
        median = plot_waveforms(label[dev], sub, ids[dev], fs, OUTPUT / f"pitch_staged_compare_{model}.png", reference)
        print(model, {key: round(float(np.median(r["corr"])), 3) for key, r in sub.items()}, flush=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        lags, values = np.arange(51), {}
        for key in keys:
            axes[0, 0].plot(lags, acf(*runs[key], lags), label=key, **style(key))
            values[key] = nis(*runs[key])
        axes[0, 0].axhline(0, color="k", lw=.8), axes[0, 0].set(title="normalized innovation autocorrelation (white = 0)",
                                                                 xlabel="lag [samples]"), axes[0, 0].legend(fontsize=9)
        axes[0, 1].boxplot(list(values.values()), tick_labels=list(values), showfliers=False)
        axes[0, 1].axhline(1, color="k", lw=.8, ls="--"), axes[0, 1].set_yscale("log")
        axes[0, 1].set(title="NIS per episode (consistent filter = 1)"), axes[0, 1].tick_params(axis="x", rotation=20)
        t = np.arange(100, 1000) / fs
        for ax, key in zip(axes[1], (reference, next((k for k in keys if "_em:" in k), keys[-1]))):
            nu, cov = runs[key]
            sigma = np.sqrt(cov[:, 1, 1])
            ax.fill_between(t, -2 * sigma, 2 * sigma, color="#cccccc", label="filter's own ±2σ (S)")
            ax.plot(t, nu[median, :, 1], lw=.8, label="innovation, a_x channel", **{k: v for k, v in style(key).items() if k == "color"})
            inside = np.mean(np.abs(nu[:, :, 1]) < 2 * sigma)
            ax.set(title=f"{key}\na_x innovation vs filter's ±2σ — inside {100 * inside:.0f}% (expect 95%)", xlabel="time [s]")
            ax.legend(fontsize=9)
        for ax in axes.flat:
            ax.grid(alpha=.25)
        fig.suptitle(f"{model}: filter consistency diagnostics (dev-test, first 1 s excluded)")
        fig.tight_layout(), fig.savefig(OUTPUT / f"pitch_staged_diagnostics_{model}.png", dpi=150), plt.close(fig)


if __name__ == "__main__":
    main()
