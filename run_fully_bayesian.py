from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
import time
import json
import joblib
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from loader import Dataset, View
from core import (Run, split_ctx, grid, Track,
                  pointwise_lpd, sum_se, auroc_trust_interval, trust_to_metric)
from core.run import seed_all
from models.fully_bayesian.model import Population
from models.fully_bayesian.features import build_feature_pipeline
from models.fully_bayesian import viz, utils


@dataclass
class Config:
    # train: tuple = ("김진명", "조현석", "박재일", "한규택", "이지환")
    # test: tuple = ("강신길",)
    train: tuple = ("강신길", "조현석", "한규택", "박재일", "이지환")
    test: tuple = ("김재호", "김진명", "김태근", "신민철", "이강근")
    view: View = View(
        features=("Pitch_rate_6D", "Bounce_rate_6D", "IMU_VerAccelVal", 
                  "IMU_LongAccelVal", "IMU_LatAccelVal"),
        around=(-2, 2), downsample=2, smooth=(10.0, 2))

    timestamp: str = None
    seed: int = 42
    verbose: int = 1

    # 6D 채널을 재구성 신호로 치환한 variant들을 추가로 학습·평가 (true와 쌍으로 비교).
    # recon_timestamp: None -> 재구성 학습까지 이 run에서 수행, recon에 지정된 모델만 피팅
    #                          (산출물은 <run>/reconstruction/에 저장)
    #                  "test" 포함 timestamp -> outputs/reconstruction/<timestamp>/ 아티팩트 로드
    recon: tuple = ("unet",)             # ()=없음 | "physics"/"kalman"/"fir"/"unet" 조합
    recon_timestamp: str = None

    # feature pipeline
    feature_method: str = "manual"     # "manual" | "manual_pca" | "ae"
    standardize: bool = True
    include_bias: bool = True
    manual_stats: dict = field(default_factory=lambda: {
        # "Pitch_rate_6D": ["abs_peak_deriv", "p2p_deriv", "rms_deriv", "vdv_deriv",
        #                   "abs_peak", "p2p", "rms", "wrms_rot", "vdv",
        #                     "mean", "std", "energy", "impulse_abs", "crest"],
        # "Bounce_rate_6D": ["abs_peak_deriv", "p2p_deriv", "rms_deriv", "wrms_z_deriv", "vdv_deriv",
        #                   "abs_peak", "p2p", "rms", "vdv",
        #                     "mean", "std", "energy", "impulse_abs", "crest"],
        # "IMU_VerAccelVal": ["abs_peak", "p2p", "rms", "wrms_z", "vdv", 
        #                     "mean", "std", "energy", "impulse_abs", "crest"],
        # "IMU_LongAccelVal": ["p2p", "abs_peak", "rms", "wrms_xy", "vdv", 
        #                      "mean", "std", "energy", "impulse_abs", "crest"],
        # "IMU_LatAccelVal": ["abs_peak", "p2p", "rms", "wrms_xy", "vdv",
        #                     "mean", "std", "energy", "impulse_abs", "crest"],

        # Primary
        # "Pitch_rate_6D": ["p2p"],
        # "Bounce_rate_6D": ["impulse_abs", "abs_peak_deriv", "p2p_deriv", "wrms_z_deriv"],
        # "IMU_VerAccelVal": ["p2p", "wrms_z", "crest", "vdv"],
        # "IMU_LongAccelVal": ["std", "wrms_xy", "impulse_abs", "crest"],

        # secondary
        "Bounce_rate_6D": ["impulse_abs", "abs_peak_deriv", "p2p_deriv", "wrms_z_deriv"],
        "IMU_VerAccelVal": ["p2p", "wrms_z", "crest", "vdv"],
        "IMU_LongAccelVal": ["std", "wrms_xy", "impulse_abs", "crest"],

    })
    manual_pca_dim: int = 20

    ae_latent_dim: int = 8
    ae_hidden_channels: int = 32
    ae_epochs: int = 500
    ae_lr: float = 0.001
    ae_batch_size: int = 128
    ae_device: str = "cuda"

    # Gibbs
    n_burnin: int = 500
    n_samples: int = 1500
    thin: int = 1
    niw_kappa0: float = 1.0
    niw_nu0: float = None
    niw_lambda0_scale: float = 1.0
    eps_var: float = None          # per-observation noise eps ~ N(0, eps_var); None disables

    # Exact-zero group spike-and-slab: beta_u = gamma * theta_u (gamma=0이면 완전 배제).
    spike_slab: bool = True
    spike_slab_unit: str = "feature"       # "feature"=(채널,통계) | "sensor"=채널 단위
    spike_slab_a: float = 1.0              # pi ~ Beta(a, b)
    spike_slab_b: float = 1.0

    newuser_n_iters: int = 8
    ctx_step: int = 10
    loo_k: int = 1                 # 0=exact LOO (한 개씩, K=n), 1=생략, K>=2=stratified K-fold

    # Feature selection (초기 feature 선택용). sensor=채널 단위, feature=(채널,통계) 하나씩.
    selection_units: tuple = ("sensor", "feature")

    projpred: bool = True                # projection-predictive selection (KL 재현율 기반)
    projpred_n_draws: int = 12
    projpred_target: float = 0.95        # bias→full KL 감소분 중 재현 목표 비율
    projpred_stop_at_target: bool = True
    projpred_max_iter: int = 50
    projpred_ridge: float = 1e-6

    conditional_mi: bool = True          # conditional-MI selection (실제 y, redundancy 제거)
    cmi_mode: str = "forward"            # "forward" | "leave_one_out"
    cmi_n_splits: int = 5
    cmi_C: float = 1.0
    cmi_solver_max_iter: int = 1000
    cmi_max_steps: int = None
    cmi_stop_at_nonpositive: bool = True
    cmi_user_interactions: bool = True
    cmi_min_gain: float = 0.0


def train(cfg, run, phi, pop, train_data):
    Xs = [train_data[n][0] for n in cfg.train]
    ys = [train_data[n][1] for n in cfg.train]
    tic = time.time()
    phi.fit(Xs, ys)
    Zs = [phi.transform(X).astype(np.float64) for X in Xs]
    print(f"Feature Pipeline Fit Done. ({time.time() - tic:.2f} sec)")
    tic = time.time()
    stats = pop.fit(Zs, ys, phi.feature_names, list(cfg.train), phi.groups)
    print(f"Population Fit Done. ({time.time() - tic:.2f} sec)")

    train_aurocs = {}
    for name, Z, y in zip(cfg.train, Zs, ys):
        if name not in pop.user_names or len(np.unique(y)) < 2:
            continue
        u = pop.user(name)
        train_aurocs[name] = float(roc_auc_score(y, u.predict(Z)[0]))

    plots = run.plots
    pca = viz.fit_theta_pca(pop)
    train_X = {n: train_data[n][0] for n in cfg.train}
    viz.plot_gibbs_trace(pop, plots / "gibbs_trace.png")
    viz.plot_population_weight_forest(pop, phi, plots / "population" / "weight_forest.png")
    viz.plot_population_covariance(pop, phi, plots / "population" / "covariance_corr.png")
    viz.plot_population_eigenspectrum(pop, plots / "population" / "eigenvalue_spectrum.png")
    viz.plot_user_similarity(pop, phi, train_X, plots / "population" / "posterior_similarity.png")
    viz.plot_train_theta_pca(pop, pca, plots / "population" / "theta_pca.png")
    viz.plot_train_user_predictions(pop, plots / "train", Zs, ys, list(cfg.train))
    viz.plot_train_predictive_panels(pop, plots / "train", Zs, ys, list(cfg.train))
    utils.write_train_user_profiles(pop, phi, run.dir, Zs, ys, list(cfg.train))
    utils.write_population_summary(pop, phi, run.dir)
    if pop.spike_slab:
        utils.write_spike_slab_summary(pop, run.dir)
        viz.plot_spike_slab_inclusion(pop, plots / "population" / "spike_slab_inclusion.png")

    tic = time.time()
    if cfg.projpred:
        from models.fully_bayesian import projpred
        print("Projection-Predictive Selection")
        for unit in cfg.selection_units:
            order = projpred.select(cfg, phi, pop, train_data, unit=unit)
            projpred.write_report(order, run.dir)
            projpred.plot(order, plots / "population" / f"projpred_{unit}.png")
        print(f"Finished Projection-Predictive Selection. ({time.time() - tic:.2f} sec)")

    tic = time.time()
    if cfg.conditional_mi:
        from models.fully_bayesian import conditional_mi
        print("Conditional-MI Selection")
        for unit in cfg.selection_units:
            result = conditional_mi.select(cfg, phi, pop, train_data, unit=unit)
            conditional_mi.write_report(result, run.dir)
            conditional_mi.plot(result, plots / "population" / f"conditional_mi_{unit}.png")
        print(f"Finished Conditional-MI Selection. ({time.time() - tic:.2f} sec)")

    stats["gibbs/train_auroc_mean"] = float(np.mean(list(train_aurocs.values()))) if train_aurocs else float("nan")
    if cfg.verbose:
        print(f"[train] loglik={stats['gibbs/loglik_final']:.1f}  auroc_mean={stats['gibbs/train_auroc_mean']:.4f}")
    return stats, pca


def eval_driver(cfg, phi, pop, tname, X_test, y_test):
    Z_test = phi.transform(X_test).astype(np.float64)
    (ctx_Z, ctx_y), (hold_Z, hold_y) = split_ctx(Z_test, y_test)
    split = len(ctx_y)

    star = pop.new_user(tname)
    prior_mean_all, _, prior_M_all = star.predict(Z_test)
    prior_mean, prior_M = prior_mean_all[split:], prior_M_all[:, split:]
    lpd_prior_all = pointwise_lpd(y_test, prior_mean_all)
    prior_ale = float((prior_M * (1.0 - prior_M)).mean())
    prior_epi = float(prior_M.var(axis=0, ddof=1).mean())

    trk = Track(hold_y)
    theta_traj = [star.theta.mean(axis=0)]
    traj_ts = [0]
    mean_epi, mean_ale = [], []
    final_M = prior_M
    peak_M = prior_M
    peak_theta_mean = star.theta.mean(axis=0)
    peak_theta_std = star.theta.std(axis=0)

    for t in grid(split, cfg.ctx_step):
        star.fit(ctx_Z[:t], ctx_y[:t])
        mean_p, _, M = star.predict(hold_Z)
        final_M = M
        mean_epi.append(float(M.var(axis=0, ddof=1).mean()))
        mean_ale.append(float((M * (1.0 - M)).mean()))
        theta_traj.append(star.theta.mean(axis=0))
        traj_ts.append(t)
        if trk.add(t, mean_p):
            peak_M = M
            peak_theta_mean = star.theta.mean(axis=0)
            peak_theta_std = star.theta.std(axis=0)

    peak_t, peak_auroc, peak_mean = trk.peak
    final_t, final_auroc, final_mean = trk.final

    # held-out 재적합 평가: loo_k=0 -> exact LOO (fold 크기 1이 N개), 1 -> 생략, K>=2 -> stratified K-fold
    loo = {}
    if cfg.loo_k != 1:
        star.fit(Z_test, y_test)                                     # fold 재적합의 warm-start 기반
        N = len(y_test)
        if cfg.loo_k:
            k_folds = int(min(cfg.loo_k, y_test.sum(), N - y_test.sum()))
            folds = list(StratifiedKFold(n_splits=k_folds, shuffle=True,
                                         random_state=cfg.seed).split(Z_test, y_test))
        else:
            k_folds = N
            idx = np.arange(N)
            folds = [(np.delete(idx, i), idx[i:i + 1]) for i in range(N)]
        loo_probs = np.empty(N)
        for tr, te in folds:
            s = star.clone()
            s.fit(Z_test[tr], y_test[tr])
            loo_probs[te] = s.predict(Z_test[te])[0]
        elpd_loo_i = pointwise_lpd(y_test, loo_probs)
        elpd_loo, se_loo = sum_se(elpd_loo_i)
        delta_elpd, se_delta = sum_se(elpd_loo_i - lpd_prior_all)    # 적응 vs prior, paired (eq 24)
        loo = dict(elpd_loo_i=elpd_loo_i, loo_probs=loo_probs,
                   elpd_loo=elpd_loo, se_elpd_loo=se_loo,
                   delta_elpd=delta_elpd, se_delta=se_delta, k_folds=k_folds)

    return dict(
        name=tname, holdout_y=hold_y, y_all=y_test, hold_X=hold_Z,
        final_mean=final_mean, prior_mean=prior_mean, peak_mean=peak_mean,
        peak_t=peak_t, peak_auroc=peak_auroc, final_t=final_t, final_auroc=final_auroc,
        prior_M=prior_M, final_M=final_M, peak_M=peak_M,
        lpd_prior=lpd_prior_all[split:],
        lpd_final=pointwise_lpd(hold_y, final_mean),
        lpd_peak=pointwise_lpd(hold_y, peak_mean),
        lpd_prior_all=lpd_prior_all, **loo,
        peak_theta_mean=peak_theta_mean, peak_theta_std=peak_theta_std,
        ctx_sizes=trk.ts, aurocs=trk.aurocs,
        theta_traj=theta_traj, traj_ts=traj_ts,
        t_grid=trk.ts, mean_epistemic=mean_epi, mean_aleatoric=mean_ale,
        prior_epi=prior_epi, prior_ale=prior_ale,
        trust_prior=auroc_trust_interval(hold_y, prior_M, cfg.seed),
        trust_final=auroc_trust_interval(hold_y, final_M, cfg.seed),
        trust_peak=auroc_trust_interval(hold_y, peak_M, cfg.seed),
    )


def evaluate(cfg, run, phi, pop, train_data, test_data, pca):
    plots = run.plots
    drivers = []

    for tname in cfg.test:
        X_test, y_test = test_data[tname]
        d = eval_driver(cfg, phi, pop, tname, X_test, y_test)
        drivers.append(d)

        viz.plot_predictive_panel(d["holdout_y"], d["peak_M"], plots / "test" / f"predictive_panel_{tname}_peak.png",
                                  title=f"{tname}  peak ctx t={d['peak_t']}  AUROC={d['peak_auroc']:.4f}")
        viz.plot_predictive_panel(d["holdout_y"], d["final_M"], plots / "test" / f"predictive_panel_{tname}_final.png",
                                  title=f"{tname}  final ctx t={d['final_t']}  AUROC={d['final_auroc']:.4f}")
        viz.plot_prior_posterior_contributions(phi, pop, d["hold_X"], pop.mu_bar, d["peak_theta_mean"],
                                               d["peak_theta_std"], plots / "test" / f"contributions_{tname}.png", tname)
        viz.plot_theta_trajectory(pop, pca, d["theta_traj"], d["traj_ts"], plots / "test" / f"theta_trajectory_{tname}.png", tname)
        viz.plot_uncertainty_decay(d["t_grid"], d["mean_epistemic"], d["mean_aleatoric"], d["prior_epi"], d["prior_ale"],
                                   plots / "test" / f"uncertainty_decay_{tname}.png", tname)
        viz.plot_weight_forest_with_test(pop, phi, d["peak_theta_mean"], d["peak_theta_std"],
                                         plots / "test" / f"weight_forest_with_{tname}.png", tname)

    names = [d["name"] for d in drivers]
    ys = [d["holdout_y"] for d in drivers]
    ms_final = viz.evaluate_predictions(ys, [d["final_mean"] for d in drivers], plots, names,
                                        save_name="metrics_online", title="Fully Bayesian (final t)")
    ms_prior = viz.evaluate_predictions(ys, [d["prior_mean"] for d in drivers], plots, names,
                                        save_name="metrics_prior", title="Fully Bayesian (prior only)")
    ms_peak = viz.evaluate_predictions(ys, [d["peak_mean"] for d in drivers], plots, names,
                                       save_name="metrics_peak", title="Fully Bayesian (oracle peak)")
    has_loo = "loo_probs" in drivers[0]
    if has_loo:
        ms_loo = viz.evaluate_predictions([d["y_all"] for d in drivers], [d["loo_probs"] for d in drivers], plots, names,
                                          save_name="metrics_loo", title="Fully Bayesian (held-out refit: LOO / K-fold)")
    viz.plot_sequential_auroc([d["ctx_sizes"] for d in drivers], [d["aurocs"] for d in drivers], plots, names)
    viz.plot_uncertainty_decay_overlay(drivers, plots / "test" / "uncertainty_decay_overlay.png")
    viz.plot_test_theta_pca(pop, pca, drivers, plots / "population" / "theta_pca_with_test.png")
    train_X = {n: train_data[n][0] for n in pop.user_names}
    viz.plot_user_similarity(pop, phi, train_X, plots / "population" / "posterior_similarity.png",
                             test_thetas=[(d["name"], d["peak_theta_mean"]) for d in drivers])
    viz.plot_trust_forest(drivers, plots / "test" / "trust_forest.png")
    viz.plot_lppd(drivers, plots / "test" / "lppd.png")

    for i, (d, m, mp, mk) in enumerate(zip(drivers, ms_final, ms_prior, ms_peak)):
        run.metrics[f"test/{d['name']}"] = {**m, "lppd": float(d["lpd_final"].mean()), **trust_to_metric(d["trust_final"])}
        run.metrics[f"test/{d['name']}_prior"] = {**mp, "lppd": float(d["lpd_prior"].mean()), **trust_to_metric(d["trust_prior"])}
        run.metrics[f"test/{d['name']}_peak"] = {**mk, "lppd": float(d["lpd_peak"].mean()), **trust_to_metric(d["trust_peak"])}
        if has_loo:
            run.metrics[f"test/{d['name']}_loo"] = {**ms_loo[i], "lppd": float(d["elpd_loo_i"].mean()),
                "elpd_loo": d["elpd_loo"], "se_elpd_loo": d["se_elpd_loo"],
                "delta_elpd_vs_prior": d["delta_elpd"], "se_delta": d["se_delta"],
                "k_folds": d["k_folds"], "n_all": len(d["y_all"])}

    utils.save_trust_metrics(drivers, run.dir / "metrics_trust.txt")
    if has_loo:
        with open(run.dir / "loo_pointwise.json", "w", encoding="utf-8") as f:
            json.dump({d["name"]: {"elpd_i": d["elpd_loo_i"].tolist(), "lpd_prior_i": d["lpd_prior_all"].tolist(),
                                   "y": d["y_all"].tolist()}
                       for d in drivers}, f, ensure_ascii=False)


def apply_recon(ds, art, method):
    signals = art[method]                             # (E, 3, T) native 단위
    row = {i: k for k, i in enumerate(art["ids"])}
    for ep in ds.episodes:
        r = signals[row[f"{ep.driver} {ep.id}"]]
        df = ep.signals.df                            # 메모리 DataFrame만 치환 (파일·캐시 불변)
        for c, ch in enumerate(art["channels"]):
            n = min(len(df), r.shape[1])
            df.loc[:n - 1, ch] = r[c, :n]
            df.loc[n:, ch] = r[c, n - 1]


def train_and_evaluate(cfg, sub, train_data, test_data):
    if sub.eval_only:
        obj = joblib.load(sub.dir / "model.joblib")
        phi, pop = obj["phi"], Population.from_state_dict(obj["pop"])
        pca = viz.fit_theta_pca(pop)
    else:
        phi = build_feature_pipeline(cfg)
        pop = Population(cfg)
        stats, pca = train(cfg, sub, phi, pop, train_data)
        joblib.dump({"phi": phi, "pop": pop.state_dict()}, sub.dir / "model.joblib")
        sub.metrics["train/gibbs"] = stats

    evaluate(cfg, sub, phi, pop, train_data, test_data, pca)

    for name in cfg.test:
        m = sub.metrics
        f, p, k = (m[f"test/{name}"]["auroc"], m[f"test/{name}_prior"]["auroc"], m[f"test/{name}_peak"]["auroc"])
        lf, lp, lk = (m[f"test/{name}"]["lppd"], m[f"test/{name}_prior"]["lppd"], m[f"test/{name}_peak"]["lppd"])
        print(f"[Done] {name}: auroc final={f:.4f} prior={p:.4f} peak={k:.4f}  |  lppd final={lf:.4f} prior={lp:.4f} peak={lk:.4f}")
        if f"test/{name}_loo" in m:
            lo = m[f"test/{name}_loo"]
            print(f"       loo(K={lo['k_folds']}, n={lo['n_all']}): auroc={lo['auroc']:.4f} brier={lo['brier']:.4f} lppd={lo['lppd']:.4f}"
                  f"  elpd={lo['elpd_loo']:+.1f}+/-{lo['se_elpd_loo']:.1f}"
                  f"  d_prior={lo['delta_elpd_vs_prior']:+.1f}+/-{lo['se_delta']:.1f}")


def main(cfg=None):
    cfg = cfg or Config()
    run = Run("fully_bayesian", cfg)
    print(f"[INFO] fully_bayesian  timestamp={cfg.timestamp} feature_method={cfg.feature_method} recon={cfg.recon}")
    ds = Dataset("datasets")

    variants = ["true"] + list(cfg.recon)
    if cfg.recon:
        if cfg.recon_timestamp is None:               # 재구성 학습을 이 run 안에서 수행 (recon 모델만 피팅)
            import run_reconstruction
            seed_all(cfg.seed)
            art_dir = run.dir / "reconstruction"
            sub = SimpleNamespace(dir=art_dir, plots=art_dir / "plots", metrics={}, eval_only=False)
            run_reconstruction.main(run_reconstruction.Config(test=cfg.test, seed=cfg.seed,
                                                              methods=cfg.recon, show=cfg.recon), sub)
            run.metrics.update({f"recon/{k}": v for k, v in sub.metrics.items()})
        else:
            art_dir = Path("outputs/reconstruction") / cfg.recon_timestamp
        art = np.load(art_dir / "reconstructed_signals.npz")

    for v in variants:                                # true 먼저 (치환이 in-place라 순서 고정)
        if v != "true":
            apply_recon(ds, art, v)
        print(f"===== variant: {v} =====")
        seed_all(cfg.seed)                            # variant 간 동일 RNG 상태에서 시작 (공정 비교)
        sub = SimpleNamespace(dir=run.dir / v, plots=run.dir / v / "plots",
                              metrics={}, eval_only=run.eval_only)
        sub.dir.mkdir(exist_ok=True)
        train_data = {n: cfg.view(ds[n]) for n in cfg.train}
        test_data = {n: cfg.view(ds[n]) for n in cfg.test}
        train_and_evaluate(cfg, sub, train_data, test_data)
        for key, val in sub.metrics.items():
            run.metrics[f"{v}/{key}"] = val

    if cfg.recon:
        for name in cfg.test:
            a = run.metrics[f"true/test/{name}"]["auroc"]
            row = "  ".join(f"{m}={run.metrics[f'{m}/test/{name}']['auroc']:.4f} ({run.metrics[f'{m}/test/{name}']['auroc'] - a:+.4f})"
                            for m in cfg.recon)
            print(f"[Recon] {name}: auroc true={a:.4f}  {row}")
    run.finish()


if __name__ == "__main__":
    main()
