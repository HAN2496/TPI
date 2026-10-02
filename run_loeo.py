"""Leave-one-evaluator-out experiments for the T-IV draft (paper Section VI).

Stages
  main     LOEO over all eligible evaluators: proposed model (+ HB without inclusion prior),
           baselines, budgets x offsets x seeds, uncertainty metrics, prequential score,
           reference-model feature roles (PIP, common / specific / inactive).
  select   projection-predictive path per fold -> refit each size -> held-out MLPD -> one-SE rule.
  sensors  exhaustive channel subsets per fold -> refit -> held-out MLPD.
  report   aggregate everything under <run>/report/.
  all      main, select, sensors, report in sequence.

Usage (desktop, real data):
  python run_loeo.py --stage all
  python run_loeo.py --stage main --set budgets="(0,5,10,20)" --set n_offsets=3
  python run_loeo.py --stage report --timestamp 20261001_120000

Smoke test (synthetic, minutes):
  python run_loeo.py --data synthetic --fast --stage all

Every fold writes <run>/folds/<evaluator>*.json as soon as it finishes, so an interrupted
run can be resumed with --timestamp <folder>; finished folds are skipped.
"""
from __future__ import annotations

import argparse
import ast
import json
import time
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import norm

from core.run import Run, seed_all
from reward.fully_bayesian.loeo import bank as B
from reward.fully_bayesian.loeo import data as D
from reward.fully_bayesian.loeo import protocol as P
from reward.fully_bayesian.loeo import baselines as BL
from reward.fully_bayesian.loeo import selection as S
from reward.fully_bayesian.loeo import report as R
from reward.fully_bayesian.loeo import metrics as M


@dataclass
class Config:
    # ---- data
    data: str = "real"                         # "real" | "synthetic"
    dataset_root: str = "datasets"
    evaluators: tuple = ()                     # () = every driver in the dataset
    folds: tuple = ()                          # () = hold out every eligible evaluator; else subset (debug)
    channels: tuple = ("Pitch_rate_6D", "Bounce_rate_6D", "IMU_VerAccelVal",
                       "IMU_LongAccelVal", "IMU_LatAccelVal")
    around: tuple = (-2.0, 2.0)
    downsample: int = 2
    smooth: tuple = (10.0, 2)
    min_labels: int = 10                       # eligibility for the held-out role
    min_per_class: int = 3
    # ---- bank
    rho_max: float = 0.95
    feature_subset: tuple = ()                 # () = full pruned bank; else only these 'channel__stat' names (main stage)
    # ---- protocol
    budgets: tuple = (0, 5, 10, 20)
    include_final: bool = True
    ctx_frac: float = 0.5
    n_offsets: int = 3
    seeds: tuple = (42,)
    # ---- model
    n_burnin: int = 500
    n_samples: int = 1500
    newuser_n_iters: int = 8
    niw_kappa0: float = 1.0
    niw_nu0: float = None
    niw_lambda0_scale: float = 1.0
    spike_slab: bool = True
    spike_slab_unit: str = "feature"
    spike_slab_a: float = 1.0
    spike_slab_b: float = 1.0
    hb_full: bool = True                       # also fit the hierarchy without the inclusion prior
    sensor_pip: bool = True                    # also fit with sensor-level inclusion groups
    pip_threshold: float = 0.5
    q_min: float = 0.9
    reliable_w_max: float = 0.15               # width criterion of the AUROC reliability interval
    # ---- baselines
    baseline_C: float = 1.0
    ebmap_M: int = 400
    gbt: bool = False
    prequential: bool = True
    preq_max: int = 120
    # ---- selection
    sel_n_burnin: int = 200
    sel_n_samples: int = 400
    sel_spike_slab: bool = False               # submodels are refit without the inclusion prior
    sel_sizes: tuple = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40)
    sel_budgets: tuple = (0, 10)               # budgets that score a candidate subset
    projpred_n_draws: int = 12
    projpred_target: float = 0.95
    projpred_stop_at_target: bool = False      # run the whole path; sizes are scored by held-out MLPD
    projpred_max_iter: int = 50
    projpred_ridge: float = 1e-6
    sensor_min_size: int = 1
    # ---- compare (stage "compare": side-by-side tables of finished runs)
    compare_runs: tuple = ()                   # run folders, e.g. ("outputs/loeo/2026...", "outputs/loeo_lam01/2026...")
    # ---- synthetic
    syn_n_evaluators: int = 8
    syn_min_episodes: int = 30
    syn_max_episodes: int = 160
    syn_k_common: int = 6
    syn_k_individual: int = 3
    # ---- run
    stage: str = "main"
    timestamp: str = None
    run_name: str = "loeo"
    seed: int = 42
    verbose: int = 1
    fast: bool = False


FAST = dict(n_burnin=30, n_samples=60, sel_n_burnin=20, sel_n_samples=40, ebmap_M=60,
            n_offsets=2, budgets=(0, 5, 10), sel_sizes=(2, 5, 10), sel_budgets=(0, 5),
            preq_max=12, projpred_n_draws=4, syn_n_evaluators=5, syn_max_episodes=80)


# ----------------------------------------------------------------------------- helpers
def log(cfg, *a):
    if cfg.verbose:
        print(*a, flush=True)


def load_data(cfg):
    if cfg.data == "synthetic":
        data, channels, fs, truth = D.load_synthetic(cfg)
    else:
        data, channels, fs = D.load_real(cfg)
        truth = None
    return data, channels, fs, truth


def fold_names(cfg, data):
    elig = D.eligible(data, cfg.min_labels, cfg.min_per_class)
    names = list(elig) if not cfg.folds else [n for n in cfg.folds if n in elig]
    return names, elig


def feature_roles(pop, cfg):
    """PIP, sign-agreement q_j, effect score S_j and role per feature of a fitted reference model."""
    d = pop.d
    pip = pop.gamma_pip
    mu = pop.slab_mu_samples
    Sd = pop.slab_Sigma_samples[:, np.arange(d), np.arange(d)]
    q = norm.cdf(np.abs(mu) / np.sqrt(np.maximum(Sd, 1e-12))).mean(0)
    mu_eff = pop.mu_samples
    Sd_eff = pop.Sigma_samples[:, np.arange(d), np.arange(d)]
    S = (mu_eff ** 2 + Sd_eff).mean(0)
    rows = []
    for j, name in enumerate(pop.feature_names):
        if name == "bias":
            continue
        role = "inactive" if pip[j] < cfg.pip_threshold else ("common" if q[j] >= cfg.q_min else "specific")
        lo, hi = np.percentile(mu_eff[:, j], [2.5, 97.5])
        rows.append(dict(feature=name, group=pop.feature_groups[j], pip=float(pip[j]),
                         mu_mean=float(mu_eff[:, j].mean()), mu_lo=float(lo), mu_hi=float(hi),
                         between_sd=float(np.sqrt(Sd_eff[:, j]).mean()), q=float(q[j]),
                         S=float(S[j]), role=role))
    rows.sort(key=lambda r: -r["S"])
    return rows


def sensor_scores(pop):
    """Sensor-level PIP (if fitted with sensor groups) or group-mean feature PIP."""
    out = {}
    if getattr(pop, "gamma_unit_names", None) and pop.spike_slab_unit == "sensor":
        for name, pip in zip(pop.gamma_unit_names, pop.gamma_unit_pip):
            out[name] = float(pip)
    else:
        for g in set(pop.feature_groups):
            if g == "bias":
                continue
            idx = [j for j, gg in enumerate(pop.feature_groups) if gg == g]
            out[g] = float(np.mean(pop.gamma_pip[idx]))
    return out


# ----------------------------------------------------------------------------- stage: main
def run_fold_main(cfg, run, name, data, channels, fs):
    out_json = run.dir / "folds" / f"{name}.json"
    if out_json.exists():
        log(cfg, f"[main] {name}: exists, skipping")
        return json.loads(out_json.read_text(encoding="utf-8"))
    tic = time.time()
    pop_names = [n for n in data if n != name]
    pop_data = {n: data[n] for n in pop_names}
    X_held, y_held = data[name]

    bank = B.full_bank(channels)
    pruned, prune_report = B.prune_bank([pop_data[n][0] for n in pop_names], channels, fs, bank, cfg.rho_max)
    phi = B.make_pipeline(pruned, channels, fs).fit([pop_data[n][0] for n in pop_names], None)
    if cfg.feature_subset:                                      # e.g. the k=1 model of the selection stage
        phi = B.ColumnSubset(phi, list(cfg.feature_subset))
    Z_held = phi.transform(X_held).astype(np.float64)
    y_held = np.asarray(y_held)
    log(cfg, f"[main] {name}: d={len(phi.feature_names)} (pruned {len(prune_report)}), "
             f"pop={len(pop_names)} evaluators, held n={len(y_held)}")

    fold = dict(name=name, n=int(len(y_held)), n_pos=int(y_held.sum()), d=len(phi.feature_names),
                feature_names=phi.feature_names, pruned_bank=pruned, prune_report=prune_report,
                budgets_cfg=list(cfg.budgets),
                proposed={}, hb_full={}, lpd={}, roles=None, sensor_pip=None, prequential=None,
                fit_stats={}, per_seed={})
    budgets = None
    # proposed (spike-and-slab) and HB-full, per seed
    for seed in cfg.seeds:
        pop, Zs, ys, stats = P.fit_population(cfg, phi, pop_data, pop_names, seed=seed)
        fold["fit_stats"][f"proposed/{seed}"] = stats
        res, lpd, info = P.evaluate_proposed(cfg, pop, Z_held, y_held, seed)
        budgets = info["budgets"]
        fold["per_seed"][f"proposed/{seed}"] = res
        fold["lpd"].setdefault("proposed", {})
        for t, v in lpd.items():
            fold["lpd"]["proposed"].setdefault(str(t), []).append(v.tolist())
        if seed == cfg.seeds[0]:
            fold["roles"] = feature_roles(pop, cfg)
            fold["info"] = info
            joblib.dump({"phi": phi, "pop": pop.state_dict()}, run.dir / "folds" / f"{name}_model.joblib")
            if cfg.prequential:
                sc = P.prequential(cfg, pop, Z_held, y_held, seed, max_steps=cfg.preq_max)
                fold["prequential"] = dict(scores=sc.tolist(), cumulative=np.cumsum(sc).tolist())
        if cfg.hb_full:
            pop_f, _, _, stats_f = P.fit_population(cfg, phi, pop_data, pop_names, spike_slab=False, seed=seed)
            fold["fit_stats"][f"hb_full/{seed}"] = stats_f
            res_f, lpd_f, _ = P.evaluate_proposed(cfg, pop_f, Z_held, y_held, seed, budgets=budgets)
            fold["per_seed"][f"hb_full/{seed}"] = res_f
            fold["lpd"].setdefault("hb_full", {})
            for t, v in lpd_f.items():
                fold["lpd"]["hb_full"].setdefault(str(t), []).append(v.tolist())
    # average over seeds
    for method in ("proposed", "hb_full"):
        keys = [k for k in fold["per_seed"] if k.startswith(method + "/")]
        if not keys:
            continue
        agg = {}
        for t in budgets:
            rows = [fold["per_seed"][k][t] for k in keys if t in fold["per_seed"][k]]
            if rows:
                agg[str(t)] = {m: float(np.nanmean([r[m] for r in rows])) for m in rows[0]}
        fold[method] = agg
        fold["lpd"][method] = {t: np.mean(np.asarray(v), axis=0).tolist() for t, v in fold["lpd"][method].items()}
    # sensor-level inclusion
    if cfg.sensor_pip and cfg.spike_slab:
        cfg_s = replace(cfg, spike_slab_unit="sensor")
        pop_s, _, _, stats_s = P.fit_population(cfg_s, phi, pop_data, pop_names, seed=cfg.seeds[0])
        fold["fit_stats"]["sensor_pip"] = stats_s
        fold["sensor_pip"] = sensor_scores(pop_s)
    # baselines
    Zs = [phi.transform(pop_data[n][0]).astype(np.float64) for n in pop_names]
    ys = [np.asarray(pop_data[n][1]) for n in pop_names]
    for bl in BL.make_baselines(cfg):
        tic_b = time.time()
        bl.fit_population(Zs, ys)
        res_b, lpd_b = P.evaluate_baseline(cfg, bl, Z_held, y_held, budgets, particles=(bl.name == "ebmap"))
        fold[bl.name] = {str(t): v for t, v in res_b.items()}
        fold["lpd"][bl.name] = {str(t): v.tolist() for t, v in lpd_b.items()}
        fold["fit_stats"][bl.name] = dict(seconds=time.time() - tic_b)
    fold["seconds"] = time.time() - tic
    R.write_json(fold, out_json)
    log(cfg, f"[main] {name}: done in {fold['seconds']:.0f}s  "
             + "  ".join(f"t={t}: {fold['proposed'][str(t)]['mlpd']:.3f}" for t in budgets if str(t) in fold["proposed"]))
    return fold


def stage_main(cfg, run, data, channels, fs, names):
    folds = {}
    for name in names:
        folds[name] = run_fold_main(cfg, run, name, data, channels, fs)
    return folds


# ----------------------------------------------------------------------------- stage: select
def run_fold_select(cfg, run, name, data, channels, fs):
    out_json = run.dir / "folds" / f"{name}_select.json"
    if out_json.exists():
        log(cfg, f"[select] {name}: exists, skipping")
        return json.loads(out_json.read_text(encoding="utf-8"))
    tic = time.time()
    pop_names = [n for n in data if n != name]
    pop_data = {n: data[n] for n in pop_names}
    held = data[name]
    model_path = run.dir / "folds" / f"{name}_model.joblib"
    if model_path.exists():
        obj = joblib.load(model_path)
        from reward.fully_bayesian.model import Population
        phi, pop = obj["phi"], Population.from_state_dict(obj["pop"])
        pruned = {}                                     # {channel: [stats]} in pipeline order
        for ch, st in phi.pairs:
            pruned.setdefault(ch, []).append(st)
    else:
        bank = B.full_bank(channels)
        pruned, _ = B.prune_bank([pop_data[n][0] for n in pop_names], channels, fs, bank, cfg.rho_max)
        phi = B.make_pipeline(pruned, channels, fs).fit([pop_data[n][0] for n in pop_names], None)
        pop, _, _, _ = P.fit_population(cfg, phi, pop_data, pop_names, seed=cfg.seeds[0])
    d_feat = len(phi.feature_names) - 1
    budgets = list(cfg.sel_budgets)
    Z_pop = [phi.transform(pop_data[n][0]).astype(np.float64) for n in pop_names]   # transform once per fold
    ys = [np.asarray(pop_data[n][1]) for n in pop_names]
    Z_held = phi.transform(held[0]).astype(np.float64)
    # projection-predictive path on the reference model
    tic_pp = time.time()
    order = S.projpred_path(cfg, phi, pop, pop_data, unit="feature")
    secs_pp = time.time() - tic_pp
    log(cfg, f"[select] {name}: d={d_feat} reference {'loaded' if model_path.exists() else 'refit'}, "
             f"projpred path {len(order) - 1} steps in {secs_pp:.1f}s")
    ks = S.size_grid(d_feat, cfg.sel_sizes)
    subsets = S.path_subsets(order, ks)
    subsets[d_feat] = [f for f in phi.feature_names if f != "bias"]
    pip_set = [f for f, g in zip(phi.feature_names, pop.gamma_pip) if f != "bias" and g >= cfg.pip_threshold]
    candidates = {f"k={k}": subsets[k] for k in ks if k in subsets}
    if pip_set:
        candidates["pip"] = pip_set
    results = {}
    for label, feats in candidates.items():
        res, lpd, secs, fn = S.refit_and_score(cfg, phi, Z_pop, ys, pop_names, Z_held, held[1], feats, budgets, cfg.seeds[0])
        results[label] = dict(features=feats, n_features=len(feats), seconds=secs["total"], timing=secs,
                              metrics={str(t): v for t, v in res.items()},
                              lpd={str(t): v.tolist() for t, v in lpd.items()})
        log(cfg, f"[select] {name}: {label:>6} ({len(feats):2d} feats) "
                 + "  ".join(f"t={t}: {res[t]['mlpd']:.3f}" for t in res)
                 + f"  [{secs['total']:.1f}s: gibbs {secs['gibbs']:.1f} eval {secs['eval']:.1f}]")
    fold = dict(name=name, d_features=d_feat, sizes=ks, path=[dict(added=r["added"], kl=r["projection_kl"],
                captured=r["captured"], n=r["n_features"] - 1) for r in order],
                pip_set=pip_set, candidates=results, seconds_projpred=secs_pp, seconds=time.time() - tic)
    R.write_json(fold, out_json)
    return fold


def stage_select(cfg, run, data, channels, fs, names):
    return {n: run_fold_select(cfg, run, n, data, channels, fs) for n in names}


# ----------------------------------------------------------------------------- stage: sensors
def run_fold_sensors(cfg, run, name, data, channels, fs):
    out_json = run.dir / "folds" / f"{name}_sensors.json"
    if out_json.exists():
        log(cfg, f"[sensors] {name}: exists, skipping")
        return json.loads(out_json.read_text(encoding="utf-8"))
    tic = time.time()
    pop_names = [n for n in data if n != name]
    pop_data = {n: data[n] for n in pop_names}
    held = data[name]
    model_path = run.dir / "folds" / f"{name}_model.joblib"
    if model_path.exists():                                     # reuse the fold's full pruned pipeline
        phi = joblib.load(model_path)["phi"]
    else:
        bank = B.full_bank(channels)
        pruned, _ = B.prune_bank([pop_data[n][0] for n in pop_names], channels, fs, bank, cfg.rho_max)
        phi = B.make_pipeline(pruned, channels, fs).fit([pop_data[n][0] for n in pop_names], None)
    budgets = list(cfg.sel_budgets)
    Z_pop = [phi.transform(pop_data[n][0]).astype(np.float64) for n in pop_names]
    ys = [np.asarray(pop_data[n][1]) for n in pop_names]
    Z_held = phi.transform(held[0]).astype(np.float64)
    results = {}
    for subset in S.sensor_subsets(channels, cfg.sensor_min_size):
        label = "+".join(subset)
        feats = S.features_of_channels(phi, subset)
        res, lpd, secs, fn = S.refit_and_score(cfg, phi, Z_pop, ys, pop_names, Z_held, held[1], feats, budgets, cfg.seeds[0])
        results[label] = dict(channels=subset, n_features=len(fn) - 1, seconds=secs["total"], timing=secs,
                              metrics={str(t): v for t, v in res.items()},
                              lpd={str(t): v.tolist() for t, v in lpd.items()})
        log(cfg, f"[sensors] {name}: {label:<70} " + "  ".join(f"t={t}: {res[t]['mlpd']:.3f}" for t in res)
                 + f"  [{secs['total']:.1f}s: gibbs {secs['gibbs']:.1f} eval {secs['eval']:.1f}]")
    fold = dict(name=name, subsets=results, seconds=time.time() - tic)
    R.write_json(fold, out_json)
    return fold


def stage_sensors(cfg, run, data, channels, fs, names):
    return {n: run_fold_sensors(cfg, run, n, data, channels, fs) for n in names}


# ----------------------------------------------------------------------------- stage: report
def _load_folds(run, suffix=""):
    out = {}
    for p in sorted((run.dir / "folds").glob(f"*{suffix}.json")):
        stem = p.stem[: -len(suffix)] if suffix else p.stem
        if not suffix and (stem.endswith("_select") or stem.endswith("_sensors")):
            continue
        out[stem] = json.loads(p.read_text(encoding="utf-8"))
    return out


def stage_report(cfg, run, truth=None):
    rep = run.dir / "report"
    rep.mkdir(exist_ok=True)
    summary = {}
    # ---- main
    folds = _load_folds(run)
    if folds:
        R.normalize_final(folds, cfg.budgets)      # per-fold t = n_ctx -> shared pseudo-budget "final"
        budgets = R.budget_keys(folds, "proposed")
        methods = [m for m in R.METHODS if any(m in f and f[m] for f in folds.values())]
        rows = R.macro_table(folds, methods, budgets)
        R.write_csv(rows, rep / "main_macro.csv")
        md = ["# LOEO main results", f"folds: {len(folds)} evaluators: {', '.join(folds)}",
              R.markdown_main(rows, budgets, methods),
              "\n## Uncertainty (proposed)\n",
              R.markdown_main(rows, budgets, ["proposed", "hb_full", "ebmap"],
                              keys=("epi_mean", "ale_mean", "epi_ale_spearman", "correctness_auroc_epi", "aurc_epi", "eaurc_epi"))]
        # paired delta vs cold start
        md.append("\n## Paired ELPD change from cold start (proposed)\n")
        ts_pos = [t for t in budgets if t != 0]
        md.append("| evaluator | " + " | ".join(f"t={t}" for t in ts_pos) + " |")
        md.append("|---|" + "---|" * len(ts_pos))
        deltas = {t: R.paired_delta(folds, "proposed", t) for t in ts_pos}
        for name in folds:
            cells = []
            for t in ts_pos:
                dl = deltas[t].get(name)
                cells.append("--" if dl is None else f"{dl['delta_elpd']:+.2f} ± {dl['se']:.2f}")
            md.append(f"| {name} | " + " | ".join(cells) + " |")
        # per-evaluator MLPD change (ELPD sums scale with holdout size; MLPD does not)
        md.append("\n## Per-evaluator MLPD change from cold start (proposed; n_hold in parentheses)\n")
        md.append("| evaluator | t=0 MLPD | " + " | ".join(f"Δ t={t}" for t in ts_pos) + " |")
        md.append("|---|---|" + "---|" * len(ts_pos))
        for name, f in folds.items():
            pr = f["proposed"]
            base0 = pr.get("0", {}).get("mlpd")
            cells = ["--" if base0 is None else f"{base0:.3f} ({f['info']['n_hold']})"]
            for t in ts_pos:
                v = pr.get(str(t), {}).get("mlpd")
                cells.append("--" if v is None or base0 is None else f"{v - base0:+.3f}")
            md.append(f"| {name} | " + " | ".join(cells) + " |")
        # feature roles from the reference models
        role_rows = []
        for name, f in folds.items():
            for r in f.get("roles") or []:
                role_rows.append(dict(fold=name, **r))
        R.write_csv(role_rows, rep / "feature_roles_by_fold.csv")
        if role_rows:
            feats = sorted({r["feature"] for r in role_rows})
            agg = []
            for ft in feats:
                rr = [r for r in role_rows if r["feature"] == ft]
                agg.append(dict(feature=ft, group=rr[0]["group"],
                                pip=np.mean([r["pip"] for r in rr]), q=np.mean([r["q"] for r in rr]),
                                S=np.mean([r["S"] for r in rr]), mu_mean=np.mean([r["mu_mean"] for r in rr]),
                                between_sd=np.mean([r["between_sd"] for r in rr]),
                                role_common=np.mean([r["role"] == "common" for r in rr]),
                                role_specific=np.mean([r["role"] == "specific" for r in rr]),
                                role_inactive=np.mean([r["role"] == "inactive" for r in rr])))
            agg.sort(key=lambda r: -r["S"])
            R.write_csv(agg, rep / "feature_roles_macro.csv")
            md.append("\n## Feature roles (mean over folds; fraction of folds per role)\n")
            md.append("| feature | PIP | q | S | mu | between sd | common | specific | inactive |")
            md.append("|---|---|---|---|---|---|---|---|---|")
            for r in agg[:40]:
                md.append(f"| {r['feature']} | {r['pip']:.2f} | {r['q']:.2f} | {r['S']:.2f} | {r['mu_mean']:+.2f} | "
                          f"{r['between_sd']:.2f} | {r['role_common']:.2f} | {r['role_specific']:.2f} | {r['role_inactive']:.2f} |")
            if truth:
                md.append(f"\nsynthetic truth: common={truth['common']}, individual={truth['individual']}")
        sp = {}
        for name, f in folds.items():
            for ch, v in (f.get("sensor_pip") or {}).items():
                sp.setdefault(ch, []).append(v)
        if sp:
            md.append("\n## Sensor-level PIP (mean over folds)\n")
            for ch, v in sorted(sp.items(), key=lambda kv: -np.mean(kv[1])):
                md.append(f"- {ch}: {np.mean(v):.3f} (min {np.min(v):.2f}, max {np.max(v):.2f})")
        # prequential
        pq = {n: f["prequential"]["cumulative"] for n, f in folds.items() if f.get("prequential")}
        if pq:
            R.write_json(pq, rep / "prequential_cumulative.json")
        (rep / "main.md").write_text("\n".join(md), encoding="utf-8")
        R.plot_curves(rows, methods, budgets, rep / "curve_mlpd.png", "mlpd", "held-out MLPD (macro)")
        R.plot_curves(rows, methods, budgets, rep / "curve_auroc.png", "auroc", "held-out AUROC (macro)")
        R.plot_uncertainty(folds, budgets, rep / "uncertainty_decay.png")
        summary["main"] = dict(n_folds=len(folds), budgets=budgets, methods=methods)
    # ---- select
    sel = _load_folds(run, "_select")
    if sel:
        # the full size differs per fold (pruning is fold-specific): relabel each fold's full candidate
        for f in sel.values():
            fl = f"k={f['d_features']}"
            if fl in f["candidates"]:
                f["candidates"]["full"] = f["candidates"].pop(fl)
        labels = sorted({l for f in sel.values() for l in f["candidates"]},
                        key=lambda l: (0, int(l[2:])) if l.startswith("k=") else (1, l))
        budgets = sorted({int(t) for f in sel.values() for c in f["candidates"].values() for t in c["metrics"]})
        full_label = "full"
        mean, se_diff, table = {}, {}, []
        for l in labels:
            vals, diffs = [], []
            for f in sel.values():
                c = f["candidates"].get(l); full = f["candidates"].get(full_label)
                if c and full:
                    v = np.mean([c["metrics"][str(t)]["mlpd"] for t in budgets if str(t) in c["metrics"]])
                    vf = np.mean([full["metrics"][str(t)]["mlpd"] for t in budgets if str(t) in full["metrics"]])
                    vals.append(v); diffs.append(v - vf)
            m, se, n = M.macro(vals)
            _, sed, _ = M.macro(diffs)
            mean[l], se_diff[l] = m, sed
            table.append(dict(candidate=l, n_features=int(np.mean([f["candidates"][l]["n_features"] for f in sel.values() if l in f["candidates"]])),
                              mlpd_mean=m, mlpd_se=se, delta_vs_full=float(np.mean(diffs)) if diffs else float("nan"),
                              se_delta=sed, n_folds=n))
        R.write_csv(table, rep / "select_sizes.csv")
        ks = [int(l[2:]) for l in labels if l.startswith("k=")]
        mean_k = {int(l[2:]): mean[l] for l in labels if l.startswith("k=")}
        se_k = {int(l[2:]): se_diff[l] for l in labels if l.startswith("k=")}
        full_k = max(f["d_features"] for f in sel.values())       # pseudo size for the per-fold full bank
        mean_k[full_k], se_k[full_k] = mean["full"], 0.0
        chosen = S.one_se_rule(ks + [full_k], mean_k, se_k, full_k)
        chosen_label = "full" if chosen == full_k else f"k={chosen}"
        sets = [f["candidates"][chosen_label]["features"] for f in sel.values() if chosen_label in f["candidates"]]
        universe = sorted({ft for f in sel.values() for ft in f["candidates"]["full"]["features"]})
        stab = S.nogueira_stability(sets, universe)
        freq = {}
        for s in sets:
            for ft in s:
                freq[ft] = freq.get(ft, 0) + 1
        md = ["# Feature selection (projection path + held-out MLPD, one-SE rule)",
              f"folds: {len(sel)}; scoring budgets: {budgets}; full size: {full_k}; selected size: {chosen}",
              f"Nogueira stability of the selected sets across folds: {stab:.3f}", "",
              "| candidate | n_feat | MLPD | SE | Δ vs full | SE(Δ) | folds |", "|---|---|---|---|---|---|---|"]
        for r in table:
            md.append(f"| {r['candidate']} | {r['n_features']} | {r['mlpd_mean']:.4f} | {r['mlpd_se']:.4f} | "
                      f"{r['delta_vs_full']:+.4f} | {r['se_delta']:.4f} | {r['n_folds']} |")
        md.append(f"\n## Features selected at k={chosen} (fold frequency)\n")
        for ft, c in sorted(freq.items(), key=lambda kv: -kv[1]):
            md.append(f"- {ft}: {c}/{len(sets)}")
        if truth:
            md.append(f"\nsynthetic truth: common={truth['common']}, individual={truth['individual']}")
        (rep / "select.md").write_text("\n".join(md), encoding="utf-8")
        R.plot_size_curve(ks + [full_k], mean_k, se_k, chosen, full_k, rep / "select_size_curve.png")
        summary["select"] = dict(chosen_size=chosen, full_size=full_k, stability=stab)
    # ---- sensors
    sen = _load_folds(run, "_sensors")
    if sen:
        labels = sorted({l for f in sen.values() for l in f["subsets"]}, key=lambda l: (-l.count("+"), l))
        budgets = sorted({int(t) for f in sen.values() for c in f["subsets"].values() for t in c["metrics"]})
        full_label = labels[0]
        table = []
        for l in labels:
            vals, diffs = [], []
            for f in sen.values():
                c = f["subsets"].get(l); full = f["subsets"].get(full_label)
                if c and full:
                    v = np.mean([c["metrics"][str(t)]["mlpd"] for t in budgets if str(t) in c["metrics"]])
                    vf = np.mean([full["metrics"][str(t)]["mlpd"] for t in budgets if str(t) in full["metrics"]])
                    vals.append(v); diffs.append(v - vf)
            m, se, n = M.macro(vals)
            dm, sed, _ = M.macro(diffs)
            table.append(dict(subset=l, n_channels=l.count("+") + 1, mlpd_mean=m, mlpd_se=se,
                              delta_vs_full=dm, se_delta=sed, n_folds=n))
        table.sort(key=lambda r: -r["mlpd_mean"] if np.isfinite(r["mlpd_mean"]) else 1e9)
        R.write_csv(table, rep / "sensor_subsets.csv")
        md = ["# Sensor subsets (exhaustive, refit + held-out MLPD)", f"folds: {len(sen)}; scoring budgets: {budgets}", "",
              "| subset | channels | MLPD | SE | Δ vs all | SE(Δ) |", "|---|---|---|---|---|---|"]
        for r in table:
            md.append(f"| {r['subset']} | {r['n_channels']} | {r['mlpd_mean']:.4f} | {r['mlpd_se']:.4f} | "
                      f"{r['delta_vs_full']:+.4f} | {r['se_delta']:.4f} |")
        # leave-one-sensor-out
        chans = full_label.split("+")
        md.append("\n## Leave-one-sensor-out (Δ MLPD vs all channels)\n")
        for ch in chans:
            l = "+".join(c for c in chans if c != ch)
            r = next((r for r in table if r["subset"] == l), None)
            if r:
                md.append(f"- without {ch}: {r['delta_vs_full']:+.4f} ± {r['se_delta']:.4f}")
        (rep / "sensors.md").write_text("\n".join(md), encoding="utf-8")
        summary["sensors"] = dict(n_subsets=len(labels))
    R.write_json(summary, rep / "summary.json")
    log(cfg, f"[report] written to {rep}")
    return summary


# ----------------------------------------------------------------------------- stage: compare
def stage_compare(cfg, run):
    """Side-by-side tables of finished runs (prior sensitivity, budget settings, seeds)."""
    import csv
    runs = {Path(p).parent.name + "/" + Path(p).name: Path(p) for p in cfg.compare_runs}
    if not runs:
        raise SystemExit("--set compare_runs=\"('outputs/loeo/<ts>', ...)\" is required for stage compare")
    macros = {}
    for label, p in runs.items():
        f = p / "report" / "main_macro.csv"
        if not f.exists():
            log(cfg, f"[compare] {label}: no report/main_macro.csv, skipped"); continue
        macros[label] = {(r["method"], r["t"], r["metric"]): (float(r["mean"]), float(r["se"]) if r["se"] not in ("nan", "") else float("nan"))
                         for r in csv.DictReader(open(f, encoding="utf-8")) if r["mean"] not in ("nan", "")}
    ts = sorted({t for m in macros.values() for (_, t, _) in m}, key=lambda t: (t == "final", int(t) if t != "final" else 0))
    md = ["# Run comparison", "runs: " + ", ".join(macros)]
    for metric, methods in [("mlpd", ("proposed", "hb_full", "ebmap", "pooled")), ("auroc", ("proposed", "hb_full")),
                            ("ece", ("proposed",)), ("cal_slope", ("proposed",)), ("epi_mean", ("proposed",)),
                            ("ale_mean", ("proposed",)), ("correctness_auroc_epi", ("proposed",)),
                            ("epi_ale_spearman", ("proposed",)), ("auroc_ci_lo", ("proposed",)), ("reliable_frac", ("proposed",))]:
        md.append(f"\n## {metric}\n")
        md.append("| run | method | " + " | ".join(f"t={t}" for t in ts) + " |")
        md.append("|---|---|" + "---|" * len(ts))
        for label, m in macros.items():
            for meth in methods:
                cells = []
                for t in ts:
                    v = m.get((meth, t, metric))
                    cells.append("--" if v is None else (f"{v[0]:.3f} ± {v[1]:.3f}" if np.isfinite(v[1]) else f"{v[0]:.3f}"))
                md.append(f"| {label} | {meth} | " + " | ".join(cells) + " |")
    md.append("\n## Per-evaluator MLPD change t=5 vs t=0 (proposed)\n")
    for label, p in runs.items():
        cells = []
        for fj in sorted((p / "folds").glob("*.json")):
            if fj.stem.endswith("_select") or fj.stem.endswith("_sensors"):
                continue
            f = json.loads(fj.read_text(encoding="utf-8")); pr = f["proposed"]
            if "0" in pr and "5" in pr:
                cells.append(f"{f['name']} {pr['5']['mlpd'] - pr['0']['mlpd']:+.3f}")
        md.append(f"- {label}: " + ", ".join(cells))
    for name in ("select.md", "sensors.md"):
        md.append(f"\n## {name} headers\n")
        for label, p in runs.items():
            f = p / "report" / name
            if f.exists():
                head = [l for l in f.read_text(encoding="utf-8").splitlines()[:3] if l.strip()]
                md.append(f"- {label}: " + " / ".join(head[1:]))
    out = run.dir / "compare.md"
    out.write_text("\n".join(md), encoding="utf-8")
    log(cfg, f"[compare] written to {out}")
    return md


# ----------------------------------------------------------------------------- main
def main(cfg=None):
    cfg = cfg or Config()
    if cfg.fast:
        cfg = replace(cfg, **FAST)
    run = Run(cfg.run_name, cfg)
    seed_all(cfg.seed)
    (run.dir / "folds").mkdir(exist_ok=True)
    data, channels, fs, truth = load_data(cfg)
    names, elig = fold_names(cfg, data)
    if truth is not None:
        R.write_json(truth, run.dir / "synthetic_truth.json")
    log(cfg, f"[INFO] data={cfg.data} evaluators={len(data)} eligible={len(elig)} folds={len(names)} "
             f"channels={len(channels)} fs={fs} stage={cfg.stage} -> {run.dir}")
    stages = ["main", "select", "sensors", "report"] if cfg.stage == "all" else [cfg.stage]
    # Run() overwrites cfg.json on every invocation; keep one snapshot per stage for provenance.
    R.write_json(asdict(cfg), run.dir / f"cfg_{cfg.stage}.json")
    # The population is every evaluator with labels (Population.fit drops single-class ones);
    # eligibility (min_labels / min_per_class) only governs who is held out.
    for st in stages:
        tic = time.time()
        if st == "main":
            stage_main(cfg, run, data, channels, fs, names)
        elif st == "select":
            stage_select(cfg, run, data, channels, fs, names)
        elif st == "sensors":
            stage_sensors(cfg, run, data, channels, fs, names)
        elif st == "report":
            stage_report(cfg, run, truth)
        elif st == "compare":
            stage_compare(cfg, run)
        else:
            raise ValueError(f"unknown stage {st!r}")
        log(cfg, f"[{st}] finished in {(time.time() - tic) / 60:.1f} min")
    run.metrics["stages"] = {"done": stages}
    run.finish()


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default=None, choices=["main", "select", "sensors", "report", "all", "compare"])
    ap.add_argument("--data", default=None, choices=["real", "synthetic"])
    ap.add_argument("--timestamp", default=None, help="reuse an existing run folder (resume / report)")
    ap.add_argument("--fast", action="store_true", help="tiny Gibbs chains and grids for a smoke test")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override any Config field, e.g. --set budgets=\"(0,5,10,20)\" --set n_offsets=3")
    a = ap.parse_args()
    over = {}
    if a.stage: over["stage"] = a.stage
    if a.data: over["data"] = a.data
    if a.timestamp: over["timestamp"] = a.timestamp
    if a.fast: over["fast"] = True
    for kv in a.set:
        k, _, v = kv.partition("=")
        if not hasattr(Config, k):
            raise SystemExit(f"unknown config field {k!r}")
        try:
            over[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            over[k] = v
    return Config(**over)


if __name__ == "__main__":
    main(parse_args())
