"""CoPL under the leave-one-evaluator-out protocol (docs/copl/claude_notes/01_copl_plan.html).

Stages
  encoders  intrinsic graph metrics per encoder x channel set (no GCF / RM training; fast)
  tune      inner LOEO inside each population set, random search over Config.tune_space
  main      CoPL + baselines (pooled fine-tuned CNN, independent CNN, k-NN vote) per fold and seed
  ablate    no item-item graph / no adaptation / oracle embedding (seed 0)
  sweep     graph rule x k x item_item_weight, encoder fitted once per fold (seed 0)
  channels  leave-one-channel-out + a_z only + IMU only, end to end (seed 0)
  report    tables and figures from the fold json files
  all       encoders, main, ablate, sweep, channels, report

Examples
  python run_copl_loeo.py --stage encoders
  python run_copl_loeo.py --stage main --set seeds="(42,43,44)"
  python run_copl_loeo.py --stage report
  python run_copl_loeo.py --stage main --data synthetic --fast

Fold results are written as they finish (folds/<evaluator>*.json); a run can be resumed with
--timestamp <folder>, and report / ablate / sweep / channels / tune default to the latest run.
"""
from __future__ import annotations

import argparse
import ast
import itertools
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from core.run import Run, seed_all
from reward.fully_bayesian.loeo import metrics as M
from reward.fully_bayesian.loeo import protocol as P
from reward.fully_bayesian.loeo import report as R
from reward.copl.loeo import data as D
from reward.copl.loeo import encoders as E
from reward.copl.loeo import fold as FD
from reward.copl.loeo import report as CR
from reward.copl.loeo.config import FAST, Config


def log(cfg, *a):
    if cfg.verbose:
        print(*a, flush=True)


def _write(obj, path):
    R.write_json(obj, path)


def _tuned(cfg, run, name):
    """Apply folds/<name>_tune.json if requested and present."""
    p = run.dir / "folds" / f"{name}_tune.json"
    if cfg.use_tuned and p.exists():
        best = json.loads(p.read_text(encoding="utf-8"))["best"]
        log(cfg, f"[tune] {name}: using {best}")
        return replace(cfg, **best)
    return cfg


# ----------------------------------------------------------------------------- stage: main
def stage_main(cfg, run, data, channels, fs, names, device):
    for name in names:
        out = run.dir / "folds" / f"{name}.json"
        if out.exists():
            log(cfg, f"[main] {name}: exists, skipping"); continue
        cfg_f = _tuned(cfg, run, name)
        fold = FD.run_fold(cfg_f, name, data, channels, fs, device, log=lambda *a: log(cfg, *a))
        fold["cfg_used"] = {k: v for k, v in asdict(cfg_f).items() if k in cfg.tune_space or k in ("encoder", "graph_rule")}
        _write(fold, out)
        log(cfg, f"[main] {name}: done in {fold['seconds']:.0f}s")


# ----------------------------------------------------------------------------- stage: ablate
ABLATIONS = {"no_item_item": dict(use_item_item=False), "no_adapt": dict(use_adapt=False), "oracle": dict(oracle=True)}


def _variant_record(fold):
    return dict(copl=fold["copl"], lpd=fold["lpd"].get("copl"), graph=next(iter(fold["graph"].values()), None),
                info=fold.get("info"), fit_stats=fold["fit_stats"], seconds=fold["seconds"])


def stage_ablate(cfg, run, data, channels, fs, names, device):
    for name in names:
        out = run.dir / "folds" / f"{name}_ablate.json"
        rec = json.loads(out.read_text(encoding="utf-8")) if out.exists() else dict(name=name, ablate={})
        cfg_f = _tuned(cfg, run, name)
        for label, over in ABLATIONS.items():
            if label in rec["ablate"]:
                continue
            oracle = over.get("oracle", False)
            c = replace(cfg_f, **{k: v for k, v in over.items() if k != "oracle"})
            fold = FD.run_fold(c, name, data, channels, fs, device, seeds=cfg.seeds[:1], baselines=False,
                               log=lambda *a: log(cfg, *a), oracle=oracle)
            rec["ablate"][label] = _variant_record(fold)
            _write(rec, out)
            log(cfg, f"[ablate] {name} {label}: " + "  ".join(f"t={t}: {v['mlpd']:.3f}" for t, v in fold["copl"].items()))


# ----------------------------------------------------------------------------- stage: sweep
def stage_sweep(cfg, run, data, channels, fs, names, device):
    for name in names:
        out = run.dir / "folds" / f"{name}_sweep.json"
        rec = json.loads(out.read_text(encoding="utf-8")) if out.exists() else dict(name=name, sweep={})
        cfg_f = _tuned(cfg, run, name)
        pop_names, pop_data = D.population_split(data, name)
        X_held, y_held = data[name]
        seed = cfg.seeds[0]
        fd = None
        for rule, k, w in itertools.product(cfg.sweep_rules, cfg.sweep_ks, cfg.sweep_item_item_weights):
            label = f"{rule}/k={k}/w={w}"
            if label in rec["sweep"]:
                continue
            if fd is None:
                seed_all(seed)
                fd = FD.prepare_fold(replace(cfg_f, seed=seed), pop_data, pop_names, channels, fs, device)
            c = replace(cfg_f, graph_rule=rule, knn_k=k, item_item_weight=w,
                        cross_min=(k // 2 if rule == "cross_forced" else 0))
            tic = time.time()
            st = fd.rebuild_graph(c)
            models = FD.train_models(c, fd, device, seed, verbose=0)
            res, lpd, info, _ = FD.evaluate_copl(c, fd, models, X_held, y_held, seed, device)
            rec["sweep"][label] = dict(copl={str(t): v for t, v in res.items()},
                                       lpd={str(t): v.tolist() for t, v in lpd.items()}, graph=st, info=info,
                                       fit_stats=models.stats, seconds=time.time() - tic)
            _write(rec, out)
            log(cfg, f"[sweep] {name} {label}: rho_cross={st['rho_cross']:.3f} comps={st['n_components']}  "
                     + "  ".join(f"t={t}: {v['mlpd']:.3f}" for t, v in res.items()))


# ----------------------------------------------------------------------------- stage: channels
def stage_channels(cfg, run, data, channels, fs, names, device):
    sets = D.channel_sets(channels, "loso")
    keep = ["full"] + [k for k in sets if k.startswith("without_")] + ["imu_only"] + \
           [k for k in sets if k.startswith("only_") and "VerAccel" in k]
    sets = {k: sets[k] for k in keep if k in sets}
    for name in names:
        out = run.dir / "folds" / f"{name}_channels.json"
        rec = json.loads(out.read_text(encoding="utf-8")) if out.exists() else dict(name=name, channels={})
        cfg_f = _tuned(cfg, run, name)
        for label, subset in sets.items():
            if label in rec["channels"]:
                continue
            c = replace(cfg_f, graph_channels=tuple(subset), rm_channels=tuple(subset))
            fold = FD.run_fold(c, name, data, channels, fs, device, seeds=cfg.seeds[:1], baselines=False,
                               log=lambda *a: log(cfg, *a))
            rec["channels"][label] = dict(channels=list(subset), **_variant_record(fold))
            _write(rec, out)
            log(cfg, f"[channels] {name} {label}: " + "  ".join(f"t={t}: {v['mlpd']:.3f}" for t, v in fold["copl"].items()))


# ----------------------------------------------------------------------------- stage: encoders
def stage_encoders(cfg, run, data, channels, fs, names, device):
    sets = D.channel_sets(channels, cfg.enc_channel_sets)
    for name in names:
        out = run.dir / "folds" / f"{name}_encoders.json"
        rec = json.loads(out.read_text(encoding="utf-8")) if out.exists() else dict(name=name, encoders={})
        pop_names, pop_data = D.population_split(data, name)
        X_held, y_held = data[name]
        y_held = np.asarray(y_held).astype(int)
        _, hold_idx = P.split_stream(len(y_held), cfg.ctx_frac)
        for enc in cfg.enc_list:
            rec["encoders"].setdefault(enc, {})
            for cs_label, subset in sets.items():
                if cs_label in rec["encoders"][enc]:
                    continue
                tic = time.time()
                c = replace(cfg, encoder=enc, graph_channels=tuple(subset), rm_channels=tuple(subset), seed=cfg.seeds[0])
                try:
                    seed_all(cfg.seeds[0])
                    fd = FD.prepare_fold(c, pop_data, pop_names, channels, fs, device)
                except Exception as ex:                       # e.g. DTW too slow / encoder unsupported
                    log(cfg, f"[encoders] {name} {enc} {cs_label}: failed ({ex})")
                    rec["encoders"][enc][cs_label] = dict(error=str(ex)); _write(rec, out); continue
                gds = fd.gds
                y_pop = np.zeros(gds.n_items, dtype=int)
                for uid, (ids, yy) in gds.per_user_items.items():
                    y_pop[ids] = yy
                metric = E.latent_metric(gds.sim_builder)
                pop = E.intrinsic_metrics(gds.Z_train, y_pop, gds.item_owner_uid, cfg.enc_ks, metric=metric,
                                          alpha=cfg.knn_vote_alpha)
                Zq = gds.sim_builder.transform_test(gds.norm(X_held[:, :, fd.gidx])[hold_idx])
                held = {}
                yh = y_held[hold_idx]
                for k in cfg.enc_ks:
                    p = E.heldout_vote(gds.Z_train, y_pop, Zq, k, metric=metric, alpha=cfg.knn_vote_alpha,
                                       gamma=None if metric == "cosine" else getattr(gds.sim_builder, "gamma", None))
                    m, _ = M.summarize_point(yh, p)
                    held[int(k)] = dict(heldout_vote_auroc=m["auroc"], heldout_vote_mlpd=m["mlpd"], heldout_vote_brier=m["brier"])
                rec["encoders"][enc][cs_label] = dict(channels=list(subset), pop={str(k): v for k, v in pop.items()},
                                                     held={str(k): v for k, v in held.items()}, graph=fd.gstats,
                                                     meta=fd.enc_meta, seconds=time.time() - tic)
                _write(rec, out)
                k0 = cfg.enc_ks[min(2, len(cfg.enc_ks) - 1)]
                log(cfg, f"[encoders] {name} {enc:8s} {cs_label:28s} k={k0}: agree_cross={pop[k0]['agree_cross']:.3f} "
                         f"rho_cross={pop[k0]['rho_cross']:.3f} vote_auroc={pop[k0]['vote_auroc']:.3f} "
                         f"held_mlpd={held[k0]['heldout_vote_mlpd']:.3f}  [{time.time() - tic:.0f}s]")


# ----------------------------------------------------------------------------- stage: tune
def _sample_trials(cfg):
    rng = np.random.default_rng(cfg.tune_seed)
    keys = list(cfg.tune_space)
    grid = list(itertools.product(*[cfg.tune_space[k] for k in keys]))
    idx = rng.permutation(len(grid))[: cfg.tune_trials]
    return [dict(zip(keys, grid[i])) for i in idx]


def stage_tune(cfg, run, data, channels, fs, names, device):
    trials = _sample_trials(cfg)
    for name in names:
        out = run.dir / "folds" / f"{name}_tune.json"
        rec = json.loads(out.read_text(encoding="utf-8")) if out.exists() else dict(name=name, trials={}, inner=[])
        pop_names, pop_data = D.population_split(data, name)
        # inner held-out: the population evaluators with the most labels (both classes present)
        elig = D.fold_names(cfg, pop_data)[1]
        inner = sorted(elig, key=lambda n: -len(elig[n][1]))[: cfg.tune_inner_folds]
        rec["inner"] = inner
        seed = cfg.seeds[0]
        cache = {}
        for i, tr in enumerate(trials):
            label = json.dumps(tr, sort_keys=True)
            if label in rec["trials"]:
                continue
            scores = {}
            for inner_name in inner:
                in_pop = [n for n in pop_names if n != inner_name]
                in_data = {n: pop_data[n] for n in in_pop}
                if inner_name not in cache:
                    seed_all(seed)
                    cache[inner_name] = FD.prepare_fold(replace(cfg, seed=seed), in_data, in_pop, channels, fs, device)
                fd = cache[inner_name]
                c = replace(cfg, **tr, graph_rule="topk", cross_min=0)
                if c.knn_k != fd.cfg.knn_k:
                    fd.rebuild_graph(c)
                models = FD.train_models(c, fd, device, seed, verbose=0)
                res, _, _, _ = FD.evaluate_copl(c, fd, models, *pop_data[inner_name], seed, device,
                                                budgets=list(cfg.tune_budgets))
                scores[inner_name] = float(np.mean([v["mlpd"] for v in res.values()]))
            rec["trials"][label] = dict(params=tr, scores=scores, score=float(np.mean(list(scores.values()))))
            best_label = max(rec["trials"], key=lambda l: rec["trials"][l]["score"])
            rec["best"] = rec["trials"][best_label]["params"]; rec["best_score"] = rec["trials"][best_label]["score"]
            _write(rec, out)
            log(cfg, f"[tune] {name} trial {i + 1}/{len(trials)} score={rec['trials'][label]['score']:.4f}  best={rec['best_score']:.4f} {rec['best']}")


# ----------------------------------------------------------------------------- stage: report
def stage_report(cfg, run):
    rep = run.dir / "report"
    rep.mkdir(exist_ok=True)
    summary = {}
    folds = CR.load_folds(run.dir)
    if folds:
        rows = CR.report_main(folds, cfg, rep)
        summary["main"] = {r["method"] + f"/t={r['t']}": r["mean"] for r in rows if r["metric"] == "mlpd"}
    md_extra = []
    for key, suffix, label_key in (("ablate", "_ablate", "variant"), ("sweep", "_sweep", "graph"),
                                   ("channels", "_channels", "channel_set")):
        fv = CR.load_folds(run.dir, suffix)
        if fv:
            CR.normalize_variants(fv, key, cfg.budgets)      # per-fold t = n_ctx -> shared key "final"
            rows = CR.report_table(fv, key, rep, f"{key}.csv", label_key=label_key)
            md_extra.append(CR.md_variant_table(rows, label_key, f"{key} (held-out MLPD, macro ± SE)"))
    if folds and md_extra:
        with open(rep / "main.md", "a", encoding="utf-8") as fh:
            fh.write("\n" + "\n".join(md_extra))
    elif md_extra:
        (rep / "variants.md").write_text("\n".join(md_extra), encoding="utf-8")
    fe = CR.load_folds(run.dir, "_encoders")
    if fe:
        CR.report_encoders(fe, rep)
    ft = CR.load_folds(run.dir, "_tune")
    if ft:
        R.write_csv([dict(fold=n, best_score=f.get("best_score"), **(f.get("best") or {})) for n, f in ft.items()],
                    rep / "tune_best.csv")
    _write(summary, rep / "summary.json")
    log(cfg, f"[report] written to {rep}")


# ----------------------------------------------------------------------------- main
READ_ONLY = ("report", "ablate", "sweep", "channels", "tune")


def main(cfg=None):
    cfg = cfg or Config()
    if cfg.fast:
        cfg = replace(cfg, **FAST)
    if cfg.timestamp is None and cfg.stage in READ_ONLY:
        existing = sorted(p.name for p in (Path("outputs") / cfg.run_name).glob("*")
                          if p.is_dir() and (p / "folds").is_dir() and p.name != "test")
        if existing and cfg.stage == "report" and not any((Path("outputs") / cfg.run_name / existing[-1] / "folds").glob("*.json")):
            existing = existing[:-1]
        if existing:
            cfg = replace(cfg, timestamp=existing[-1])
            print(f"[INFO] --timestamp not given; using latest run outputs/{cfg.run_name}/{cfg.timestamp}")
        elif cfg.stage == "report":
            raise SystemExit(f"no run under outputs/{cfg.run_name}; run a producing stage first")
    run = Run(cfg.run_name, cfg)
    seed_all(cfg.seed)
    (run.dir / "folds").mkdir(exist_ok=True)
    device = FD.resolve_device(cfg)
    data, channels, fs = D.load_data(cfg)
    names, elig = D.fold_names(cfg, data)
    log(cfg, f"[INFO] data={cfg.data} evaluators={len(data)} eligible={len(elig)} folds={len(names)} "
             f"channels={channels} fs={fs} device={device} stage={cfg.stage} -> {run.dir}")
    stages = ["encoders", "main", "ablate", "sweep", "channels", "report"] if cfg.stage == "all" else [cfg.stage]
    R.write_json(asdict(cfg), run.dir / f"cfg_{cfg.stage}.json")
    for st in stages:
        tic = time.time()
        fn = {"main": stage_main, "ablate": stage_ablate, "sweep": stage_sweep, "channels": stage_channels,
              "encoders": stage_encoders, "tune": stage_tune}.get(st)
        if fn is not None:
            fn(cfg, run, data, channels, fs, names, device)
        elif st == "report":
            stage_report(cfg, run)
        else:
            raise ValueError(f"unknown stage {st!r}")
        log(cfg, f"[{st}] finished in {(time.time() - tic) / 60:.1f} min")
    run.metrics["stages"] = {"done": stages}
    run.finish()


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default=None, choices=["encoders", "tune", "main", "ablate", "sweep", "channels", "report", "all"])
    ap.add_argument("--data", default=None, choices=["real", "synthetic"])
    ap.add_argument("--timestamp", default=None)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
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
