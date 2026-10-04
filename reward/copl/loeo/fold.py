"""One LOEO fold of the CoPL model: encoder -> item graph -> GCF -> reward model -> adaptation.

Everything that depends on data is fitted on the population set only.  The held-out
evaluator's stream is split in recorded order into context / holdout exactly as in
reward/fully_bayesian/loeo/protocol.py, and the same budget / offset grid is used.
"""
from __future__ import annotations

import time
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from core.run import seed_all
from reward.fully_bayesian.loeo import metrics as M
from reward.fully_bayesian.loeo import protocol as P
from ..bayesian_reward_models import BayesianRM, EnsembleRM, MCDropoutRM
from ..dataset import CoPLGraphDataset
from ..gcf import CoPLGCF
from ..rm import (CNNRewardModel, MoLECNNRewardModel, PreferenceTransformerRewardModel, RewardModel,
                  RMEdgeDataset, rm_collate)
from ..trainer import CoPLGCFTrainer, CoPLRMTrainer
from . import baselines as BL
from . import encoders as E
from . import graph as G
from .data import channel_indices

E.register_extra_encoders()


def resolve_device(cfg):
    if cfg.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.device)


def dataset_namespace(cfg, device, graph_channel_names, fs):
    """CoPLGraphDataset and the similarity builders read a flat attribute namespace."""
    d = asdict(cfg)
    d.update(similarity_method=cfg.encoder, normalize=True, device=str(device),
             graph_channel_names=list(graph_channel_names), fs=float(fs))
    return SimpleNamespace(**d)


# ----------------------------------------------------------------------------- fold data
class FoldData:
    def __init__(self, cfg, gds, rm_series, rm_mean, rm_std, gidx, ridx, pop_names, gstats, seconds):
        self.cfg, self.gds = cfg, gds
        self.rm_series, self.rm_mean, self.rm_std = rm_series, rm_mean, rm_std
        self.gidx, self.ridx, self.pop_names = gidx, ridx, pop_names
        self.gstats, self.seconds = gstats, seconds

    def norm_rm(self, X):
        return (X - self.rm_mean) / self.rm_std

    @property
    def enc_meta(self):
        return getattr(self.gds, "Aii_meta", {})

    def rebuild_graph(self, cfg):
        """Re-sparsify the item graph from the fitted encoder's latents (no retraining)."""
        b = self.gds.sim_builder
        metric = E.latent_metric(b)
        A, st = G.build_graph(self.gds.Z_train, self.gds.item_owner_uid, rule=cfg.graph_rule, k=cfg.knn_k,
                              metric=metric, gamma=None if metric == "cosine" else getattr(b, "gamma", None),
                              temperature=getattr(b, "temperature", 0.2), cross_min=cfg.cross_min)
        self.gds.Aii_norm = A.to(self.gds.Apos_norm.device)
        self.gstats = st
        self.cfg = cfg
        return st


def prepare_fold(cfg, pop_data, pop_names, channels, fs, device, extra_user=None):
    """Fit encoder, item graph and normalization on the population (plus an optional oracle user)."""
    tic = time.time()
    gidx = channel_indices(channels, cfg.graph_channels)
    ridx = channel_indices(channels, cfg.rm_channels)
    order = list(pop_names)
    graph_data = {n: (pop_data[n][0][:, :, gidx], pop_data[n][1]) for n in order}
    rm_list = [pop_data[n][0][:, :, ridx] for n in order]
    if extra_user is not None:                                   # oracle: held-out context joins the graph
        name, Xc, yc = extra_user
        graph_data[name] = (Xc[:, :, gidx], yc)
        rm_list.append(Xc[:, :, ridx])
    ns = dataset_namespace(cfg, device, [channels[i] for i in gidx], fs)
    gds = CoPLGraphDataset(graph_data, ns).to(device)
    if cfg.graph_rule != "topk" or cfg.cross_min > 0:
        fd_tmp = FoldData(cfg, gds, None, None, None, gidx, ridx, order, None, 0.0)
        gstats = fd_tmp.rebuild_graph(cfg)
    else:
        gstats = G.graph_stats(gds.Aii_norm, gds.item_owner_uid)
    rm_series = np.concatenate(rm_list, axis=0).astype(np.float32)
    rm_mean = rm_series.mean(axis=(0, 1), keepdims=True)
    rm_std = rm_series.std(axis=(0, 1), keepdims=True) + 1e-6
    rm_series = (rm_series - rm_mean) / rm_std
    return FoldData(cfg, gds, rm_series, rm_mean, rm_std, gidx, ridx, order, gstats, time.time() - tic)


# ----------------------------------------------------------------------------- models
RM_MODELS = {
    "mlp": lambda cfg, obs: RewardModel(obs, cfg.gcf_emb_dim, hidden=cfg.rm_mlp_hidden, dropout=cfg.rm_dropout),
    "cnn": lambda cfg, obs: CNNRewardModel(obs_dim=obs, user_dim=cfg.gcf_emb_dim, hidden=cfg.rm_hidden,
                                           mlp_hidden=cfg.rm_mlp_hidden, kernel_size=cfg.rm_kernel_size,
                                           layers=cfg.rm_layers, dropout=cfg.rm_dropout),
    "mole_cnn": lambda cfg, obs: MoLECNNRewardModel(obs_dim=obs, user_dim=cfg.gcf_emb_dim, hidden=cfg.rm_hidden,
                                                    mlp_hidden=cfg.rm_mlp_hidden, kernel_size=cfg.rm_kernel_size,
                                                    layers=cfg.rm_layers, num_experts=cfg.rm_num_experts,
                                                    rank=cfg.rm_mole_rank, tau=cfg.rm_mole_tau),
    "preference_transformer": lambda cfg, obs: PreferenceTransformerRewardModel(
        obs_dim=obs, user_dim=cfg.gcf_emb_dim, hidden=cfg.rm_hidden, num_heads=cfg.rm_num_heads,
        num_layers=cfg.rm_layers, max_len=cfg.rm_max_len),
}


def build_gcf(cfg, gds, device):
    Z = torch.tensor(gds.Z_train, dtype=torch.float32)
    if Z.shape[1] != cfg.gcf_emb_dim:
        proj = torch.nn.Linear(Z.shape[1], cfg.gcf_emb_dim, bias=False)
        torch.nn.init.xavier_uniform_(proj.weight)
        with torch.no_grad():
            Z = proj(Z)
    return CoPLGCF(n_u=gds.n_users, n_i=gds.n_items, d=cfg.gcf_emb_dim, pos_adj_norm=gds.Apos_norm,
                   neg_adj_norm=gds.Aneg_norm, dropout=cfg.gcf_dropout, l=cfg.gcf_layers,
                   item_item_adj_norm=gds.Aii_norm, item_item_weight=cfg.item_item_weight if cfg.use_item_item else 0.0,
                   loss_type=cfg.gcf_loss_type, loss_kwargs=cfg.gcf_loss_kwargs, item_feat_init=Z,
                   m_i_type=cfg.gcf_m_i_type).to(device)


class Models:
    def __init__(self, E_u, E_i, rm, stats):
        self.E_u, self.E_i, self.rm, self.stats = E_u, E_i, rm, stats


def train_models(cfg, fd, device, seed, verbose=0):
    """GCF embeddings and the (possibly Bayesian) reward model for one seed."""
    seed_all(seed)
    gds = fd.gds
    tic = time.time()
    gcf = build_gcf(cfg, gds, device)
    gcf_cfg = {"device": str(device), "gcf_lr": cfg.gcf_lr, "gcf_weight_decay": cfg.gcf_weight_decay,
               "gcf_lambda_reg": cfg.gcf_lambda_reg, "gcf_epochs": cfg.gcf_epochs, "use_pos_weight": cfg.use_pos_weight}
    gcf_auc, _, E_u, E_i, _ = CoPLGCFTrainer(gcf, gcf_cfg, log_dir=None).train(
        gds.tr_u, gds.tr_i, gds.tr_y, gds.va_u, gds.va_i, gds.va_y, verbose=verbose)
    E_u = E_u.detach().to(device); E_i = E_i.detach()
    t_gcf = time.time() - tic

    tic = time.time()
    tr = DataLoader(RMEdgeDataset(gds.tr_u, gds.tr_i, gds.tr_y, fd.rm_series), batch_size=cfg.rm_batch_size,
                    shuffle=True, collate_fn=rm_collate)
    va = DataLoader(RMEdgeDataset(gds.va_u, gds.va_i, gds.va_y, fd.rm_series), batch_size=cfg.rm_batch_size,
                    shuffle=False, collate_fn=rm_collate)
    rm_cfg = {"device": str(device), "rm_lr": cfg.rm_lr, "rm_weight_decay": cfg.rm_weight_decay,
              "rm_lambda_reg": cfg.rm_lambda_reg, "rm_epochs": cfg.rm_epochs, "use_pos_weight": cfg.use_pos_weight}
    obs_dim = fd.rm_series.shape[2]
    rm_aucs = []
    if cfg.rm_bayes == "ensemble":
        members = []
        for k in range(cfg.rm_ensemble_k):
            seed_all(seed * 100 + k)
            m = RM_MODELS[cfg.rm_model](cfg, obs_dim).to(device)
            auc, _ = CoPLRMTrainer(m, rm_cfg, log_dir=None).train(tr, va, E_u, gds.tr_y, verbose=verbose)
            rm_aucs.append(auc); members.append(m)
        rm = EnsembleRM(members)
    else:
        m = RM_MODELS[cfg.rm_model](cfg, obs_dim).to(device)
        auc, _ = CoPLRMTrainer(m, rm_cfg, log_dir=None).train(tr, va, E_u, gds.tr_y, verbose=verbose)
        rm_aucs.append(auc)
        rm = MCDropoutRM(m, cfg.rm_mc_samples) if cfg.rm_bayes == "mc_dropout" else m
    rm.eval()
    stats = dict(gcf_val_auc=float(gcf_auc), rm_val_auc=float(np.mean(rm_aucs)), seconds_gcf=t_gcf,
                 seconds_rm=time.time() - tic, n_train_users=int(gds.n_users), n_items=int(gds.n_items))
    return Models(E_u, E_i, rm, stats)


# ----------------------------------------------------------------------------- evaluation
def _entropy(w):
    w = np.clip(np.asarray(w, float), 1e-12, 1)
    return float(-(w * np.log(w)).sum())


def evaluate_copl(cfg, fd, models, X_held, y_held, seed, device, budgets=None, oracle_uid=None):
    """{t: metrics}, {t: lpd}, info, {t: mean adaptation weights}.  Cold start = uniform weights."""
    gds = fd.gds
    Xg = gds.norm(X_held[:, :, fd.gidx]); Xr = fd.norm_rm(X_held[:, :, fd.ridx])
    y_held = np.asarray(y_held).astype(int)
    ctx_idx, hold_idx = P.split_stream(len(y_held), cfg.ctx_frac)
    Xg_ctx, y_ctx = Xg[ctx_idx], y_held[ctx_idx]
    y_hold = y_held[hold_idx]
    n_ctx = len(y_ctx)
    ts, offs = P._budgets_and_offsets(cfg, budgets if budgets is not None else P.budget_grid(cfg, n_ctx), n_ctx)
    if oracle_uid is not None:
        ts, offs = [n_ctx], [0]
    hold_obs = torch.as_tensor(Xr[hold_idx], dtype=torch.float32, device=device)
    E_u, E_i, rm = models.E_u, models.E_i, models.rm
    U = E_u.shape[0]
    per_budget = {t: [] for t in ts}; lpds = {t: [] for t in ts}; wus = {t: [] for t in ts}
    width_max = getattr(cfg, "reliable_w_max", 0.3)
    for o in offs:
        for t in ts:
            if oracle_uid is not None:
                w_u = np.eye(U)[oracle_uid]; e_u = E_u[oracle_uid]
            elif t == 0 or not cfg.use_adapt:
                w_u = np.full(U, 1.0 / U); e_u = E_u.mean(dim=0)
            else:
                lo, hi = o, min(o + t, n_ctx)
                if hi - lo < t and o > 0:
                    continue
                _, nidx, nw = gds.attach_test_items(Xg_ctx[lo:hi], E_i.cpu(), topk=cfg.adapt_topk, device=device)
                e_u, w_u = gds.adapt_test_user(y_ctx[lo:hi], nidx, nw, E_u, device=device)
            emb = e_u.unsqueeze(0).expand(len(y_hold), -1)
            with torch.no_grad():
                if isinstance(rm, BayesianRM):
                    Pm = rm.sample_probs(emb, hold_obs).cpu().numpy()
                    m, lpd = M.summarize(y_hold, Pm, seed=seed, width_max=width_max)
                else:
                    p = torch.sigmoid(rm(emb, hold_obs)).cpu().numpy()
                    m, lpd = M.summarize_point(y_hold, p)
            m["t"], m["offset"], m["w_entropy"] = int(t), int(o), _entropy(w_u)
            per_budget[t].append(m); lpds[t].append(lpd); wus[t].append(np.asarray(w_u, float))
    out, wu_mean = {}, {}
    for t in ts:
        if not per_budget[t]:
            continue
        keys = [k for k in per_budget[t][0] if isinstance(per_budget[t][0][k], (int, float, bool))]
        agg = {k: P._nanmean([mm[k] for mm in per_budget[t]]) for k in keys}
        agg["n_offsets"] = len(per_budget[t])
        if "reliable" in per_budget[t][0]:
            agg["reliable_frac"] = float(np.mean([mm["reliable"] for mm in per_budget[t]]))
        out[t] = agg
        wu_mean[t] = np.mean(np.stack(wus[t]), axis=0).tolist()
    lpd_mean = {t: np.mean(np.stack(v), axis=0) for t, v in lpds.items() if v}
    info = dict(n_ctx=n_ctx, n_hold=int(len(y_hold)), budgets=ts, offsets=offs)
    return out, lpd_mean, info, wu_mean


def evaluate_knn_vote(cfg, fd, X_held, y_held, budgets):
    """Context-independent vote; replicated over budgets so the tables align."""
    kv = BL.KNNVote(cfg).fit_population(fd.gds)
    y_held = np.asarray(y_held).astype(int)
    ctx_idx, hold_idx = P.split_stream(len(y_held), cfg.ctx_frac)
    Xg = fd.gds.norm(X_held[:, :, fd.gidx])
    p = kv.predict_graph(Xg[hold_idx])
    m, lpd = M.summarize_point(y_held[hold_idx], p)
    m["n_offsets"] = 1
    return {t: dict(m) for t in budgets}, {t: lpd for t in budgets}


def evaluate_point_baseline(cfg, model, fd, X_held, y_held, budgets):
    """Pooled / Indep through the shared protocol helper (predict(X_ctx, y_ctx, X_hold))."""
    Xr = fd.norm_rm(X_held[:, :, fd.ridx])
    return P.evaluate_baseline(cfg, model, Xr, np.asarray(y_held).astype(int), budgets, particles=False)


# ----------------------------------------------------------------------------- one fold, all methods
def seed_average(per_seed, prefix, budgets):
    keys = [k for k in per_seed if k.startswith(prefix + "/")]
    agg = {}
    for t in budgets:
        rows = [per_seed[k][t] for k in keys if t in per_seed[k]]
        if rows:
            agg[str(t)] = {m: float(np.nanmean([r[m] for r in rows])) for m in rows[0]}
    return agg


def run_fold(cfg, name, data, channels, fs, device, seeds=None, baselines=True, budgets=None, log=print,
             oracle=False, encoder_per_seed=True):
    """Full evaluation of one held-out evaluator. Returns the fold dict written to folds/<name>.json."""
    tic = time.time()
    seeds = list(seeds or cfg.seeds)
    pop_names = [n for n in data if n != name]
    pop_data = {n: data[n] for n in pop_names}
    X_held, y_held = data[name]
    y_held = np.asarray(y_held).astype(int)
    fold = dict(name=name, n=int(len(y_held)), n_pos=int(y_held.sum()), budgets_cfg=list(cfg.budgets),
                encoder=cfg.encoder, graph_rule=cfg.graph_rule, knn_k=cfg.knn_k,
                graph_channels=list(cfg.graph_channels or channels), rm_channels=list(cfg.rm_channels or channels),
                copl={}, lpd={}, per_seed={}, fit_stats={}, w_u={}, graph={}, enc_meta={})
    extra = None
    oracle_uid = None
    if oracle:
        ctx_idx, _ = P.split_stream(len(y_held), cfg.ctx_frac)
        extra = (name, X_held[ctx_idx], y_held[ctx_idx])
    fd = None
    ts_all = None
    for si, seed in enumerate(seeds):
        if fd is None or encoder_per_seed:
            seed_all(seed)
            fd = prepare_fold(replace(cfg, seed=seed), pop_data, pop_names, channels, fs, device, extra_user=extra)
            fold["graph"][str(seed)] = fd.gstats
            fold["enc_meta"][str(seed)] = fd.enc_meta
            fold["fit_stats"][f"encoder/{seed}"] = dict(seconds=fd.seconds)
            if oracle:
                oracle_uid = fd.gds.user_to_uid[name]
        models = train_models(cfg, fd, device, seed, verbose=max(0, cfg.verbose - 1))
        fold["fit_stats"][f"copl/{seed}"] = models.stats
        res, lpd, info, wu = evaluate_copl(cfg, fd, models, X_held, y_held, seed, device, budgets=budgets,
                                           oracle_uid=oracle_uid)
        ts_all = info["budgets"]
        fold["per_seed"][f"copl/{seed}"] = res
        fold["lpd"].setdefault("copl", {})
        for t, v in lpd.items():
            fold["lpd"]["copl"].setdefault(str(t), []).append(v.tolist())
        for t, v in wu.items():
            fold["w_u"].setdefault(str(t), []).append(v)
        if si == 0:
            fold["info"] = info
            fold["population_users"] = list(fd.gds.train_drivers)
        log(f"[fold] {name} seed={seed}: " + "  ".join(f"t={t}: {res[t]['mlpd']:.3f}" for t in res)
            + f"  (gcf auc {models.stats['gcf_val_auc']:.3f}, rm auc {models.stats['rm_val_auc']:.3f})")
    fold["copl"] = seed_average(fold["per_seed"], "copl", ts_all)
    fold["lpd"]["copl"] = {t: np.mean(np.asarray(v), axis=0).tolist() for t, v in fold["lpd"]["copl"].items()}
    fold["w_u"] = {t: np.mean(np.asarray(v), axis=0).tolist() for t, v in fold["w_u"].items()}
    if baselines and not oracle:
        bts = ts_all
        if cfg.knn_vote:
            tic_b = time.time()
            res_k, lpd_k = evaluate_knn_vote(cfg, fd, X_held, y_held, bts)
            fold["knn_vote"] = {str(t): v for t, v in res_k.items()}
            fold["lpd"]["knn_vote"] = {str(t): np.asarray(v).tolist() for t, v in lpd_k.items()}
            fold["fit_stats"]["knn_vote"] = dict(seconds=time.time() - tic_b)
        for flag, cls in (("pooled", BL.PooledCNN), ("indep", BL.IndepCNN)):
            if not getattr(cfg, flag):
                continue
            tic_b = time.time()
            seed_all(seeds[0])
            bl = cls(cfg, device).fit_population(fd.gds, fd.rm_series, verbose=max(0, cfg.verbose - 1))
            res_b, lpd_b = evaluate_point_baseline(cfg, bl, fd, X_held, y_held, bts)
            fold[bl.name] = {str(t): v for t, v in res_b.items()}
            fold["lpd"][bl.name] = {str(t): np.asarray(v).tolist() for t, v in lpd_b.items()}
            fold["fit_stats"][bl.name] = dict(seconds=time.time() - tic_b, val_auc=getattr(bl, "val_auc", None))
            log(f"[fold] {name} {bl.name}: " + "  ".join(f"t={t}: {res_b[t]['mlpd']:.3f}" for t in res_b))
    fold["seconds"] = time.time() - tic
    return fold
