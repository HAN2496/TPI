from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader

from loader import Dataset, View
from core import (Run, split_ctx, grid, Track,
                      evaluate_predictions, plot_sequential_auroc, plot_training_curves)
from models.copl.dataset import CoPLGraphDataset
from models.copl.gcf import CoPLGCF
from models.copl.rm import (RewardModel, CNNRewardModel, MoLECNNRewardModel,
                         PreferenceTransformerRewardModel, RMEdgeDataset, rm_collate)
from models.copl.similarity import build_similarity
from models.copl.trainer import CoPLGCFTrainer, CoPLRMTrainer
from models.copl import viz




@dataclass
class Config:
    train: tuple = ("조현석", "한규택", "박재일", "이지환", "강신길")
    test: tuple = ("김재호", "김진명", "김태근", "신민철", "이강근")
    view: View = View(
        features=("Pitch_rate_6D", "Bounce_rate_6D", "IMU_LongAccelVal", "IMU_LatAccelVal",
                  "IMU_YawRtVal", "Roll_rate_6D", "SAS_AnglVal", "SAS_SpdVal", "IMU_RollRtVal",
                  "VCU_AccPedDepVal", "IEB_StrkDpthPcVal", "IEB_BrkActvSta", "IEB_EstTtlBrkFrcNmV"),
        around=(-1, 1.5), downsample=5, smooth=(15.0, 2))
    normalize: bool = True
    val_size: float = 0.1

    timestamp: str = None                      # None=새 실험, "test"=디버그 고정폴더, "YYYY..."=eval only
    load: dict = field(default_factory=lambda: {"sim": None, "gcf": None, "rm": None})  # 컴포넌트별 재사용할 run timestamp

    # GCF
    gcf_model: str = "gcf"                     # "gcf" | "gcf_gcn"
    gcf_m_i_type: str = "b"
    gcf_loss_type: str = "bce_diversity"       # "bce" | "bpr" | "cosine" | "softmax" | "margin"
    gcf_emb_dim: int = 32
    gcf_layers: int = 2
    gcf_dropout: float = 0.0
    item_item_weight: float = 0.5
    gcf_lr: float = 0.00068
    gcf_weight_decay: float = 0.001
    gcf_lambda_reg: float = 0.0
    gcf_epochs: int = 50
    use_pos_weight: bool = True
    gcf_loss_kwargs: dict = field(default_factory=lambda: {
        "w_ii": 2.0,         # item-item interaction weight
        "lambda_div": 0.5,   # bce_diversity loss
        "margin": 0.5,       # cosine, margin loss
        "temperature": 0.1,  # softmax loss
    })

    # Similarity graph
    similarity_method: str = "ae"              # "ae" | "vae" | "pca" | "kernel_pca" | "dtw"
    mutual: bool = True
    knn_k: int = 30
    gamma_mul: float = 1.0
    pca_dim: int = 2
    dtw_gamma: float = 1.0
    ae_latent_dim: int = 8
    ae_epochs: int = 500
    ae_lr: float = 0.001
    ae_batch_size: int = 128
    ae_hidden_channels: int = 32
    ae_metric: str = "cosine"
    ae_temperature: float = 0.2
    vae_latent_dim: int = 16
    vae_epochs: int = 400
    vae_lr: float = 0.001
    vae_kl_weight: float = 0.05
    vae_batch_size: int = 128
    vae_hidden_channels: int = 32
    vae_metric: str = "cosine"
    vae_temperature: float = 0.2

    # RM
    rm_model: str = "cnn"                      # "mlp" | "cnn" | "mole_cnn" | "preference_transformer"
    rm_hidden: int = 32
    rm_mlp_hidden: int = 64
    rm_lr: float = 0.00026
    rm_weight_decay: float = 0.0
    rm_lambda_reg: float = 1e-6
    rm_epochs: int = 100
    rm_batch_size: int = 256
    rm_num_experts: int = 3
    rm_mole_rank: int = 6
    rm_mole_tau: float = 2.0
    rm_kernel_size: int = 3
    rm_layers: int = 2
    rm_num_heads: int = 8
    rm_max_len: int = 1000

    # Test-time adaptation
    adapt_topk: int = 30
    adapt_use_neg: bool = True
    adapt_neg_weight: float = 1.0
    adapt_user_softmax_temp: float = 1.15

    seed: int = 42
    device: str = "cuda"
    verbose: int = 1


RM_MODELS = {
    "mlp": lambda cfg, obs: RewardModel(obs_dim=obs, user_dim=cfg.gcf_emb_dim, hidden=cfg.rm_mlp_hidden),
    "cnn": lambda cfg, obs: CNNRewardModel(obs_dim=obs, user_dim=cfg.gcf_emb_dim, hidden=cfg.rm_hidden,
                                           mlp_hidden=cfg.rm_mlp_hidden, kernel_size=cfg.rm_kernel_size,
                                           layers=cfg.rm_layers),
    "mole_cnn": lambda cfg, obs: MoLECNNRewardModel(obs_dim=obs, user_dim=cfg.gcf_emb_dim, hidden=cfg.rm_hidden,
                                                    mlp_hidden=cfg.rm_mlp_hidden, kernel_size=cfg.rm_kernel_size,
                                                    layers=cfg.rm_layers, num_experts=cfg.rm_num_experts,
                                                    rank=cfg.rm_mole_rank, tau=cfg.rm_mole_tau),
    "preference_transformer": lambda cfg, obs: PreferenceTransformerRewardModel(
        obs_dim=obs, user_dim=cfg.gcf_emb_dim, hidden=cfg.rm_hidden, num_heads=cfg.rm_num_heads,
        num_layers=cfg.rm_layers, max_len=cfg.rm_max_len),
}


def load_path(cfg, run, comp):
    """컴포넌트 로드 경로. eval_only면 이 run 폴더, cfg.load[comp] 지정 시 해당 run, 아니면 None(학습)."""
    if run.eval_only:
        return run.dir
    ts = cfg.load[comp]
    return run.at(ts) if ts else None


def build_graph(cfg, run, train_data, device):
    print("=" * 60)
    print("[CoPL] Building graph dataset...")
    sim_path = load_path(cfg, run, "sim")
    ckpt = f"{cfg.similarity_method}.pt"
    if sim_path is not None:
        print(f"  [SIM:{cfg.similarity_method}] Loading from {sim_path}...")
        sim = build_similarity(cfg.similarity_method)
        sim.load(sim_path / ckpt, device=device)
        gds = CoPLGraphDataset(train_data, cfg, sim_builder=sim)
    else:
        gds = CoPLGraphDataset(train_data, cfg)
        gds.sim_builder.save(run.dir / ckpt)
    return gds.to(device)


def build_gcf(cfg, gds, device):
    common = dict(n_u=gds.n_users, n_i=gds.n_items, d=cfg.gcf_emb_dim,
                  pos_adj_norm=gds.Apos_norm, neg_adj_norm=gds.Aneg_norm,
                  dropout=cfg.gcf_dropout, l=cfg.gcf_layers,
                  item_item_adj_norm=gds.Aii_norm, item_item_weight=cfg.item_item_weight)
    if cfg.gcf_model == "gcf_gcn":
        from models.copl.gcf_gcn import CoPLGCF_PyG
        return CoPLGCF_PyG(**common).to(device)

    Z = torch.tensor(gds.Z_train, dtype=torch.float32)
    if Z.shape[1] != cfg.gcf_emb_dim:
        proj = torch.nn.Linear(Z.shape[1], cfg.gcf_emb_dim, bias=False)
        torch.nn.init.xavier_uniform_(proj.weight)
        with torch.no_grad():
            Z = proj(Z)
    return CoPLGCF(**common, loss_type=cfg.gcf_loss_type, loss_kwargs=cfg.gcf_loss_kwargs,
                   item_feat_init=Z, m_i_type=cfg.gcf_m_i_type).to(device)


def train(cfg, run, gds, gcf, rm, device):
    gcf_path = load_path(cfg, run, "gcf")
    if gcf_path is None:
        print("\n[1] Training GCF...")
        trainer = CoPLGCFTrainer(gcf, {
            "device": cfg.device, "gcf_lr": cfg.gcf_lr,
            "gcf_weight_decay": cfg.gcf_weight_decay, "gcf_lambda_reg": cfg.gcf_lambda_reg,
            "gcf_epochs": cfg.gcf_epochs, "use_pos_weight": cfg.use_pos_weight,
        }, log_dir=run.dir)
        gcf_auc, _, E_u, E_i, gcf_hist = trainer.train(
            gds.tr_u, gds.tr_i, gds.tr_y, gds.va_u, gds.va_i, gds.va_y, verbose=cfg.verbose)
        print(f"  GCF Best Val AUC: {gcf_auc:.4f}")
    else:
        print(f"\n[1] Loading GCF from {gcf_path}...")
        gcf.load_state_dict(torch.load(gcf_path / "best_gcf.pt", map_location=device, weights_only=True))
        gcf.eval()
        with torch.no_grad():
            E_u, E_i = gcf.encode_graph(test=True)
        gcf_auc, gcf_hist = float("nan"), {}
    E_u = E_u.to(device)

    train_loader = DataLoader(RMEdgeDataset(gds.tr_u, gds.tr_i, gds.tr_y, gds.item_series),
                              batch_size=cfg.rm_batch_size, shuffle=True, collate_fn=rm_collate)
    val_loader = DataLoader(RMEdgeDataset(gds.va_u, gds.va_i, gds.va_y, gds.item_series),
                            batch_size=cfg.rm_batch_size, shuffle=False, collate_fn=rm_collate)

    rm_path = load_path(cfg, run, "rm")
    if rm_path is None:
        print("\n[2] Training Reward Model...")
        trainer = CoPLRMTrainer(rm, {
            "device": cfg.device, "rm_lr": cfg.rm_lr,
            "rm_weight_decay": cfg.rm_weight_decay, "rm_lambda_reg": cfg.rm_lambda_reg,
            "rm_epochs": cfg.rm_epochs, "use_pos_weight": cfg.use_pos_weight,
        }, log_dir=run.dir)
        rm_auc, rm_hist = trainer.train(train_loader, val_loader, E_u, gds.tr_y, verbose=cfg.verbose)
        print(f"  RM Best Val AUC: {rm_auc:.4f}")
    else:
        print(f"\n[2] Loading Reward Model from {rm_path}...")
        rm.load_state_dict(torch.load(rm_path / "best_rm.pt", map_location=device, weights_only=True))
        rm_auc, rm_hist = float("nan"), {}

    hist = {}
    for src, tag in ((gcf_hist, "gcf"), (rm_hist, "rm")):
        for k, v in src.items():
            prefix, name = k.split("/", 1)
            hist[f"{prefix}/{tag}_{name}"] = v
    if hist:
        plot_training_curves(hist, run.plots / "training_curves.png", title="CoPL Training")
    run.metrics["train/val_auroc"] = {"gcf": gcf_auc, "rm": rm_auc}
    return E_u, E_i


def eval_driver(cfg, gds, rm, E_u, E_i, name, X, y, device):
    print(f"\n[3] Test-time evaluation on {name}...")
    X = gds.norm(X)
    (ctx_X, ctx_y), (hold_X, hold_y) = split_ctx(X, y)
    if len(ctx_y) < 1 or len(np.unique(hold_y)) < 2:
        print(f"  [Skip] {name}: holdout lacks both classes")
        return None

    hold_obs = torch.tensor(hold_X, dtype=torch.float32, device=device)
    pcts = {max(1, int(len(y) * p)): int(p * 100) for p in (0.1, 0.2, 0.3, 0.4, 0.5)}
    pcts = {t: p for t, p in pcts.items() if t <= len(ctx_y)}

    trk = Track(hold_y)
    d = {"name": name, "holdout_y": hold_y, "snapshots": {}, "wu_history": []}
    for t in grid(len(ctx_y)):
        _, neigh_idx, neigh_w = gds.attach_test_items(ctx_X[:t], E_i.cpu(), topk=cfg.adapt_topk, device=device)
        e_u, w_u = gds.adapt_test_user(ctx_y[:t], neigh_idx, neigh_w, E_u, device=device)
        d["wu_history"].append(w_u)
        with torch.no_grad():
            probs = torch.sigmoid(rm(e_u.unsqueeze(0).expand(len(hold_X), -1), hold_obs)).cpu().numpy()
        if trk.add(t, probs):
            d["best"] = dict(probs=probs, w_u=w_u, neigh_idx=neigh_idx, neigh_w=neigh_w,
                             ctx=t, auroc=trk.aurocs[-1])
        if t in pcts:
            print(f"  [Context {pcts[t]:>3}%] AUROC={trk.aurocs[-1]:.4f}")
            d["snapshots"][pcts[t]] = dict(probs=probs, w_u=w_u, neigh_idx=neigh_idx, neigh_w=neigh_w, ctx=t)
        d["e_u_final"], d["neigh_idx_last"], d["neigh_w_last"] = e_u, neigh_idx, neigh_w

    d["ctx_sizes"], d["aurocs"] = trk.ts, trk.aurocs
    d["best_pct"] = round(d["best"]["ctx"] / len(y) * 100)
    return d


def evaluate(cfg, run, gds, rm, E_u, E_i, test_data, device):
    rm.eval()
    plots = run.plots
    drivers = [d for d in (eval_driver(cfg, gds, rm, E_u, E_i, n, *test_data[n], device) for n in cfg.test) if d]
    if not drivers:
        return

    names = [d["name"] for d in drivers]
    ys = [d["holdout_y"] for d in drivers]

    for pct in sorted({p for d in drivers for p in d["snapshots"]}):
        ds_with = [d for d in drivers if pct in d["snapshots"]]
        wn = [d["name"] for d in ds_with]
        snap = plots / "snapshots" / f"context_{pct}pct"
        evaluate_predictions([d["holdout_y"] for d in ds_with],
                             [d["snapshots"][pct]["probs"] for d in ds_with],
                             snap, wn, save_name="metrics", title=f"CoPL (Context {pct}%)")
        viz.plot_attention_bars([d["snapshots"][pct]["w_u"] for d in ds_with], gds.train_drivers, wn,
                                snap / "user_attention.png", suffix=f" (Context {pct}%)")
        viz.plot_test_item_bridge([d["snapshots"][pct]["neigh_idx"] for d in ds_with],
                                  [d["snapshots"][pct]["neigh_w"] for d in ds_with],
                                  gds.item_owner_uid, gds.train_drivers, wn, snap / "item_bridge.png")

    plot_sequential_auroc([d["ctx_sizes"] for d in drivers], [d["aurocs"] for d in drivers],
                          plots, names, save_name="seq_auroc")

    ms = evaluate_predictions(ys, [d["best"]["probs"] for d in drivers], plots, names,
                              save_name="metrics_best", title="CoPL (best ctx per driver)")
    for d, m in zip(drivers, ms):
        run.metrics[f"test/{d['name']}"] = {**m, "best_ctx_pct": d["best_pct"]}
        print(f"  Test [{d['name']}] (best ctx={d['best_pct']}%) "
              f"AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}  Brier={m['brier']:.4f}")

    snap = plots / "snapshots" / "context_best"
    evaluate_predictions(ys, [d["best"]["probs"] for d in drivers], snap, names,
                         save_name="metrics", title="CoPL (best ctx)")
    viz.plot_attention_bars([d["best"]["w_u"] for d in drivers], gds.train_drivers, names,
                            snap / "user_attention.png", suffix=" (best ctx)")
    viz.plot_test_item_bridge([d["best"]["neigh_idx"] for d in drivers], [d["best"]["neigh_w"] for d in drivers],
                              gds.item_owner_uid, gds.train_drivers, names, snap / "item_bridge.png")
    viz.plot_test_item_bridge([d["neigh_idx_last"] for d in drivers], [d["neigh_w_last"] for d in drivers],
                              gds.item_owner_uid, gds.train_drivers, names, plots / "bridge.png")

    print("\n[4] Evaluating training drivers...")
    tr_ys, tr_ps, tr_names = [], [], []
    for uid, uname in enumerate(gds.train_drivers):
        iids = np.concatenate([gds.tr_i[gds.tr_u == uid], gds.va_i[gds.va_u == uid]])
        labels = np.concatenate([gds.tr_y[gds.tr_u == uid], gds.va_y[gds.va_u == uid]])
        obs = torch.tensor(gds.item_series[iids], dtype=torch.float32, device=device)
        emb = E_u[uid].unsqueeze(0).expand(len(iids), -1)
        with torch.no_grad():
            probs = torch.sigmoid(rm(emb, obs)).cpu().numpy()
        tr_ys.append(labels); tr_ps.append(probs); tr_names.append(uname)
    ms = evaluate_predictions(tr_ys, tr_ps, plots / "train", tr_names,
                              save_name="metrics", title="CoPL Train Drivers")
    for uname, m in zip(tr_names, ms):
        run.metrics[f"train/{uname}"] = m
        print(f"  {uname}: AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}  Brier={m['brier']:.4f}")

    item_labels = np.zeros(gds.n_items, dtype=np.int64)
    for uid, (item_ids, y) in gds.per_user_items.items():
        item_labels[item_ids] = y
    gds.sim_builder.visualize(plots / "similarity", gds.item_series, gds.item_owner_uid,
                              gds.train_drivers, cfg.view.cols, item_labels)

    E_i_cpu = E_i.detach().cpu().numpy()
    E_i_tests = [torch.tensor(np.stack([
        (d["neigh_w_last"][i][:, None] * E_i_cpu[d["neigh_idx_last"][i]]).sum(axis=0)
        for i in range(d["neigh_idx_last"].shape[0])])) for d in drivers]
    viz.plot_item_embeddings(E_i, gds.item_owner_uid, gds.train_drivers, plots / "embeddings",
                             E_i_tests=E_i_tests, test_driver_names=names)
    viz.plot_user_embeddings(E_u, gds.train_drivers, [d["e_u_final"] for d in drivers], names,
                             plots / "embeddings")
    viz.plot_rm_distributions(rm, E_u, gds.item_series, gds.item_owner_uid, item_labels,
                              gds.train_drivers, device, plots / "reward_model")
    viz.plot_wu_evolution([d["wu_history"] for d in drivers], [d["ctx_sizes"] for d in drivers],
                          gds.train_drivers, names, plots / "wu_evolution.png")


def main(cfg=None):
    cfg = cfg or Config()
    run = Run("copl", cfg)
    device = torch.device(cfg.device)
    ds = Dataset("datasets")

    train_data = {n: cfg.view(ds[n]) for n in cfg.train}
    test_data = {n: cfg.view(ds[n]) for n in cfg.test}

    gds = build_graph(cfg, run, train_data, device)
    gcf = build_gcf(cfg, gds, device)
    rm = RM_MODELS[cfg.rm_model](cfg, gds.obs_dim).to(device)

    E_u, E_i = train(cfg, run, gds, gcf, rm, device)
    evaluate(cfg, run, gds, rm, E_u, E_i, test_data, device)
    run.finish()


if __name__ == "__main__":
    main()
