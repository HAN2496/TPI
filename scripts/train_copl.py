# scripts/train_copl.py
import json
import wandb
import optuna
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from dataclasses import dataclass

import torch

from sklearn.metrics import roc_auc_score, roc_curve

from src.utils.utils import seed_all
from src.model.CoPL_new.utils import CoPLGraphDataset
from src.model.CoPL_new.gcf import CoPLGCF, CoPLGCFCosine, CoPLGCFPointwiseBPR, CoPLGCFSoftmax, CoPLGCFMargin
from src.model.CoPL_new.gcf_gcn import CoPLGCF_GCN
from src.model.CoPL_new.rm import RewardModel, CNNRewardModel, weighted_bce_logits, rm_collate, RMEdgeDataset
from src.model.CoPL_new.visualization import plot_distance_gamma_analysis, plot_driver_similarity_matrix, plot_roc, plot_reward_scatter, compare_viz_plot, plot_test_item_bridge

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# =========================
# Config
# =========================
@dataclass
class CFG:
    model_type: str = "gcf_margin"  # "gcf" or "gcf_gcn" or "gcf_cosine" or "gcf_pointwise_bpr" or "gcf_softmax" or "gcf_margin"

    timestamp = None # None for timestamp training, test for debug, else load from string

    # data
    features: tuple = ("IMU_VerAccelVal", "Bounce_rate_6D", "Pitch_rate_6D", "IMU_LongAccelVal")
    train_driver_names: tuple = ("김진명", "김태근", "조현석", "한규택", "박재일", "이지환")
    test_driver_name: str = "강신길"
    time_range: tuple = (5, 7)
    downsample: int = 5
    context_size: int = 16  # context size for chunk-based visualization

    # item-item graph
    similarity_method: str = "vae"  # "pca" or "vae" or "kernel_pca" or "dtw"
    gamma_mul: float = 8.0              # gamma = median_gamma * gamma_mul
    knn_k: int = 5
    mutual: bool = False

    # PCA similarity (only used when similarity_method="pca")
    pca_dim: int = 2

    # VAE similarity (only used when similarity_method="vae")
    vae_latent_dim: int = 16
    vae_epochs: int = 100
    vae_lr: float = 0.001
    vae_kl_weight: float = 0.1          # β-VAE weight
    vae_batch_size: int = 128
    vae_hidden_channels: int = 32
    vae_metric: str = "cosine"        # "euclidean" or "cosine"

    # GCF
    hidden_dim: int = 128
    gcf_layers: int = 2
    gcf_dropout: float = 0.3
    item_item_weight: float = 0.72
    gcf_lr: float = 0.00068
    gcf_weight_decay: float = 0.001
    gcf_epochs: int = 100
    gcf_lambda_reg: float = 0.0

    # Loss weights
    use_pos_weight: bool = True

    # Other GCF Hyperparams
    margin: float = 0.5  # for pointwise BPR
    temperature: float = 0.1 # for softmax

    # RM
    rm_model_type: str = "cnn"  # "mlp" or "cnn"
    rm_hidden: int = 32
    rm_mlp_hidden: int = 64
    rm_lr: float = 0.00026
    rm_weight_decay: float = 0.0
    rm_epochs: int = 200
    rm_batch_size: int = 256
    rm_lambda_reg: float = 1e-6

    # WandB
    wandb_project: str = "TPI-CoPL"
    wandb_entity: str = None  # None -> use default user
    wandb_mode: str = "online"  # "online", "offline", "disabled"

    # adaptation for test user
    adapt_use_neg: bool = True
    adapt_neg_weight: float = 0.81
    adapt_user_softmax_temp: float = 1.15
    attach_topk_items: int = 20  # test item -> train items topk for embedding

    # split
    val_size: float = 0.1
    seed: int = 42

    # viz
    tsne_max_items_per_driver: int = 400
    tsne_perplexity: int = 30

    # system
    device: str = "cuda"
    save_root: str = "artifacts/copl"
    verbose: int = 1




# =========================
# Training: GCF
# =========================
def train_gcf(cfg, gcf, tr_u, tr_i, tr_y, va_u, va_i, va_y,
              device, log_dir, is_training, trial=None):
    """
    GCF 모델 학습 루프.
    Returns: (best_auc, best_val_loss, E_u_train, E_i_train)
    """
    tr_u_t = torch.tensor(tr_u, dtype=torch.long, device=device)
    tr_i_t = torch.tensor(tr_i, dtype=torch.long, device=device)
    tr_y_t = torch.tensor(tr_y, dtype=torch.float32, device=device)
    va_u_t = torch.tensor(va_u, dtype=torch.long, device=device)
    va_i_t = torch.tensor(va_i, dtype=torch.long, device=device)
    va_y_np = va_y.astype(np.int64)

    best_auc = -1.0
    best_gcf_state_dict = None
    best_val_loss = float("inf")

    if is_training:
        opt_gcf = torch.optim.AdamW(gcf.parameters(), lr=cfg.gcf_lr, weight_decay=cfg.gcf_weight_decay)

        pos_cnt = int(tr_y.sum())
        neg_cnt = int((1 - tr_y).sum())
        if cfg.use_pos_weight:
            pos_weight = torch.tensor([neg_cnt / max(1, pos_cnt)], dtype=torch.float32, device=device)
        else:
            pos_weight = None

        for epoch in range(cfg.gcf_epochs):
            gcf.train()
            opt_gcf.zero_grad()
            loss, _ = gcf.forward_pointwise(
                tr_u_t, tr_i_t, tr_y_t,
                pos_weight=pos_weight,
                sample_weight=None,
                test=False,
                lambda_reg=cfg.gcf_lambda_reg,
            )
            loss.backward()
            opt_gcf.step()

            gcf.eval()
            with torch.no_grad():
                val_loss, val_logits = gcf.forward_pointwise(
                    va_u_t, va_i_t, torch.tensor(va_y_np, dtype=torch.float32, device=device),
                    pos_weight=None,
                    sample_weight=None,
                    test=True,
                    lambda_reg=0.0,
                )
                val_prob = torch.sigmoid(val_logits).detach().cpu().numpy()
                val_auc = roc_auc_score(va_y_np, val_prob) if len(np.unique(va_y_np)) > 1 else 0.0

            if val_auc > best_auc:
                best_auc = val_auc
            #     best_gcf_state_dict = {k: v.cpu() for k, v in gcf.state_dict().items()}

            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                best_gcf_state_dict = {k: v.cpu() for k, v in gcf.state_dict().items()}

            if cfg.verbose > 0 and (epoch % 10 == 0 or epoch == cfg.gcf_epochs - 1):
                print(f"  [GCF] epoch={epoch:03d} loss={float(loss.item()):.4f} val_auc={val_auc:.4f} best={best_auc:.4f}")

            if is_training:
                wandb.log({
                    "GCF/loss": loss.item(),
                    "GCF/val_auc": val_auc,
                    "GCF/epoch": epoch
                })

            # Optuna Pruning
            if trial is not None:
                trial.report(val_auc, step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        if cfg.verbose > 0:
            print("Best GCF val AUC:", best_auc)
            if best_gcf_state_dict is not None:
                torch.save(best_gcf_state_dict, log_dir / "best_gcf.pt")
                print("Saved:", log_dir / "best_gcf.pt")

        # load best & extract embeddings
        if best_gcf_state_dict is not None:
            gcf.load_state_dict({k: v.to(device) for k, v in best_gcf_state_dict.items()})
    else:
        # Inference: Load
        p = log_dir / "best_gcf.pt"
        if p.exists():
            print(f"Loading GCF model from {p}")
            st = torch.load(p, map_location=device)
            gcf.load_state_dict(st)
        else:
            print(f"Warning: {p} not found. Using random init.")
    
    gcf.eval()
    with torch.no_grad():
        E_u_train, E_i_train = gcf.encode_graph(test=True)  # (n_users,d), (n_items,d)

    return best_auc, best_val_loss, E_u_train, E_i_train


# =========================
# Training: RM
# =========================
def train_rm(cfg, rm, E_u_train, tr_u, tr_i, tr_y, va_u, va_i, va_y,
             item_series, device, log_dir, is_training, trial=None):
    """
    Reward Model 학습 루프.
    Returns: best_rm_auc
    """
    best_rm_auc = -1.0
    best_rm_state_dict = None

    if is_training:
        opt_rm = torch.optim.AdamW(rm.parameters(), lr=cfg.rm_lr, weight_decay=cfg.rm_weight_decay)

        # RM datasets use same split (tr_u,tr_i,tr_y) and (va_u,va_i,va_y)
        tr_ds = RMEdgeDataset(tr_u, tr_i, tr_y, item_series)
        va_ds = RMEdgeDataset(va_u, va_i, va_y, item_series)

        tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=cfg.rm_batch_size, shuffle=True, collate_fn=rm_collate, drop_last=True)
        va_loader = torch.utils.data.DataLoader(va_ds, batch_size=cfg.rm_batch_size, shuffle=False, collate_fn=rm_collate, drop_last=False)

        pos_cnt_rm = int(tr_y.sum())
        neg_cnt_rm = int((1 - tr_y).sum())
        pos_weight_rm = None
        if cfg.use_pos_weight:
            pos_weight_rm = torch.tensor([neg_cnt_rm / max(1, pos_cnt_rm)], dtype=torch.float32, device=device)

        for epoch in range(cfg.rm_epochs):
            rm.train()
            tr_loss_accum = 0.0
            n_seen = 0

            for uids_b, obs_b, y_b in tr_loader:
                uids_b = uids_b.to(device)
                obs_b = obs_b.to(device)
                y_b = y_b.to(device)

                user_emb = E_u_train[uids_b]  # fixed user embedding

                logits = rm(user_emb, obs_b)
                loss_bce = weighted_bce_logits(logits, y_b, pos_weight=pos_weight_rm)
                loss_reg = (user_emb.norm(2).pow(2).mean())  # small reg (optional)
                loss = loss_bce + cfg.rm_lambda_reg * loss_reg

                opt_rm.zero_grad()
                loss.backward()
                opt_rm.step()

                tr_loss_accum += float(loss.item()) * len(uids_b)
                n_seen += len(uids_b)

            rm.eval()
            all_prob = []
            all_y = []
            with torch.no_grad():
                for uids_b, obs_b, y_b in va_loader:
                    uids_b = uids_b.to(device)
                    obs_b = obs_b.to(device)
                    y_b = y_b.to(device)
                    user_emb = E_u_train[uids_b]
                    logits = rm(user_emb, obs_b)
                    prob = torch.sigmoid(logits).detach().cpu().numpy()
                    all_prob.append(prob)
                    all_y.append(y_b.detach().cpu().numpy())

            all_prob = np.concatenate(all_prob) if all_prob else np.array([])
            all_y = np.concatenate(all_y).astype(np.int64) if all_y else np.array([])
            val_auc = roc_auc_score(all_y, all_prob) if len(np.unique(all_y)) > 1 else 0.0

            if val_auc > best_rm_auc:
                best_rm_auc = val_auc
                best_rm_state_dict = {k: v.cpu() for k, v in rm.state_dict().items()}

            if cfg.verbose > 0 and (epoch % 5 == 0 or epoch == cfg.rm_epochs - 1):
                print(f"  [RM ] epoch={epoch:03d} train_loss={tr_loss_accum/max(1,n_seen):.4f} val_auc={val_auc:.4f} best={best_rm_auc:.4f}")
            
            if is_training:
                wandb.log({
                    "RM/train_loss": tr_loss_accum/max(1,n_seen),
                    "RM/val_auc": val_auc,
                    "RM/epoch": epoch
                })

            # Optuna Pruning
            if trial is not None:
                current_step = cfg.gcf_epochs + epoch
                trial.report(val_auc, step=current_step)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        if cfg.verbose > 0:
            print("Best RM val AUC:", best_rm_auc)
            if best_rm_state_dict is not None:
                torch.save(best_rm_state_dict, log_dir / "best_rm.pt")
                print("Saved:", log_dir / "best_rm.pt")

        if best_rm_state_dict is not None:
            rm.load_state_dict({k: v.to(device) for k, v in best_rm_state_dict.items()})
    else:
        # Inference: Load
        p = log_dir / "best_rm.pt"
        if p.exists():
            print(f"Loading RM model from {p}")
            st = torch.load(p, map_location=device)
            rm.load_state_dict(st)
        else:
            print(f"Warning: {p} not found. Using random init.")
    rm.eval()

    return best_rm_auc


# =========================
# Main
# =========================
def run_copl_training(cfg: CFG, trial: optuna.Trial = None):
    seed_all(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Determine mode & log_dir
    is_training = True
    log_dir = None

    if cfg.timestamp is not None:
        if cfg.timestamp == "test":
            log_dir = Path(cfg.save_root) / "test"
        else:
            log_dir = Path(cfg.save_root) / cfg.timestamp
            is_training = False
            print(f"Inference Mode: Loading model from {log_dir}")
    else:
        if cfg.verbose > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = Path(cfg.save_root) / timestamp

    # WandB Init
    if is_training:
        run_name = f"{cfg.model_type}-{cfg.test_driver_name}"
        if cfg.timestamp: # If timestamp is explicitly set for training, use it in run name
            run_name = f"{cfg.timestamp}-{run_name}"
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            config=cfg.__dict__,
            name=run_name,
            mode=cfg.wandb_mode,
            dir=log_dir # Save wandb files in the log directory
        )
        wandb.run.log_code(".") # Log current code

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)

        # Inference: Load saved config to match model architecture
        if not is_training:
            cfg_path = log_dir / "cfg.json"
            if cfg_path.exists():
                print(f"Loading config from {cfg_path}...")
                with open(cfg_path, "r", encoding="utf-8") as f:
                    saved_cfg = json.load(f)
                for k, v in saved_cfg.items():
                    # Don't overwrite the timestamp (which controls the mode)
                    if k == "timestamp":
                        continue
                    if hasattr(cfg, k):
                        setattr(cfg, k, v)
            else:
                print("Warning: cfg.json not found. Using current script config.")

        if is_training and cfg.verbose > 0:
            with open(log_dir / "cfg.json", "w", encoding="utf-8") as f:
                json.dump(cfg.__dict__, f, ensure_ascii=False, indent=2)
        print("Log dir:", log_dir)

    # -------------------------
    # 1-4) Build dataset (load drivers, split, build graphs)
    # -------------------------
    if cfg.verbose > 0:
        print("\n[1] Loading train drivers & building graphs...")

    dataset = CoPLGraphDataset(cfg)
    dataset.to(device)

    # Convenience aliases (minimize downstream code changes)
    train_drivers = dataset.train_drivers
    test_driver = cfg.test_driver_name
    n_users, n_items = dataset.n_users, dataset.n_items
    obs_dim = dataset.obs_dim
    item_series = dataset.item_series
    item_owner_uid = dataset.item_owner_uid
    per_user_items = dataset.per_user_items
    tr_u, tr_i, tr_y = dataset.tr_u, dataset.tr_i, dataset.tr_y
    va_u, va_i, va_y = dataset.va_u, dataset.va_i, dataset.va_y
    Apos_norm, Aneg_norm = dataset.Apos_norm, dataset.Aneg_norm
    Aii_norm = dataset.Aii_norm

    if cfg.verbose > 0 and log_dir is not None:
        with open(log_dir / "Aii_meta.json", "w", encoding="utf-8") as f:
            json.dump(dataset.Aii_meta, f, ensure_ascii=False, indent=2)
    if is_training:
        wandb.log({"Aii_meta": dataset.Aii_meta})

    # -------------------------
    # 5) Train GCF (pointwise BCE, imbalance via pos_weight)
    # -------------------------
    if cfg.verbose > 0:
        print("\n[3] Training GCF (pointwise BCE)...")
    
    GCF_REGISTRY = {
        "gcf":               CoPLGCF,
        "gcf_cosine":        CoPLGCFCosine,
        "gcf_pointwise_bpr": CoPLGCFPointwiseBPR,
        "gcf_softmax":       CoPLGCFSoftmax,
        "gcf_gcn":           CoPLGCF_GCN,
        "gcf_margin":        CoPLGCFMargin,
    }
    common_kwargs = dict(
        n_u=n_users, n_i=n_items, d=cfg.hidden_dim,
        pos_adj_norm=Apos_norm, neg_adj_norm=Aneg_norm,
        dropout=cfg.gcf_dropout, l=cfg.gcf_layers,
        item_item_adj_norm=Aii_norm, item_item_weight=cfg.item_item_weight,
    )
    extra_kwargs = {}
    if cfg.model_type == "gcf_cosine" or cfg.model_type == "gcf_margin":
        extra_kwargs["margin"] = cfg.margin
    elif cfg.model_type == "gcf_softmax":
        extra_kwargs["temperature"] = cfg.temperature

    gcf_cls = GCF_REGISTRY.get(cfg.model_type)
    if gcf_cls is None:
        raise NotImplementedError(f"Unknown model_type: {cfg.model_type}")
    gcf = gcf_cls(**common_kwargs, **extra_kwargs).to(device)

    best_auc, best_val_loss, E_u_train, E_i_train = train_gcf(
        cfg=cfg, gcf=gcf,
        tr_u=tr_u, tr_i=tr_i, tr_y=tr_y,
        va_u=va_u, va_i=va_i, va_y=va_y,
        device=device, log_dir=log_dir,
        is_training=is_training, trial=trial,
    )

    # -------------------------
    # 6) Train Reward Model (RM) using fixed user embeddings
    # -------------------------
    if cfg.verbose > 0:
        print("\n[4] Training RM (time-series, pointwise BCE)...")
    
    if cfg.rm_model_type == "mlp":
        rm = RewardModel(
            obs_dim=obs_dim, 
            user_dim=E_u_train.shape[1], 
            hidden=cfg.rm_hidden, 
            mlp_hidden=cfg.rm_mlp_hidden
        ).to(device)
    elif cfg.rm_model_type == "cnn":
        rm = CNNRewardModel(
            obs_dim=obs_dim, 
            user_dim=E_u_train.shape[1], 
            hidden=cfg.rm_hidden, 
            mlp_hidden=cfg.rm_mlp_hidden,
            kernel_size=3,
            layers=2
        ).to(device)
    else:
        raise NotImplementedError
    best_rm_auc = train_rm(
        cfg=cfg, rm=rm, E_u_train=E_u_train,
        tr_u=tr_u, tr_i=tr_i, tr_y=tr_y,
        va_u=va_u, va_i=va_i, va_y=va_y,
        item_series=item_series,
        device=device, log_dir=log_dir,
        is_training=is_training, trial=trial,
    )

    # -------------------------
    # 7) Test driver evaluation (AUROC + plots) using adaptation
    # -------------------------
    if cfg.verbose > 0:
        print("\n[5] Evaluating on test driver:", test_driver)
    
    X_test, y_test = dataset.load_test_driver(test_driver)

    # attach test items into train item embedding space
    E_i_test, neigh_idx, neigh_w = dataset.attach_test_items(
        X_test=X_test.astype(np.float32),
        E_i_train=E_i_train,
        topk=cfg.attach_topk_items,
        device=device,
    )

    # adapt test user embedding using item-item bridge + train pos edges
    e_u_test, w_u = dataset.adapt_test_user(
        y_test=y_test,
        neigh_idx=neigh_idx,
        neigh_w=neigh_w,
        E_u_train=E_u_train,
        device=device,
    )

    # -------------------------
    # Evaluation: GCF (Dot Product)
    # -------------------------
    with torch.no_grad():
        # scores = (test items) dot (test user)
        # E_i_test: (n_test, d), e_u_test: (d,)
        logits_gcf = (E_i_test * e_u_test.unsqueeze(0)).sum(dim=1)
        prob_gcf = torch.sigmoid(logits_gcf).detach().cpu().numpy()
    
    test_auc_gcf = roc_auc_score(y_test, prob_gcf) if len(np.unique(y_test)) > 1 else 0.0

    # RM prediction on test
    with torch.no_grad():
        user_emb_test = e_u_test.unsqueeze(0).repeat(X_test.shape[0], 1)  # (n_test, d)
        obs_test = torch.tensor(X_test.astype(np.float32), dtype=torch.float32, device=device)
        logits_test = rm(user_emb_test, obs_test)
        prob_test = torch.sigmoid(logits_test).detach().cpu().numpy()

    test_auc = roc_auc_score(y_test, prob_test) if len(np.unique(y_test)) > 1 else 0.0
    
    if cfg.verbose > 0:
        print(f"Test AUROC (GCF): {test_auc_gcf:.4f}")
        print(f"Test AUROC (RM): {test_auc:.4f}")

        # Filename with AUC
        roc_test_filename = f"roc_test_{test_driver}_auc{test_auc:.4f}.png"
        scatter_test_filename = f"scatter_test_{test_driver}_auc{test_auc:.4f}.png"

        plot_roc(
            y_true=y_test,
            y_score=prob_test,
            save_path=log_dir / roc_test_filename,
            title=f"ROC (RM) - Test Driver {test_driver} (AUC={test_auc:.4f})"
        )

        plot_reward_scatter(
            y_true=y_test,
            y_prob=prob_test,
            save_path=log_dir / scatter_test_filename,
            title=f"RM Prediction Scatter - Test Driver {test_driver}"
        )

        if is_training:
            # Log metrics
            wandb.log({"Test/AUROC": test_auc})
            wandb.log({"Test/AUROC_GCF": test_auc_gcf}) # Log GCF AUC too
            
            # Log plots
            roc_img_path = log_dir / roc_test_filename
            scatter_img_path = log_dir / scatter_test_filename
            
            if roc_img_path.exists():
                wandb.log({"Test/ROC_Curve": wandb.Image(str(roc_img_path))})
            if scatter_img_path.exists():
                wandb.log({"Test/Reward_Scatter": wandb.Image(str(scatter_img_path))})

    # -------------------------
    # 8) t-SNE visualization (users + items)
    # -------------------------
    if cfg.verbose > 0:
        print("\n[6] t-SNE visualization...")

        # users: train users + test user (star)
        user_emb_all = torch.cat([E_u_train, e_u_test.unsqueeze(0)], dim=0).detach().cpu().numpy()
        user_labels = train_drivers + [f"{test_driver} (Test)"]
        star_mask = np.array([False] * len(train_drivers) + [True])

        compare_viz_plot(
            cfg=cfg,
            emb=user_emb_all,
            labels=user_labels,
            save_path=log_dir / "tsne_users.png",
            title="t-SNE: User Embeddings (Train + Test)",
            star_mask=star_mask,
        )
        if is_training:
            wandb.log({"Viz/tSNE_Users": wandb.Image(str(log_dir / "tsne_users.png"))})

        # items: sample train items per driver + all test items (or sample)
        rng = np.random.default_rng(cfg.seed)
        train_item_idx = []
        train_item_labels = []
        for uid, uname in enumerate(train_drivers):
            ids = np.where(item_owner_uid == uid)[0]
            if len(ids) == 0:
                continue
            take = min(cfg.tsne_max_items_per_driver, len(ids))
            picked = rng.choice(ids, size=take, replace=False)
            train_item_idx.append(picked)
            train_item_labels += [f"{uname} (Train)"] * take

        train_item_idx = np.concatenate(train_item_idx) if train_item_idx else np.array([], dtype=np.int64)
        E_i_train_sample = E_i_train[torch.tensor(train_item_idx, dtype=torch.long, device=device)].detach().cpu().numpy()

        # test items: 샘플링(너무 많으면)
        n_test_items = X_test.shape[0]
        take_test = min(cfg.tsne_max_items_per_driver, n_test_items)
        test_pick = rng.choice(np.arange(n_test_items), size=take_test, replace=False) if n_test_items > take_test else np.arange(n_test_items)
        E_i_test_sample = E_i_test[torch.tensor(test_pick, dtype=torch.long, device=device)].detach().cpu().numpy()
        test_item_labels = [f"{test_driver} (Test)"] * len(test_pick)

        item_emb_all = np.concatenate([E_i_train_sample, E_i_test_sample], axis=0)
        item_labels_all = train_item_labels + test_item_labels

        star_mask_items = np.array([False] * len(train_item_labels) + [True] * len(test_item_labels))

        compare_viz_plot(
            cfg=cfg,
            emb=item_emb_all,
            labels=item_labels_all,
            save_path=log_dir / "tsne_items.png",
            title="t-SNE: Item Embeddings (Train Items + Attached Test Items)",
            star_mask=star_mask_items,
        )
        if is_training:
            wandb.log({"Viz/tSNE_Items": wandb.Image(str(log_dir / "tsne_items.png"))})

        # -------------------------
        if cfg.context_size > 0:
            print("\n[7] t-SNE visualization (Context-based)...")
            chunk_embs = []
            chunk_labels = []

            def process_chunks_for_viz(X_full, y_full, label_name):
                # 1. Attach items to train (get weights)
                _, neigh_idx_all, neigh_w_all = dataset.attach_test_items(
                    X_test=X_full,
                    E_i_train=E_i_train,
                    topk=cfg.attach_topk_items,
                    device=device
                )
                
                # Shuffle to get random chunks
                n_samples = len(X_full)
                rng = np.random.default_rng(cfg.seed)
                perm = rng.permutation(n_samples)
                
                y_shuff = y_full[perm]
                neigh_idx_shuff = neigh_idx_all[perm]
                neigh_w_shuff = neigh_w_all[perm]
                
                num_chunks = n_samples // cfg.context_size
                for i in range(num_chunks):
                    st = i * cfg.context_size
                    ed = st + cfg.context_size
                    
                    # Adapt: chunk -> user embedding
                    e_u_chunk, _ = dataset.adapt_test_user(
                        y_test=y_shuff[st:ed],
                        neigh_idx=neigh_idx_shuff[st:ed],
                        neigh_w=neigh_w_shuff[st:ed],
                        E_u_train=E_u_train,
                        device=device
                    )
                    chunk_embs.append(e_u_chunk.detach().cpu().numpy())
                    chunk_labels.append(label_name)

            # Train users
            for uid, uname in enumerate(train_drivers):
                item_ids, y_u = per_user_items[uid]
                X_u = item_series[item_ids]
                process_chunks_for_viz(X_u.astype(np.float32), y_u, f"{uname} (Train)")
                
            # Test user
            process_chunks_for_viz(X_test.astype(np.float32), y_test, f"{test_driver} (Test)")
            
            if chunk_embs:
                chunk_embs = np.stack(chunk_embs)
                is_test_chunk = np.array(["(Test)" in l for l in chunk_labels])
                
                compare_viz_plot(
                    cfg=cfg,
                    emb=chunk_embs,
                    labels=chunk_labels,
                    save_path=log_dir / "tsne_users_chunks.png",
                    title=f"t-SNE: User Embeddings by Context (size={cfg.context_size})",
                    star_mask=is_test_chunk
                )
                if is_training:
                    wandb.log({"Viz/tSNE_User_Chunks": wandb.Image(str(log_dir / "tsne_users_chunks.png"))})


    # -------------------------
    # 9) Deep Analysis (Visualization)
    # -------------------------
    if cfg.verbose > 0:
        print("\n[8] Deep analysis visualization...")
        
        # 1. Distance & Gamma Analysis
        plot_distance_gamma_analysis(
            dataset.Z_train, dataset.gamma, 
            save_path=log_dir / "analysis_distance_gamma.png"
        )
        if is_training:
            wandb.log({"Analysis/DistanceGamma": wandb.Image(str(log_dir / "analysis_distance_gamma.png"))})

        # 2. Driver Similarity Matrix (based on Item-Item graph connectivity)
        plot_driver_similarity_matrix(
            Aii_norm=Aii_norm, 
            item_owner_uid=item_owner_uid, 
            train_drivers=train_drivers, 
            save_path=log_dir / "analysis_driver_sim_matrix.png"
        )
        if is_training:
            wandb.log({"Analysis/DriverSimMatrix": wandb.Image(str(log_dir / "analysis_driver_sim_matrix.png"))})

        # 3. 테스트 드라이버가 누구로부터 임베딩을 빌려왔는가?
        plot_test_item_bridge(
            neigh_idx=neigh_idx, 
            neigh_w=neigh_w, 
            item_owner_uid=item_owner_uid, 
            train_drivers=train_drivers, 
            save_path=log_dir / "analysis_test_bridge_weights.png"
        )
        if is_training:
            wandb.log({"Analysis/TestBridgeWeights": wandb.Image(str(log_dir / "analysis_test_bridge_weights.png"))})

        print(f"Deep analysis plots saved to {log_dir}")
        # -------------------------
        # Save summary
        # -------------------------
        summary = {
            "gcf_best_val_auc": float(best_auc),
            "rm_best_val_auc": float(best_rm_auc),
            "rm_test_auc": float(test_auc),
            "Aii_meta": dataset.Aii_meta,
            "test_user_weight_top5": sorted(
                [(train_drivers[i], float(w_u[i])) for i in range(len(train_drivers))],
                key=lambda x: -x[1]
            )[:5],
        }
        with open(log_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print("\nSaved artifacts to:", log_dir)
        print("Summary:", summary)

    if is_training:
        wandb.finish()

    return test_auc


if __name__ == "__main__":
    cfg = CFG()
    run_copl_training(cfg)
