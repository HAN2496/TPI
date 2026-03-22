# scripts/train/copl.py
import json
import wandb
import optuna
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class Dataset_Config:
    features: tuple = ("IMU_VerAccelVal", "Pitch_rate_6D", "Bounce_rate_6D", "IMU_LongAccelVal")
    train_driver_names: tuple = ("김진명", "김태근", "조현석", "한규택", "박재일", "이지환")
    test_driver_name: str = "강신길"
    time_range: tuple = (5, 7)
    downsample: int = 5

    val_size: float = 0.2

@dataclass
class Model_Config:
    model_type: str = "gcf"  # "gcf" or "gcf_gcn"

    timestamp = "test"

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
    rm_model_type: str = "mole_cnn"  # "mlp" or "cnn" or "mole_cnn"
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
    wandb_mode: str = "disabled"  # "online", "offline", "disabled"

    # adaptation for test user
    adapt_use_neg: bool = True
    adapt_neg_weight: float = 0.81
    adapt_user_softmax_temp: float = 1.15
    attach_topk_items: int = 20  # test item -> train items topk for embedding

    # split
    seed: int = 42

    # viz
    tsne_max_items_per_driver: int = 400
    tsne_perplexity: int = 30

    # system
    device: str = "cuda"
    save_root: str = "artifacts/copl"
    verbose: int = 1