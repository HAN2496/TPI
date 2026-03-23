# scripts/train_copl_new.py
import json
import wandb
import optuna
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, roc_curve

from src.model.CoPL.utils import CoPLGraphDataset

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# =========================
# Config
# =========================

@dataclass
class Dataset_Config:
    features: tuple = ("IMU_VerAccelVal", "Pitch_rate_6D", "Bounce_rate_6D", "IMU_LongAccelVal")
    train_driver_names: tuple = ("김진명", "김태근", "조현석", "한규택", "박재일", "이지환")
    test_driver_name: str = "강신길"
    time_range: tuple = (5, 7)
    downsample: int = 5

    val_size: float = 0.2

@dataclass
class GCF_Config:
    model_type: str = "gcf"  # "gcf" or "gcf_gcn"
    hidden_dim: int = 64
    n_layers: int = 2
    temperature: float = 0.1 # for softmax

@dataclass
class RM_Config:
    model_type: str = "mlp"  # "mlp" or "cnn"
    hidden_dim: int = 32
    mlp_hidden: int = 64
    lr: float = 0.00026
    weight_decay: float = 0.0
    epochs: int = 200
    batch_size: int = 256
    lambda_reg: float = 0.001

@dataclass
class Config:
    dataset: Dataset_Config = Dataset_Config()
    gcf: GCF_Config = GCF_Config()
    rm: RM_Config = RM_Config()
