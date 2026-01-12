import torch

from src.model.vpl_new.vae import VAEModel
from src.model.vpl_new.trainer import VPLTrainer

from src.utils.data_loader import DatasetManager
from src.utils.logger import ExperimentLogger
from src.utils.vpl_dataset import create_vpl_dataset, VPLDataset

FLAGS = {
    # data
    "features": ["IMU_VerAccelVal", "Bounce_rate_6D", "Pitch_rate_6D", "IMU_LongAccelVal"],
    "test_driver_name": "강신길",
    "time_range": (5, 7),
    "downsample": 5,
    "tie_ratio": 0.0,

    # model
    "hidden_dim": 256,
    "bath_size": 256,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "early_stop": False,
    "patience": 10,
    "min_delta": 3e-4,

    "latent_dim": 32,
    "kl_weight": 1.0,
    "flow_prior": False,
    "use_annealing": False,
    "annealer_baseline": 0.0,
    "annealer_type": "cosine",
    "annealer_cycles":4,

    # Training
    "n_epochs": 500,
    "eval_freq": 50,
    "save_freq": 50,

    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

def main():
    # Create dataset

    logger = ExperimentLogger(
        log_dir="artifacts/vpl_pairwise_new/",
        experiment_name="vpl_pairwise_new_experiment",
        add_timestamp=True
    )