from dataclasses import dataclass, field

import torch

from src_new.utils.seed import seed_all
from src_new.utils.paths import artifact_dir
from src_new.model.vpl.experiment_binary import VPLBinaryExperiment


@dataclass
class Config:
    train_driver_names: list = field(default_factory=lambda: [
        "김진명", "김태근", "조현석", "한규택", "박재일", "이지환"
    ])
    test_driver_names: list = field(default_factory=lambda: ["강신길"])
    features: list = field(default_factory=lambda: [
        "IMU_VerAccelVal", "Pitch_rate_6D", "Bounce_rate_6D", "IMU_LongAccelVal"
    ])
    time_range: tuple = (5, 7)
    downsample: int = 5
    smooth: bool = True
    smooth_cutoff: float = 12.0
    smooth_order: int = 2
    context_size: int = 8
    normalize: bool = False
    balanced: bool = True
    val_size: float = 0.1
    timestamp: str = "test"

    latent_dim: int = 8
    hidden_dim: int = 128
    kl_weight: float = 0.05
    flow_prior: bool = True
    reward_scaling: object = "T"

    lr: float = 0.001
    weight_decay: float = 0.0
    n_epochs: int = 500
    batch_size: int = 128
    eval_freq: int = 10
    early_stop: bool = False
    patience: int = 20
    min_delta: float = 0.0

    anneal_shape: str = "cosine"
    anneal_cycles: int = 4

    device: str = "cuda"
    seed: int = 42
    verbose: int = 1


def main():
    cfg = Config()
    seed_all(cfg.seed)
    out_dir, eval_only = artifact_dir("vpl_binary", timestamp=cfg.timestamp)
    VPLBinaryExperiment(cfg).run(out_dir, eval_only=eval_only)
    print(f"[Done]  →  {out_dir}")


if __name__ == "__main__":
    main()
