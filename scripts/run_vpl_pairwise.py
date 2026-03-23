from dataclasses import dataclass, field

import torch

from src.utils.seed import seed_all
from src.utils.paths import artifact_dir
from src.model.vpl.experiment_pairwise import VPLPairwiseExperiment


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
    smooth: bool = False
    smooth_cutoff: float = 12.0
    smooth_order: int = 2
    context_size: int = 64
    tie_ratio: float = 0.0
    normalize: bool = False
    val_size: float = 0.1
    timestamp: str = None  # None=새 실험, "test"=디버그 고정폴더, "20250101_120000"=평가만

    latent_dim: int = 8
    hidden_dim: int = 128
    kl_weight: float = 0.1
    flow_prior: bool = False
    reward_scaling: object = "T"

    lr: float = 1e-3
    weight_decay: float = 1e-4
    n_epochs: int = 300
    batch_size: int = 32
    eval_freq: int = 10
    early_stop: bool = False
    patience: int = 20
    min_delta: float = 0.0

    anneal_shape: str = "linear"
    anneal_cycles: int = 4

    device: str = "cuda"
    seed: int = 42
    verbose: int = 1


def main():
    cfg = Config()
    seed_all(cfg.seed)
    out_dir, eval_only = artifact_dir("vpl_pairwise", timestamp=cfg.timestamp)
    VPLPairwiseExperiment(cfg).run(out_dir, eval_only=eval_only)
    print(f"[Done]  →  {out_dir}")


if __name__ == "__main__":
    main()
