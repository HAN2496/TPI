from dataclasses import dataclass, field

import torch

from src.utils.seed import seed_all
from src.utils.paths import artifact_dir
from src.model.independent.experiment import IndependentExperiment


@dataclass
class Config:
    driver_name: str = "강신길"
    features: list = field(default_factory=lambda: [
        "IMU_VerAccelVal", "Pitch_rate_6D", "Bounce_rate_6D", "IMU_LongAccelVal"
    ])
    time_range: tuple = (5, 7)
    downsample: int = 5
    smooth: bool = False
    smooth_cutoff: float = 12.0
    smooth_order: int = 2
    normalize: bool = False
    test_size: float = 0.2
    seed: int = 42
    device: str = "cuda"
    verbose: int = 1
    timestamp: str = None  # None=새 실험, "test"=디버그 고정폴더, "20250101_120000"=평가만

    model_type: str = "online_mlp"  # "online_mlp" | "online_lstm" | "online_attention" | "offline_lstm"

    trainer: dict = field(default_factory=lambda: {
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 200,
        "batch_size": 32,
    })

    hidden_dims: list = field(default_factory=lambda: [64, 32])
    dropout_rates: list = field(default_factory=lambda: [0.1, 0.0])
    act_name: str = "relu"
    reduce: str = "mean"


def main():
    cfg = Config()
    seed_all(cfg.seed)
    out_dir, eval_only = artifact_dir("independent", timestamp=cfg.timestamp)
    IndependentExperiment(cfg).run(out_dir, eval_only=eval_only)
    print(f"[Done]  →  {out_dir}")


if __name__ == "__main__":
    main()
