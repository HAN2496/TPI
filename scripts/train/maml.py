# scripts/train/maml.py

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
class MetaConfig:
    hidden_dim: int = 64
    inner_lr: float = 0.005
    outer_lr: float = 0.001
    inner_steps: int = 3
    n_support: int = 20
    n_query: int = 20

    meta_epochs: int = 200
    device: str = "cuda"
    seed: int = 42
    verbose: int = 1







