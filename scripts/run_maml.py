from dataclasses import dataclass, field

import torch

from src.utils.seed import seed_all
from src.utils.paths import artifact_dir
from src.model.maml.experiment import MAMLExperiment


@dataclass
class Config:
    features: list = field(default_factory=lambda: [
        "IMU_VerAccelVal", "Pitch_rate_6D", "Bounce_rate_6D", "IMU_LongAccelVal"
    ])
    train_driver_names: list = field(default_factory=lambda: [
        "김진명", "김태근", "조현석", "한규택", "박재일", "이지환"
    ])
    test_driver_name: str = "강신길"
    time_range: tuple = (5, 7)
    downsample: int = 5
    smooth: bool = False
    smooth_cutoff: float = 12.0
    smooth_order: int = 2
    normalize: bool = False
    timestamp: str = None

    hidden_dim: int = 64
    inner_lr: float = 0.005
    outer_lr: float = 0.001
    inner_steps: int = 3
    n_support: int = 20
    n_query: int = 20
    n_tasks_per_epoch: int = 20
    meta_epochs: int = 200
    device: str = "cuda"
    seed: int = 42
    verbose: int = 1


def main():
    cfg = Config()
    seed_all(cfg.seed)
    out_dir, eval_only = artifact_dir("maml", timestamp=cfg.timestamp)
    results = MAMLExperiment(cfg).run(out_dir, eval_only=eval_only)
    auroc = results.get(f"test/{cfg.test_driver_name}", {}).get("auroc", float("nan"))
    print(f"[Done] AUROC={auroc:.4f}  →  {out_dir}")


if __name__ == "__main__":
    main()
