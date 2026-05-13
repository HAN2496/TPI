from dataclasses import dataclass, field
from src.utils.seed import seed_all
from src.utils.paths import artifact_dir
from src.model.moe.experiment import MoEExperiment # 경로에 맞게 수정

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
    smooth: bool = True
    smooth_cutoff: float = 15.0
    smooth_order: int = 2
    normalize: bool = True

    timestamp: str = "test"

    # MoE & Training Hyperparams
    user_dim: int = 32
    hidden_dim: int = 64
    num_experts: int = 4
    batch_size: int = 128
    lr: float = 0.001
    weight_decay: float = 1e-4
    epochs: int = 200
    val_size: float = 0.1

    # Test-time Adaptation Hyperparams
    adapt_lr: float = 0.05
    adapt_steps: int = 20

    device: str = "cuda"
    seed: int = 42
    verbose: int = 1

def main():
    cfg = Config()
    seed_all(cfg.seed)
    out_dir, eval_only = artifact_dir("moe", timestamp=cfg.timestamp)
    results = MoEExperiment(cfg).run(out_dir, eval_only=eval_only)
    auroc = results.get(f"test/{cfg.test_driver_name}", {}).get("auroc", float("nan"))
    print(f"[Done] AUROC={auroc:.4f}  →  {out_dir}")

if __name__ == "__main__":
    main()