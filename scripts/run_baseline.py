from dataclasses import dataclass, field

from src.utils.seed import seed_all
from src.utils.paths import artifact_dir
from src.model.independent.experiment import BaselineExperiment


@dataclass
class Config:
    train_driver_names: list = field(default_factory=lambda: [
        "김진명", "김태근", "조현석", "한규택", "박재일", "이지환"
    ])
    test_driver: str = "강신길"
    features: list = field(default_factory=lambda: [
        "IMU_VerAccelVal", "Pitch_rate_6D", "Bounce_rate_6D", "IMU_LongAccelVal"
    ])

    time_range: tuple = (5, 7)
    downsample: int = 5
    smooth: bool = False
    smooth_cutoff: float = 12.0
    smooth_order: int = 2
    normalize: bool = False
    val_size: float = 0.1
    seed: int = 42
    device: str = "cuda"
    verbose: int = 1

    timestamp: str = None  # None=새 실험, "test"=디버그 고정폴더, "20250101_120000"=평가만

    # Model
    model_type: str = "online_mlp"  # "online_mlp" | "online_lstm" | "online_attention" | "offline_lstm"
    hidden_dims: list = field(default_factory=lambda: [64, 1])
    dropout_rates: list = field(default_factory=lambda: [0.0, 0.0])
    act_name: str = "relu"
    reduce: str = "mean"

    # LSTM / Attention specific
    lstm_hidden: int = 64
    lstm_layers: int = 1
    lstm_dropout: float = 0.0
    attn_dim: int = 64
    attn_heads: int = 4
    attn_dropout: float = 0.1
    ffn_dim: int = 128
    ffn_dropout: float = 0.1

    # Training
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    batch_size: int = 64


def main():
    cfg = Config()
    seed_all(cfg.seed)
    out_dir, eval_only = artifact_dir("baseline", timestamp=cfg.timestamp)
    results = BaselineExperiment(cfg).run(out_dir, eval_only=eval_only)
    auroc = results.get(f"test/{cfg.test_driver}", {}).get("auroc", float("nan"))
    print(f"[Done] AUROC={auroc:.4f}  →  {out_dir}")


if __name__ == "__main__":
    main()
