from dataclasses import dataclass, field

from src_new.utils.seed import seed_all
from src_new.utils.paths import artifact_dir
from src_new.model.tsfresh.experiment import TSFreshExperiment


@dataclass
class Config:
    train_driver_names: list = field(default_factory=lambda: [
        "김진명", "김태근", "조현석", "한규택", "박재일", "이지환"
    ])
    test_driver_name: str = "강신길"
    features: list = field(default_factory=lambda: [
        "IMU_VerAccelVal", "Pitch_rate_6D", "Bounce_rate_6D", "IMU_LongAccelVal"
    ])
    time_range: tuple = (5, 7)
    downsample: int = 5
    smooth: bool = False
    smooth_cutoff: float = 12.0
    smooth_order: int = 2
    normalize: bool = False
    seed: int = 42
    verbose: int = 1
    timestamp: str = None  # None=새 실험, "test"=디버그 고정폴더, "20250101_120000"=평가만

    n_components: int = 8
    extraction_settings: str = "efficient"
    C: float = 1.0
    k_folds: int = 5


def main():
    cfg = Config()
    seed_all(cfg.seed)
    out_dir, eval_only = artifact_dir("tsfresh", timestamp=cfg.timestamp)
    TSFreshExperiment(cfg).run(out_dir, eval_only=eval_only)
    print(f"[Done]  →  {out_dir}")


if __name__ == "__main__":
    main()
