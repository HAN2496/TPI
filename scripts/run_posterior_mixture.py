from dataclasses import dataclass, field
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.seed import seed_all
from src.utils.paths import artifact_dir
from src.model.posterior_mixture import PosteriorMixtureExperiment


@dataclass
class Config:
    train_driver_names: list = field(default_factory=lambda: [
        "김진명", "조현석", "한규택", "박재일", "이지환"
    ])
    test_driver_names: list = field(default_factory=lambda: ["강신길"])
    features: list = field(default_factory=lambda: [
        # "IMU_VerAccelVal",
        "Pitch_rate_6D",
        "Bounce_rate_6D",
        "IMU_LongAccelVal",
        "IMU_LatAccelVal",
        "IMU_YawRtVal",
        # "IMU_RollRtVal",
        "Roll_rate_6D",
        "SAS_AnglVal",
        "SAS_SpdVal",
        "VCU_AccPedDepVal",
        # "IEB_EstTtlBrkFrcNmV",
    ])

    time_range: tuple = (4, 6)
    downsample: int = 5
    smooth: bool = True
    smooth_cutoff: float = 10.0
    smooth_order: int = 3

    timestamp: str = "test"
    seed: int = 42
    verbose: int = 1

    # Feature map: standardized linear terms plus soft hinge bases.
    n_subwindows: int = 2
    knots: tuple = ()
    include_below: bool = False

    # Posterior-mixture training hyperparameters.
    prior_mode: str = "posterior_mixture"
    penalty: str = "l1"
    C: float = 0.08
    prior_var_floor: float = 0.2
    user_var_scale: float = 0.25
    update_temperature: float = 0.75
    map_max_iter: int = 16
    map_tol: float = 1e-5
    map_use_full_cov: bool = True
    prototype_shrinkage: float = 1.0
    component_temperature: float = 0.4
    global_component_weight: float = 0.05

    # Posterior-mixture specific.
    transfer_var_scale: float = 0.25
    posterior_var_floor: float = 1e-6
    component_var_floor: float = 1e-6


def main():
    cfg = Config()
    seed_all(cfg.seed)
    out_dir, eval_only = artifact_dir("posterior_mixture", timestamp=cfg.timestamp)
    results = PosteriorMixtureExperiment(cfg).run(out_dir, eval_only=eval_only)
    for name in cfg.test_driver_names:
        auroc = results.get(f"test/{name}", {}).get("auroc", float("nan"))
        print(f"[Done] {name}: AUROC={auroc:.4f}")
    print(f"  -> {out_dir}")


if __name__ == "__main__":
    main()
