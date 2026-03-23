from dataclasses import dataclass, field

from src.utils.seed import seed_all
from src.utils.paths import artifact_dir
from src.model.copl.experiment import CoPLExperiment


@dataclass
class Config:
    train_driver_names: list = field(default_factory=lambda: [
        "김진명", "조현석", "한규택", "박재일", "이지환"
    ])
    test_driver: str = "강신길"
    features: list = field(default_factory=lambda: [
        "IMU_VerAccelVal", "Pitch_rate_6D", "Bounce_rate_6D", "IMU_LongAccelVal"
    ])

    time_range: tuple = (5, 7)
    downsample: int = 5
    smooth: bool = True
    smooth_cutoff: float = 15.0
    smooth_order: int = 2
    normalize: bool = True
    val_size: float = 0.1
    seed: int = 42
    device: str = "cuda"
    verbose: int = 1

    timestamp: str = "test"  # None=새 실험, "test"=디버그 고정폴더, "20250101_120000"=해당폴더 전체 로드
    load_vae: str = "test"    # None=학습, "test"/"20250101_120000"=해당 폴더에서 로드 (timestamp=시간이면 무시)
    load_gcf: str = None    # 동일
    load_rm: str = "test"     # 동일

    # GCF model selection
    gcf_model: str = "gcf"  # "gcf" | "gcf_gcn"
    gcf_m_i_type: str = "f"
    gcf_loss_type: str = "bce_diversity"  # "bce" | "bpr" | "cosine" | "softmax" | "margin"
    gcf_emb_dim: int = 32
    gcf_layers: int = 2
    gcf_dropout: float = 0.0
    item_item_weight: float = 0.5
    gcf_lr: float = 0.0005
    gcf_weight_decay: float = 0.0
    gcf_lambda_reg: float = 0.0
    gcf_epochs: int = 150
    use_pos_weight: bool = True
    gcf_loss_kwargs: dict = field(default_factory=lambda: {
        "w_ii": 1.4,         # item-item interaction weight
        "lambda_div": 0.0,   # bce_diversity loss
        "margin": 0.5,       # cosine, margin loss
        "temperature": 0.1,  # softmax loss
    })

    # Similarity graph
    similarity_method: str = "vae"
    mutual: bool = True
    knn_k: int = 10
    gamma_mul: float = 1.0
    ## PCA, Kernal PCA similarity
    pca_dim: int = 2
    ## DTW similarity
    dtw_gamma: float = 1.0
    ## VAE similarity
    vae_latent_dim: int = 16
    vae_epochs: int = 400
    vae_lr: float = 0.001
    vae_kl_weight: float = 0.05
    vae_batch_size: int = 128
    vae_hidden_channels: int = 32
    vae_metric: str = "cosine"
    vae_temperature: float = 0.2


    # RM model selection
    rm_model: str = "cnn"  # "mlp" | "cnn" | "mole_cnn"
    rm_hidden: int = 32
    rm_mlp_hidden: int = 64
    rm_kernel_size: int = 3
    rm_layers: int = 2
    rm_lr: float = 0.00026
    rm_weight_decay: float = 0.0
    rm_lambda_reg: float = 1e-6
    rm_epochs: int = 200
    rm_batch_size: int = 256
    rm_num_experts: int = 4
    rm_mole_rank: int = 6
    rm_mole_tau: float = 2.0

    # Test-time adaptation
    adapt_topk: int = 20
    adapt_use_neg: bool = True
    adapt_neg_weight: float = 0.81
    adapt_user_softmax_temp: float = 1.15

def main():
    cfg = Config()
    seed_all(cfg.seed)
    out_dir, eval_only = artifact_dir("copl", timestamp=cfg.timestamp)
    results = CoPLExperiment(cfg).run(out_dir, eval_only=eval_only)
    auroc = results.get(f"test/{cfg.test_driver}", {}).get("auroc", float("nan"))
    print(f"[Done] AUROC={auroc:.4f}  →  {out_dir}")


if __name__ == "__main__":
    main()
