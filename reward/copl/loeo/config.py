"""Configuration of the CoPL LOEO driver (docs/copl/claude_notes/01_copl_plan.html)."""
from __future__ import annotations

from dataclasses import dataclass, field

CHANNELS5 = ("Pitch_rate_6D", "Bounce_rate_6D", "IMU_VerAccelVal", "IMU_LongAccelVal", "IMU_LatAccelVal")


@dataclass
class Config:
    # ---- data (identical to run_loeo.py so folds, splits and pseudonyms coincide)
    data: str = "real"                         # "real" | "synthetic"
    dataset_root: str = "datasets"
    evaluators: tuple = ()
    folds: tuple = ()                          # () = every eligible evaluator; else a subset (debug)
    channels: tuple = CHANNELS5                # channels loaded from the dataset
    graph_channels: tuple = ()                 # () = channels; channels the encoder / item graph sees
    rm_channels: tuple = ()                    # () = channels; channels the reward model sees
    around: tuple = (-2.0, 2.0)
    downsample: int = 2
    smooth: tuple = (10.0, 2)
    min_labels: int = 10
    min_per_class: int = 3
    pop_min_labels: int = 0
    pop_min_per_class: int = 0
    val_size: float = 0.1                      # per-user validation split inside the population (early stopping)
    # ---- protocol
    budgets: tuple = (0, 5, 10, 20)
    include_final: bool = True
    ctx_frac: float = 0.5
    n_offsets: int = 3
    seeds: tuple = (42,)
    reliable_w_max: float = 0.3
    # ---- encoder (item similarity)
    encoder: str = "scatter"                   # ae | vae | pca | kernel_pca | dtw | scatter | isobank | raw_rbf
                                               # scatter chosen from the encoders stage (docs/copl/claude_notes/03, E1)
    enc_list: tuple = ("ae", "vae", "pca", "scatter", "isobank", "raw_rbf")   # stage "encoders"
    enc_channel_sets: str = "loso"             # "full" | "loso" (full + leave-one-channel-out + singles) | "all" (31)
    enc_ks: tuple = (5, 10, 20, 30, 50, 100)   # neighbourhood sizes for the intrinsic metrics
    mutual: bool = False
    knn_k: int = 30
    gamma_mul: float = 1.0
    pca_dim: int = 16
    dtw_gamma: float = 1.0
    ae_latent_dim: int = 8
    ae_epochs: int = 300
    ae_lr: float = 1e-3
    ae_batch_size: int = 128
    ae_hidden_channels: int = 32
    ae_metric: str = "cosine"
    ae_temperature: float = 0.2
    vae_latent_dim: int = 16
    vae_epochs: int = 300
    vae_lr: float = 1e-3
    vae_kl_weight: float = 0.05
    vae_batch_size: int = 128
    vae_hidden_channels: int = 32
    vae_metric: str = "cosine"
    vae_temperature: float = 0.2
    scatter_n1: int = 8
    scatter_pca_dim: int = 0                   # 0 = no PCA after scattering
    isobank_rho_max: float = 0.95
    # ---- item-item graph sparsification (stage "sweep" varies these)
    graph_rule: str = "topk"                   # topk | mutual | epsilon | dense | cross_forced
    cross_min: int = 0                         # cross_forced: at least this many neighbours from other users
    sweep_ks: tuple = (5, 10, 20, 30, 50, 100)
    sweep_rules: tuple = ("topk", "mutual", "epsilon", "cross_forced")
    sweep_item_item_weights: tuple = (0.5,)
    # ---- GCF
    gcf_model: str = "gcf"
    gcf_m_i_type: str = "b"
    gcf_loss_type: str = "bce_diversity"
    gcf_emb_dim: int = 32
    gcf_layers: int = 2
    gcf_dropout: float = 0.0
    item_item_weight: float = 0.5
    gcf_lr: float = 6.8e-4
    gcf_weight_decay: float = 1e-3
    gcf_lambda_reg: float = 0.0
    gcf_epochs: int = 50
    use_pos_weight: bool = True
    gcf_loss_kwargs: dict = field(default_factory=lambda: {"w_ii": 2.0, "lambda_div": 0.5, "margin": 0.5, "temperature": 0.1})
    # ---- reward model
    rm_model: str = "cnn"                      # mlp | cnn | mole_cnn | preference_transformer
    rm_hidden: int = 32
    rm_mlp_hidden: int = 64
    rm_lr: float = 2.6e-4
    rm_weight_decay: float = 0.0
    rm_lambda_reg: float = 1e-6
    rm_epochs: int = 100
    rm_batch_size: int = 256
    rm_num_experts: int = 3
    rm_mole_rank: int = 6
    rm_mole_tau: float = 2.0
    rm_kernel_size: int = 3
    rm_layers: int = 2
    rm_num_heads: int = 8
    rm_max_len: int = 1000
    rm_bayes: str = "ensemble"                 # none | ensemble | mc_dropout  (ensemble: variance + calibration)
    rm_dropout: float = 0.0
    rm_mc_samples: int = 30
    rm_ensemble_k: int = 3
    rm_select: str = "loss"                    # early stopping on validation BCE ('loss') or AUROC ('auc')
    rm_calibrate: bool = True                  # temperature scaling fitted on the population validation set
    # ---- adaptation (CoPL vote -> softmax over population users)
    adapt_normalize: bool = True               # standardize the vote scores c_u before the softmax
    adapt_evidence: str = "none"               # "none" = original CoPL scale (grows with t; best in E4 at tau ~ 1),
                                               # "unit" = standardized (independent of t), "sqrt" = standardized * sqrt(t)
    # ---- mixture-consistent reward-model training: the reward model is queried with convex mixtures of
    # population embeddings (cold start = mean, adaptation = softmax mixture), so it is trained on them too
    rm_mix_prob: float = 0.5                   # fraction of training rows whose user embedding is replaced by a mixture
    rm_mix_alpha: float = 1.0                  # Dirichlet concentration of the mixture weights (1 = uniform simplex)
    # ---- test-time adaptation
    adapt_topk: int = 30
    adapt_use_neg: bool = True
    adapt_neg_weight: float = 1.0
    adapt_user_softmax_temp: float = 1.15
    # ---- ablations (stage "ablate" toggles these one at a time)
    use_item_item: bool = True
    use_adapt: bool = True
    oracle: bool = False                       # held-out evaluator's *context* labels join the GCF graph
    # ---- baselines
    pooled: bool = True
    indep: bool = True
    knn_vote: bool = True
    pooled_ft_epochs: int = 3                  # fine-tune epochs of the pooled CNN per (t, offset)
    pooled_ft_lr: float = 1e-4
    pooled_ctx_weight: float = 1.0             # sample weight of context episodes during fine-tuning
    indep_epochs: int = 60
    indep_min_labels: int = 5
    knn_vote_k: int = 30
    knn_vote_alpha: float = 1.0                # Laplace smoothing of the vote
    # ---- tuning (stage "tune": inner LOEO inside each population set)
    tune_trials: int = 40
    tune_inner_folds: int = 3
    tune_inner_mix: bool = True                # longest streams plus the shortest eligible one (short-stream coverage)
    tune_budgets: tuple = (0, 5, 10, 20)
    tune_seed: int = 7
    tune_rm_bayes: str = "none"                # single reward model during tuning (cost); main uses rm_bayes
    tune_space: dict = field(default_factory=lambda: {
        "gcf_emb_dim": [16, 32, 64],
        "item_item_weight": [0.25, 0.5, 1.0, 2.0],
        "knn_k": [10, 30, 100],
        "adapt_user_softmax_temp": [0.25, 0.5, 1.0, 2.0, 4.0],
        "adapt_evidence": ["none", "unit", "sqrt"],
        "adapt_neg_weight": [0.0, 0.5, 1.0],
        "gcf_lr": [2e-4, 7e-4, 2e-3],
        "rm_lr": [1e-4, 3e-4, 1e-3],
    })
    use_tuned: bool = False                    # main/ablate/sweep/channels read folds/<name>_tune.json if present
    # ---- channels study (stage "channels")
    channel_configs: str = "loso"              # "loso" = LOCO 5 + a_z only + IMU only
    # ---- synthetic
    syn_n_evaluators: int = 8
    syn_min_episodes: int = 30
    syn_max_episodes: int = 160
    syn_k_common: int = 6
    syn_k_individual: int = 3
    rho_max: float = 0.95                      # used by the synthetic generator only
    # ---- run
    stage: str = "main"                        # encoders | main | ablate | sweep | channels | tune | report | all
    timestamp: str = None
    run_name: str = "copl_loeo"
    seed: int = 42
    device: str = "auto"                       # auto | cuda | cpu
    verbose: int = 1
    fast: bool = False


FAST = dict(ae_epochs=5, vae_epochs=5, gcf_epochs=5, rm_epochs=5, indep_epochs=5, pooled_ft_epochs=1,
            rm_ensemble_k=2, rm_mc_samples=5, scatter_n1=4, n_offsets=2, enc_ks=(5, 10), sweep_ks=(5, 20),
            sweep_rules=("topk", "mutual"), tune_trials=2, tune_inner_folds=2, knn_k=10, adapt_topk=10,
            syn_n_evaluators=5, syn_min_episodes=20, syn_max_episodes=60, dtw_gamma=1.0)
