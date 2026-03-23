import optuna
from pathlib import Path

from src.utils.seed import seed_all
from src.model.copl.experiment import CoPLExperiment
from scripts.run_copl import Config


# ── 설정 ──────────────────────────────────────────────────────────────────────
VAE_TIMESTAMP = "test"   # 재사용할 VAE 폴더 (None이면 매 trial 학습)
GCF_TIMESTAMP = None     # 재사용할 GCF 폴더 (None이면 매 trial 학습)
RM_TIMESTAMP  = "test"     # 재사용할 RM 폴더  (None이면 매 trial 학습)
STUDY_NAME    = "copl_opt_v1"       # 바꾸면 새 study 시작, 유지하면 resume
N_TRIALS      = 50
TEST_DRIVER   = "강신길"
# ─────────────────────────────────────────────────────────────────────────────


def make_cfg(trial: optuna.Trial) -> Config:
    cfg = Config()
    cfg.verbose  = 0
    cfg.load_vae = VAE_TIMESTAMP
    if GCF_TIMESTAMP is not None:
        cfg.load_gcf = GCF_TIMESTAMP
    if RM_TIMESTAMP is not None:
        cfg.load_rm = RM_TIMESTAMP

    # ── GCF ──────────────────────────────────────────────────────────────────
    # cfg.gcf_m_i_type     = trial.suggest_categorical("gcf_m_i_type", ["d", "e", "f"])
    if cfg.gcf_m_i_type == "f":
        cfg.gcf_loss_kwargs["w_ii"] = trial.suggest_float("gcf_w_ii", 0.1, 3.0)
    # cfg.gcf_emb_dim      = trial.suggest_categorical("gcf_emb_dim", [32, 64, 128])
    # cfg.gcf_layers       = trial.suggest_int("gcf_layers", 1, 3)
    # cfg.gcf_weight_decay = trial.suggest_float("gcf_weight_decay", 1e-4, 1e-1, log=True)
    # cfg.item_item_weight = trial.suggest_float("item_item_weight", 0.1, 3.0)
    # cfg.gcf_loss_kwargs["lambda_div"] = trial.suggest_float("lambda_div", 0.0, 1.0)

    # cfg.knn_k       = trial.suggest_int("knn_k", 5, 100)

    # ── RM ───────────────────────────────────────────────────────────────────
    # cfg.rm_hidden = trial.suggest_categorical("rm_hidden", [32, 64, 128])
    # cfg.rm_layers = trial.suggest_int("rm_layers", 1, 3)

    # # ── Test-time adaptation ──────────────────────────────────────────────────
    # cfg.adapt_topk              = trial.suggest_int("adapt_topk", 5, 50)
    # cfg.adapt_neg_weight        = trial.suggest_float("adapt_neg_weight", 0.3, 1.5)
    # cfg.adapt_user_softmax_temp = trial.suggest_float("adapt_user_softmax_temp", 0.5, 2.0)

    return cfg


def objective(trial: optuna.Trial) -> float:
    cfg = make_cfg(trial)
    seed_all(cfg.seed)

    out_dir = Path(f"artifacts/copl/opt_{STUDY_NAME}/trial_{trial.number:03d}")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = CoPLExperiment(cfg).run(out_dir, eval_only=False)
    auroc = results.get(f"test/{TEST_DRIVER}", {}).get("auroc", float("nan"))

    trial.set_user_attr("out_dir", str(out_dir))
    trial.set_user_attr("auroc", auroc)
    return auroc


def main():
    storage = f"sqlite:///artifacts/copl/opt_{STUDY_NAME}.db"
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage,
        direction="maximize",
        load_if_exists=True,   # resume 지원
    )

    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"[Optimize] Study: {STUDY_NAME}  |  완료된 trials: {completed}  |  목표: {N_TRIALS}")

    study.optimize(objective, n_trials=N_TRIALS - completed)

    print("\n=== Best Trial ===")
    best = study.best_trial
    print(f"  AUROC   : {best.value:.4f}")
    print(f"  out_dir : {best.user_attrs.get('out_dir', '-')}")
    print("  Params  :")
    for k, v in best.params.items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
