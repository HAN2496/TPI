import os
import json
import optuna
import argparse

from scripts.train_copl import CFG, run_copl_training


def objective(trial):
    cfg = CFG()
    cfg.verbose = 0

    # Graph Structure
    cfg.pca_dim = trial.suggest_int('pca_dim', 4, 48)
    cfg.gamma_mul = trial.suggest_float('gamma_mul', 0.1, 5.0, log=True)
    cfg.knn_k = trial.suggest_int('knn_k', 3, 30)

    # GCF Embedding Learning
    cfg.hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
    cfg.gcf_layers = trial.suggest_int('gcf_layers', 1, 5)
    cfg.gcf_dropout = trial.suggest_float('gcf_dropout', 0.0, 0.5)
    cfg.item_item_weight = trial.suggest_float('item_item_weight', 0.0, 10.0)
    cfg.gcf_lr = trial.suggest_float('gcf_lr', 1e-5, 5e-3, log=True)
    cfg.gcf_epochs = trial.suggest_int('gcf_epochs', 50, 300)
    cfg.gcf_lambda_reg = trial.suggest_float('gcf_lambda_reg', 1e-6, 1e-2, log=True)

    # Reward Model & Adaptation
    cfg.rm_hidden = trial.suggest_categorical('rm_hidden', [32, 64, 128])
    cfg.rm_mlp_hidden = trial.suggest_categorical('rm_mlp_hidden', [64, 128, 256])
    cfg.rm_batch_size = trial.suggest_categorical('rm_batch_size', [128, 256, 512])
    cfg.rm_lr = trial.suggest_float('rm_lr', 1e-5, 5e-3, log=True)

    cfg.attach_topk_items = trial.suggest_int('attach_topk_items', 3, 50)
    cfg.adapt_neg_weight = trial.suggest_float('adapt_neg_weight', 0.0, 3.0)
    cfg.adapt_user_softmax_temp = trial.suggest_float('adapt_user_softmax_temp', 0.1, 10.0)

    cfg.save_root = f"artifacts/copl/opt/trial_{trial.number}"

    try:
        test_auc = run_copl_training(cfg, trial=trial)
        return test_auc
    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"[ERROR] Trial {trial.number} failed: {e}")
        return float('-inf')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=500)
    parser.add_argument("--study_name", type=str, default="copl_optimization")
    parser.add_argument("--db_path", type=str, default="artifacts/copl/opt/study.db")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.db_path), exist_ok=True)

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=10,
        interval_steps=1,
    )

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=f"sqlite:///{args.db_path}",
        load_if_exists=True,
        pruner=pruner
    )

    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)

    print("\nOptimization Finished!")
    print(f"Best value: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    best_path = "artifacts/copl/opt/best_params.json"
    with open(best_path, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"Best params saved to {best_path}")


if __name__ == "__main__":
    main()
