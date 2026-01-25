import sys
import os
import optuna
import json
import argparse
from pathlib import Path

# Add project root to path to allow imports from src and scripts
sys.path.append(os.getcwd())

from scripts.train_copl import CFG, run_copl_training

def objective(trial):
    cfg = CFG()
    cfg.verbose = 0  # Suppress output for optimization
        
    cfg.hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
    cfg.gcf_layers = trial.suggest_int('gcf_layers', 1, 4)
    cfg.gcf_dropout = trial.suggest_float('gcf_dropout', 0.1, 0.5)
    cfg.item_item_weight = trial.suggest_float('item_item_weight', 0.0, 5.0)
    cfg.gcf_lr = trial.suggest_float('gcf_lr', 1e-4, 1e-3, log=True)
    
    cfg.rm_hidden = trial.suggest_categorical('rm_hidden', [32, 64, 128])
    cfg.rm_lr = trial.suggest_float('rm_lr', 1e-4, 1e-3, log=True)
    
    cfg.adapt_neg_weight = trial.suggest_float('adapt_neg_weight', 0.0, 2.0)
    cfg.adapt_user_softmax_temp = trial.suggest_float('adapt_user_softmax_temp', 0.1, 5.0)
    
    # Unique save root for each trial to avoid conflicts and organize outputs
    cfg.save_root = f"artifacts/copl/opt/trial_{trial.number}"

    try:
        metric = run_copl_training(cfg)
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        # Return a low value to indicate failure
        return 0.0
        
    return metric

def main():
    parser = argparse.ArgumentParser(description="Optimize CoPL hyperparameters using Optuna.")
    parser.add_argument("--n_trials", type=int, default=150, help="Number of trials.")
    parser.add_argument("--study_name", type=str, default="copl_optimization", help="Study name.")
    parser.add_argument("--db_path", type=str, default="artifacts/copl/opt/study.db", help="Path to SQLite DB.")
    args = parser.parse_args()

    # Create artifact directory
    os.makedirs(os.path.dirname(args.db_path), exist_ok=True)

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=f"sqlite:///{args.db_path}",
        load_if_exists=True
    )
    
    study.optimize(objective, n_trials=args.n_trials)
    
    print("\nOptimization Finished!")
    print("Best params:", study.best_params)
    print("Best value:", study.best_value)
    
    # Save best params
    best_params_path = Path("artifacts/copl/opt/best_params.json")
    with open(best_params_path, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"Best params saved to {best_params_path}")

if __name__ == "__main__":
    main()
