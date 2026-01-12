import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

from src.optimizer.bayesian_2 import OPTIMIZERS_V2
from src.utils import convert_driver_name, prepare_training_data_kfold, ExperimentPaths
from src.utils.utils import save_config
from src.configs.utils import load_config
from src.configs.registries import MODELS
from src.model.base import RegressionModel
from model.trainer import BaseTrainer, RegressionTrainer

def evaluate_on_test_set(driver_name, config, fold_idx, test_loader, time_range, downsample, n_splits, test_ratio, device):
    _, fold_gen = prepare_training_data_kfold(driver_name, config, time_range, train_downsample=downsample, n_splits=n_splits, test_ratio=test_ratio)

    for current_fold_idx, train_loader, val_loader in fold_gen:
        if current_fold_idx != fold_idx:
            continue

        input_dim = len(config['features'])
        config['args']['input_dim'] = input_dim
        model = MODELS[config['model_type']](**config['args'])
        is_regression = RegressionModel in type(model).__mro__

        if is_regression:
            trainer = RegressionTrainer(model, config['trainer'])
            trainer.train(train_loader, val_loader, verbose=False)
        else:
            model.to("cpu")
            trainer = BaseTrainer(model, config['trainer'], device="cpu")
            trainer.train(train_loader, val_loader, epochs=30, verbose=False)

        model.eval()
        X_test, y_test = test_loader.dataset.tensors
        y_probs = model.predict_proba(X_test)

        y_probs_np = y_probs.cpu().numpy() if torch.is_tensor(y_probs) else y_probs
        y_test_np = y_test.cpu().numpy() if torch.is_tensor(y_test) else y_test

        test_auroc = roc_auc_score(y_test_np, y_probs_np)
        test_acc = accuracy_score(y_test_np, (y_probs_np >= model.best_threshold).astype(int))

        return test_auroc, test_acc

def optimize_model(driver_name, model_type, time_range, downsample, n_splits, test_ratio=0.1,
                   n_trials=100, use_feature_selection=False, device="cpu", optimizer_type="bayesian", tag="optuna", verbose=1):
    driver_name = convert_driver_name(driver_name)
    base_config = load_config(driver_name, model_type, 'base')
    test_loader, _ = prepare_training_data_kfold(driver_name, base_config, time_range, downsample, n_splits, test_ratio)

    fold_results = []

    for fold_idx in range(n_splits):
        print(f"\n{'='*80}")
        print(f"Optimizing Fold {fold_idx+1}/{n_splits}")
        print(f"{'='*80}")

        optimizer_class = OPTIMIZERS_V2[base_config['model_type']]
        optimizer = optimizer_class(driver_name, model_type, time_range, downsample, n_splits, test_ratio, fold_idx, use_feature_selection, device)

        study, best_config = optimizer.optimize(n_trials, tag)

        test_auroc, test_acc = evaluate_on_test_set(driver_name, best_config, fold_idx, test_loader, time_range, downsample, n_splits, test_ratio, device)

        fold_results.append({
            'fold_idx': fold_idx,
            'best_val_auroc': study.best_value,
            'test_auroc': test_auroc,
            'test_acc': test_acc,
            'best_config': best_config
        })

        print(f"Fold {fold_idx} - Val AUROC: {study.best_value:.4f} | Test AUROC: {test_auroc:.4f} | Test Acc: {test_acc:.4f}")

    paths = ExperimentPaths(driver_name, model_type, None, time_range, tag=tag)
    results_json = paths.get("all_folds_results.json", create=True)

    all_results = {
        'n_splits': n_splits,
        'n_trials': n_trials,
        'test_ratio': test_ratio,
        'use_feature_selection': use_feature_selection,
        'folds': fold_results,
        'summary': {
            'mean_val_auroc': np.mean([r['best_val_auroc'] for r in fold_results]),
            'std_val_auroc': np.std([r['best_val_auroc'] for r in fold_results]),
            'mean_test_auroc': np.mean([r['test_auroc'] for r in fold_results]),
            'std_test_auroc': np.std([r['test_auroc'] for r in fold_results]),
            'mean_test_acc': np.mean([r['test_acc'] for r in fold_results]),
            'std_test_acc': np.std([r['test_acc'] for r in fold_results]),
        }
    }

    with open(results_json, 'w') as f:
        json.dump(all_results, f, indent=2)

    for fold_idx, result in enumerate(fold_results):
        model_name = f"bo_fold_{fold_idx}"
        save_config(result['best_config'], driver_name, model_type, model_name)

    best_fold_idx = max(range(n_splits), key=lambda i: fold_results[i]['test_auroc'])

    print(f"\n{'='*80}")
    print(f"Per-Fold Optimization Results")
    print(f"{'='*80}")
    print(f"{'Fold':<6} {'Val AUROC':<11} {'Test Acc':<10} {'Test AUROC':<11}")
    print("-" * 80)

    for result in fold_results:
        print(f"{result['fold_idx']:<6} {result['best_val_auroc']:<11.4f} {result['test_acc']:<10.4f} {result['test_auroc']:<11.4f}")

    print("-" * 80)
    print(f"{'Mean':<6} {all_results['summary']['mean_val_auroc']:<11.4f} "
          f"{all_results['summary']['mean_test_acc']:<10.4f} {all_results['summary']['mean_test_auroc']:<11.4f}")
    print(f"{'Std':<6} {all_results['summary']['std_val_auroc']:<11.4f} "
          f"{all_results['summary']['std_test_acc']:<10.4f} {all_results['summary']['std_test_auroc']:<11.4f}")
    print(f"{'Best':<6} Fold {best_fold_idx} (Test AUROC: {fold_results[best_fold_idx]['test_auroc']:.4f})")
    print("=" * 80)
    print(f"\nResults saved to: {results_json}")
    print(f"Configs saved as: {driver_name}/{model_type}/bo_fold_0 ~ bo_fold_{n_splits-1}")

    return all_results

def main():
    from src.utils.arg_parser import optimize_model_parser
    parser = optimize_model_parser()
    parser.add_argument('--test-ratio', type=float, default=0.1, help='Test set ratio (0.1-1.0, default: 0.1)')
    args = parser.parse_args()

    tag = args.tag
    if args.use_feature_selection:
        tag = f"{tag}_fs"

    optimize_model(
        args.driver_name,
        args.model_type,
        args.time_range,
        args.downsample,
        args.n_splits,
        n_trials=args.n_trials,
        test_ratio=args.test_ratio,
        use_feature_selection=args.use_feature_selection,
        device=args.device,
        optimizer_type=args.optimizer_type,
        tag=tag,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()
