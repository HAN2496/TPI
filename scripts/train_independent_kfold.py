import torch
import numpy as np
from sklearn.metrics import roc_auc_score

from src.configs.utils import create_model, load_config
from src.utils import ExperimentPaths, convert_driver_name, prepare_training_data_kfold
from src.utils.evaluator import Evaluator
from src.utils.logger import ExperimentLogger
from model.trainer import create_trainer
from src.utils.visualization import plot_training_curves, save_regression_params, plot_kfold_roc_curves

def train_model_kfold(driver_name, model_type, model_name, time_range=None,
                      downsample=1, n_splits=5, test_ratio=0.1, device="cpu", verbose=1):
    driver_name = convert_driver_name(driver_name)
    config = load_config(driver_name, model_type, model_name, verbose=verbose)
    model = create_model(driver_name, model_type, model_name, is_train=True, device=device, verbose=0)
    is_neural = model.is_neural

    kfold_paths = ExperimentPaths(driver_name, model_type, model_name, time_range, tag="kfold", tag_as_subdir=True)
    logger = ExperimentLogger(kfold_paths.run_dir, 'kfold_results', add_timestamp=False)
    logger.start()

    if verbose:
        print(f"\n{'='*60}")
        print(f"K-Fold Cross-Validation ({n_splits} folds)")
        print(f"Driver: {driver_name}, Model: {model_type}/{model_name}")
        print(f"{'='*60}\n")

    test_loader, folds = prepare_training_data_kfold(
        driver_name, config, time_range, downsample, n_splits, test_ratio, is_neural, random_state=42
    )

    fold_results = {'val_auroc': [], 'test_auroc': [], 'val_acc': [], 'test_acc': []}
    fold_predictions = [] # for OOF

    print(f"{'Fold':<6} {'Train':<7} {'Val':<5} {'Val AUROC':<11} {'Test AUROC':<11} {'Best Threshold':<12}")
    print("-" * 60)

    for fold_idx, train_loader, val_loader in folds:
        paths = ExperimentPaths(driver_name, model_type, model_name, time_range, tag=f"kfold/fold_{fold_idx}", tag_as_subdir=True)

        fold_model = create_model(driver_name, model_type, model_name, is_train=True, device=device, verbose=0)
        trainer = create_trainer(fold_model, config, best_model_path=paths.best_model, device=device, is_neural=is_neural)
        history, _ = trainer.train(train_loader, val_loader, verbose=(verbose >= 2))

        if is_neural:
            plot_training_curves(history, save_path=paths.get('training_curves.png', create=True))
        if is_neural:
            fold_model.load_state_dict(torch.load(paths.best_model, weights_only=True))

        evaluator = Evaluator(driver_name, fold_model, paths, device=device)

        # evluate validation
        val_metrics, y_val, p_val = evaluator.evaluate(val_loader, save_plots=True)
        best_threshold = val_metrics['threshold']

        # evaluate test
        test_metrics, _, _ = evaluator.evaluate(test_loader, threshold=best_threshold, save_plots=True, prefix='test')

        fold_results['val_auroc'].append(val_metrics['auroc'])
        fold_results['test_auroc'].append(test_metrics['auroc'])
        fold_results['val_acc'].append(val_metrics['accuracy'])
        fold_results['test_acc'].append(test_metrics['accuracy'])
        fold_predictions.append((y_val, p_val))

        if not is_neural:
            save_regression_params(fold_model, config['features'], save_path=paths.get('regression_params.png', create=True))

        print(f"{fold_idx:<6} {len(train_loader.dataset):<7} {len(val_loader.dataset):<5} "
                f"{val_metrics['auroc']:<11.4f} {test_metrics['auroc']:<11.4f} {best_threshold:<12.2f}")

    # OOF AUROC
    y_true_all = np.concatenate([np.asarray(y).reshape(-1) for y, _ in fold_predictions])
    y_prob_all = np.concatenate([np.asarray(p).reshape(-1) for _, p in fold_predictions])
    y_true_all = (y_true_all >= 0.5).astype(np.int32)
    oof_auroc = roc_auc_score(y_true_all, y_prob_all)

    plot_kfold_roc_curves(fold_predictions, save_path=kfold_paths.get('kfold_roc_curves.png', create=True))

    best_fold_idx = np.argmax(fold_results['val_auroc'])
    best_val_auroc = fold_results['val_auroc'][best_fold_idx]
    best_test_auroc = fold_results['test_auroc'][best_fold_idx]

    print("-" * 60)
    print(f"{'Best':<6} {'Fold '+str(best_fold_idx):<7} {'':<5} "
          f"{best_val_auroc:<11.4f} {best_test_auroc:<12.4f}")
    print(f"{'Mean':<6} {'':<7} {'':<5} "
          f"{np.mean(fold_results['val_auroc']):<11.4f} {np.mean(fold_results['test_auroc']):<12.4f}")
    print(f"{'OOF':<6} {'':<7} {'':<5} {oof_auroc:<11.4f}")
    print("=" * 60)

    logger.stop()

    return fold_results


# uv run python .\scripts\train_model_kfold.py -n 강신길 -mt online_mlp -mn base -t "[5,7]" -d 5 --n-splits 5 --test-ratio 0.1 --device cuda -v 1
def main():
    from src.utils.arg_parser import train_model_parser
    parser = train_model_parser()
    args = parser.parse_args()

    train_model_kfold(
        args.driver_name,
        args.model_type,
        args.model_name,
        args.time_range,
        args.downsample,
        n_splits=args.n_splits,
        test_ratio=args.test_ratio,
        device=args.device,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
