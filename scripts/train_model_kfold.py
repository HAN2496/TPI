import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

from src.configs.utils import create_model, load_config
from src.configs.registries import MODELS
from src.model.base import RegressionModel
from src.utils import convert_driver_name, ExperimentPaths
from src.utils.utils import prepare_training_data_kfold
from src.utils.trainer import BaseTrainer, RegressionTrainer
from src.utils.visualization import save_all_plots, plot_kfold_roc_curves


def train_model_kfold(driver_name, model_type, model_name='base', time_range=None,
                      train_downsample=1, val_downsample=None, n_splits=5, test_ratio=0.1, device="cpu", verbose=1):
    
    driver_name = convert_driver_name(driver_name)
    config = load_config(driver_name, model_type, model_name, verbose=verbose)
    is_regression = issubclass(MODELS[config['model_type']], RegressionModel)
    is_online = config['model_type'] == 'online_mlp'

    if verbose:
        print(f"\n{'='*60}")
        print(f"K-Fold Cross-Validation ({n_splits} folds)")
        print(f"Driver: {driver_name}, Model: {model_type}/{model_name}")
        print(f"Time range: {time_range}, Train DS: {train_downsample}, Val DS: {val_downsample or train_downsample}")
        print(f"Test set: {test_ratio*100:.0f}%")
        print(f"{'='*60}\n")

    fold_results = {'val_acc': [], 'val_auroc': [], 'test_acc': [], 'test_auroc': []}
    fold_predictions = []

    test_loader, fold_gen = prepare_training_data_kfold(
        driver_name, config, time_range, train_downsample, val_downsample, n_splits, test_ratio
    )

    print(f"{'Fold':<6} {'Train':<7} {'Val':<5} {'Val Acc':<9} {'Val AUROC':<11} {'Test Acc':<10} {'Test AUROC':<11}")
    print("-" * 70)

    for fold_idx, train_loader, val_loader in fold_gen:
        model = create_model(driver_name, model_type, model_name, is_train=True, device=device, verbose=0)
        paths = ExperimentPaths(driver_name, model_type, model_name, time_range, tag=f"kfold/fold_{fold_idx}", tag_as_subdir=True)

        if is_regression:
            trainer = RegressionTrainer(model, config['trainer'], best_model_path=None)
        else:
            trainer = BaseTrainer(model, config['trainer'], best_model_path=paths.best_model, device=device)

        val_acc, val_auroc, history = trainer.train(train_loader, val_loader, verbose=(verbose >= 2))

        if not is_regression:
            model.load_state_dict(torch.load(paths.best_model, weights_only=True))
            model.to('cpu')

        y_true, y_probs = save_all_plots(model, history, val_loader, paths, is_online=is_online, is_regression=is_regression, feature_cols=config['features'])
        fold_predictions.append((y_true, y_probs))

        X_test, y_test = test_loader.dataset.tensors
        y_test_probs = model.predict_probability(X_test)
        test_auroc = roc_auc_score(y_test.numpy(), y_test_probs)
        test_preds = (y_test_probs >= model.best_threshold).astype(int)
        test_acc = accuracy_score(y_test.numpy(), test_preds)

        fold_results['val_acc'].append(val_acc)
        fold_results['val_auroc'].append(val_auroc)
        fold_results['test_acc'].append(test_acc)
        fold_results['test_auroc'].append(test_auroc)

        if verbose:
            print(f"{fold_idx+1:<6} {len(train_loader.dataset):<7} {len(val_loader.dataset):<5} "
                  f"{val_acc:<9.4f} {val_auroc:<11.4f} {test_acc:<10.4f} {test_auroc:<11.4f}")

    y_true_all = np.concatenate([y_true for y_true, _ in fold_predictions])
    y_probs_all = np.concatenate([y_probs for _, y_probs in fold_predictions])
    oof_auroc = roc_auc_score(y_true_all, y_probs_all)

    kfold_paths = ExperimentPaths(driver_name, model_type, model_name, time_range, tag="kfold", tag_as_subdir=True)
    plot_kfold_roc_curves(fold_predictions, save_path=kfold_paths.get('kfold_roc_curves.png', create=True))

    print("-" * 70)
    print(f"{'Mean':<6} {'':<7} {'':<5} "
            f"{np.mean(fold_results['val_acc']):<9.4f} {np.mean(fold_results['val_auroc']):<11.4f} "
            f"{np.mean(fold_results['test_acc']):<10.4f} {np.mean(fold_results['test_auroc']):<11.4f}")
    print(f"{'Std':<6} {'':<7} {'':<5} "
            f"{np.std(fold_results['val_acc']):<9.4f} {np.std(fold_results['val_auroc']):<11.4f} "
            f"{np.std(fold_results['test_acc']):<10.4f} {np.std(fold_results['test_auroc']):<11.4f}")
    print(f"{'OOF':<6} {'':<7} {'':<5} {'':<9} {oof_auroc:<11.4f}")
    print("=" * 70)

    return fold_results


# uv run python .\scripts\train_model_kfold.py -d 강신길 -mt abs_quad -mn base -t "[5,7]" -td 5 --n-splits 5 --device cuda -v 1
# uv run python .\scripts\train_model_kfold.py -d 강신길 -mt online_mlp -mn base -t "[5,7]" -td 5 --n-splits 5 --device cuda -v 1
# uv run python .\scripts\train_model_kfold.py -d 강신길 -mt online_lstm -mn base -t "[5,8]" -td 5 --n-splits 5 --device cuda -v 1
# With test set (20%):
# uv run python .\scripts\train_model_kfold.py -d 강신길 -mt online_regression -mn base -t "[5,7]" -td 5 --n-splits 5 --test-ratio 0.2 --device cuda -v 1
def main():
    from src.utils.arg_parser import train_model_parser
    parser = train_model_parser()
    parser.add_argument('--n-splits', type=int, default=5, help='Number of folds')
    parser.add_argument('--test-ratio', type=float, default=0.1, help='Test set ratio (0.0-1.0, default: 0.1)')
    args = parser.parse_args()

    train_model_kfold(
        args.driver_name,
        args.model_type,
        args.model_name,
        args.time_range,
        args.train_downsample,
        args.val_downsample,
        n_splits=args.n_splits,
        test_ratio=args.test_ratio,
        device=args.device,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
