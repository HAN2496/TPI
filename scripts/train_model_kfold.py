import numpy as np
import torch
from src.configs.utils import create_model, load_config, load_features
from src.utils import convert_driver_name, ExperimentPaths
from src.utils.utils import prepare_training_data_kfold
from src.utils.trainer import BaseTrainer, CombinationTrainer
from src.utils.visualization import save_all_plots, plot_kfold_roc_curves


def train_model_kfold(driver_name, model_name, time_range, feature_version, train_downsample, val_downsample=None, n_splits=5, device="cpu", verbose=1):
    driver_name = convert_driver_name(driver_name)
    config = load_config(driver_name, model_name, verbose=verbose)
    is_combination = config['model_type'] == 'online_combination'
    is_online = config['model_type'] == 'online_mlp' or config['model_type'] == 'online_combination'
    feature_cols = load_features(feature_version)

    if verbose:
        print(f"\n{'='*60}")
        print(f"K-Fold Cross-Validation ({n_splits} folds)")
        print(f"Driver: {driver_name}, Model: {model_name}")
        print(f"Time range: {time_range}, Train downsample: {train_downsample}, Val downsample: {val_downsample or train_downsample}")
        print(f"{'='*60}\n")

    fold_results = {'acc': [], 'auroc': []}
    fold_predictions = []

    for fold_idx, train_loader, val_loader in prepare_training_data_kfold(
        driver_name, config, time_range, train_downsample, val_downsample, n_splits=n_splits, feature_version=feature_version
    ):
        if verbose:
            print(f"Fold {fold_idx + 1}/{n_splits} - Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

        model = create_model(driver_name, model_name, feature_version, is_train=True, device=device, verbose=0)

        paths = ExperimentPaths(driver_name, model_name, feature_version, time_range,
                                tag=f"kfold/fold_{fold_idx}", tag_as_subdir=True)

        if is_combination:
            trainer = CombinationTrainer(model, config['trainer'], best_model_path=None)
        else:
            trainer = BaseTrainer(model, config['trainer'], best_model_path=paths.best_model, device=device)

        val_acc, val_auroc, history = trainer.train(train_loader, val_loader, verbose=(verbose >= 2))

        if not is_combination:
            model.load_state_dict(torch.load(paths.best_model, weights_only=True))
            model.to('cpu')

        y_true, y_probs = save_all_plots(model, history, val_loader, paths, is_online=is_online, feature_cols=feature_cols)

        fold_results['acc'].append(val_acc)
        fold_results['auroc'].append(val_auroc)
        fold_predictions.append((y_true, y_probs))

        if verbose:
            print(f"  - Acc: {val_acc:.4f}, AUROC: {val_auroc:.4f}")
            print(f"  - Plots saved to: {paths.run_dir}\n")

    mean_acc = np.mean(fold_results['acc'])
    std_acc = np.std(fold_results['acc'])
    mean_auroc = np.mean(fold_results['auroc'])
    std_auroc = np.std(fold_results['auroc'])

    kfold_paths = ExperimentPaths(driver_name, model_name, feature_version, time_range, tag="kfold")
    plot_kfold_roc_curves(fold_predictions, save_path=kfold_paths.get('kfold_roc_curves.png', create=True))

    if verbose:
        print(f"{'='*60}")
        print(f"K-Fold CV Results")
        print(f"{'='*60}")
        print(f"Mean Accuracy:  {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"Mean AUROC:     {mean_auroc:.4f} ± {std_auroc:.4f}")
        print(f"\nFold-wise AUROC: {[f'{x:.4f}' for x in fold_results['auroc']]}")
        print(f"K-Fold combined ROC curve saved to: {kfold_paths.get('kfold_roc_curves.png')}")
        print(f"{'='*60}\n")

    return fold_results


# uv run python .\scripts\train_model_kfold.py -d 강신길 -m online_mlp -t "[5,7]" -f v1 -td 5 --n-splits 5 --device cuda -v 1

def main():
    from src.utils.arg_parser import train_model_parser
    parser = train_model_parser()
    parser.add_argument('--n-splits', type=int, default=5, help='Number of folds')
    args = parser.parse_args()

    train_model_kfold(
        args.driver_name,
        args.model_name,
        args.time_range,
        args.feature_version,
        args.train_downsample,
        args.val_downsample,
        n_splits=args.n_splits,
        device=args.device,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
