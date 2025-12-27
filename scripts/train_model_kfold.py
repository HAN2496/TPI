import numpy as np
import torch
from src.configs.utils import create_model, load_config
from src.configs.registries import MODELS
from src.model import OnlineCombination
from src.utils import convert_driver_name, ExperimentPaths
from src.utils.utils import prepare_training_data_kfold
from src.utils.trainer import BaseTrainer, CombinationTrainer
from src.utils.visualization import save_all_plots, plot_kfold_roc_curves


def train_model_kfold(driver_name, model_type, model_name='base', time_range=None, train_downsample=1, val_downsample=None, n_splits=5, device="cpu", verbose=1):
    driver_name = convert_driver_name(driver_name)
    config = load_config(driver_name, model_type, model_name, verbose=verbose)
    is_combination = MODELS[config['model_type']] is OnlineCombination
    is_online = config['model_type'] == 'online_mlp' or is_combination

    if verbose:
        print(f"\n{'='*60}")
        print(f"K-Fold Cross-Validation ({n_splits} folds)")
        print(f"Driver: {driver_name}")
        print(f"Model Type: {model_type}")
        print(f"Model Name: {model_name}")
        print(f"Time range: {time_range}, Train downsample: {train_downsample}, Val downsample: {val_downsample or train_downsample}")
        print(f"{'='*60}\n")

    fold_results = {'acc': [], 'auroc': []}
    fold_predictions = []

    for fold_idx, train_loader, val_loader in prepare_training_data_kfold(
        driver_name, config, time_range, train_downsample, val_downsample, n_splits=n_splits
    ):
        if verbose:
            print(f"Fold {fold_idx + 1}/{n_splits} - Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

        model = create_model(driver_name, model_type, model_name, is_train=True, device=device, verbose=0)

        paths = ExperimentPaths(driver_name, model_type, model_name, time_range,
                                tag=f"kfold/fold_{fold_idx}", tag_as_subdir=True)

        if is_combination:
            trainer = CombinationTrainer(model, config['trainer'], best_model_path=None)
        else:
            trainer = BaseTrainer(model, config['trainer'], best_model_path=paths.best_model, device=device)

        val_acc, val_auroc, history = trainer.train(train_loader, val_loader, verbose=(verbose >= 2))

        if not is_combination:
            model.load_state_dict(torch.load(paths.best_model, weights_only=True))
            model.to('cpu')

        y_true, y_probs = save_all_plots(model, history, val_loader, paths, is_online=is_online, feature_cols=config['features'])

        fold_results['acc'].append(val_acc)
        fold_results['auroc'].append(val_auroc)
        fold_predictions.append((y_true, y_probs))

        if verbose:
            print(f"  - Acc: {val_acc:.4f}, AUROC: {val_auroc:.4f}")

    mean_acc = np.mean(fold_results['acc'])
    std_acc = np.std(fold_results['acc'])
    mean_auroc = np.mean(fold_results['auroc'])
    std_auroc = np.std(fold_results['auroc'])

    from sklearn.metrics import roc_auc_score
    y_true_all = np.concatenate([y_true for y_true, _ in fold_predictions])
    y_probs_all = np.concatenate([y_probs for _, y_probs in fold_predictions])
    oof_auroc = roc_auc_score(y_true_all, y_probs_all)

    kfold_paths = ExperimentPaths(driver_name, model_type, model_name, time_range, tag="kfold", tag_as_subdir=True)
    plot_kfold_roc_curves(fold_predictions, save_path=kfold_paths.get('kfold_roc_curves.png', create=True))

    if verbose:
        print(f"{'='*60}")
        print(f"K-Fold CV Results")
        print(f"{'='*60}")
        print(f"Mean Accuracy:  {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"Mean AUROC:     {mean_auroc:.4f} ± {std_auroc:.4f}")
        print(f"OOF AUROC:      {oof_auroc:.4f}")
        print(f"\nFold-wise AUROC: {[f'{x:.4f}' for x in fold_results['auroc']]}")
        print(f"K-Fold combined ROC curve saved to: {kfold_paths.get('kfold_roc_curves.png')}")
        print(f"{'='*60}\n")

    return fold_results


# uv run python .\scripts\train_model_kfold.py -d 강신길 -mt online_mlp -mn base -t "[5,7]" -td 5 --n-splits 5 --device cuda -v 1
# uv run python .\scripts\train_model_kfold.py -d 강신길 -mt online_lstm -mn base -t "[5,7]" -td 5 --n-splits 5 --device cuda - v 1
def main():
    from src.utils.arg_parser import train_model_parser
    parser = train_model_parser()
    parser.add_argument('--n-splits', type=int, default=5, help='Number of folds')
    args = parser.parse_args()

    train_model_kfold(
        args.driver_name,
        args.model_type,
        args.model_name,
        args.time_range,
        args.train_downsample,
        args.val_downsample,
        n_splits=args.n_splits,
        device=args.device,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
