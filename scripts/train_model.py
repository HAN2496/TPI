from src.configs.utils import create_model, load_config
from src.utils import prepare_training_data, ExperimentPaths
from src.utils.trainer import BaseTrainer, CombinationTrainer
from src.utils.arg_parser import train_model_parser
from src.utils.visualization import save_all_plots

def train_model(driver_name, model_name, time_range, downsample, device="cpu", tag=None, config_path=None, verbose=1):
    if config_path:
        config = load_config(None, None, config_path=config_path, verbose=verbose)
    else:
        config = load_config(driver_name, model_name, verbose=verbose)
    model = create_model(driver_name, model_name, is_train=True, device=device, verbose=verbose)

    is_combination = config['model_type'] == 'online_combination'
    flatten = is_combination

    train_loader, X_train, y_train, X_val, y_val = prepare_training_data(
        driver_name, config, time_range, downsample, flatten=flatten
    )

    paths = ExperimentPaths(driver_name, model_name, config['feature_version'], time_range, tag)

    if is_combination:
        trainer = CombinationTrainer(model, config['trainer'], best_model_path=paths.best_model)
        val_acc, val_auroc, history = trainer.fit(X_train, y_train, X_val, y_val, verbose=verbose)
    else:
        trainer = BaseTrainer(model, config['trainer'], best_model_path=paths.best_model)
        val_acc, val_auroc, history = trainer.fit(train_loader, X_val, y_val, verbose=verbose)

    if verbose==1:
        print(f"Final - Val Acc: {val_acc:.4f}, Val AUROC: {val_auroc:.4f}")

    is_online = config['model_type'] == 'online_mlp'
    save_all_plots(model, history, X_val, y_val, paths, is_online=is_online, is_combination=is_combination)

    if verbose:
        print(f"Plots saved to: {paths.run_dir}")

    return model, history


def main():
    parser = train_model_parser()
    args = parser.parse_args()

    train_model(args.driver_name,
                args.model_name,
                args.time_range,
                args.downsample,
                device=args.device,
                verbose=args.verbose)

if __name__ == "__main__":
    main()