import torch
from src.configs.utils import create_model, load_config, load_features
from src.utils import ExperimentPaths, prepare_training_data, convert_driver_name
from src.utils.trainer import BaseTrainer, CombinationTrainer
from src.utils.visualization import save_all_plots

def train_model(driver_name, model_name, time_range, feature_version, train_downsample=1, val_downsample=1, device="cpu", tag=None, config_path=None, verbose=1):
    driver_name = convert_driver_name(driver_name)
    if config_path:
        config = load_config(driver_name, model_name, config_path=config_path, verbose=verbose)
    else:
        config = load_config(driver_name, model_name, verbose=verbose)
    model = create_model(driver_name, model_name, feature_version, is_train=True, device=device, verbose=verbose)

    is_combination = config['model_type'] == 'online_combination'
    feature_cols = load_features(feature_version)

    train_loader, val_loader = prepare_training_data(driver_name, feature_version, config, time_range, train_downsample, val_downsample)

    paths = ExperimentPaths(driver_name, model_name, feature_version, time_range, tag)

    if is_combination:
        trainer = CombinationTrainer(model, config['trainer'], best_model_path=paths.best_model)
        val_acc, val_auroc, history = trainer.train(train_loader, val_loader, verbose=verbose)
    else:
        trainer = BaseTrainer(model, config['trainer'], best_model_path=paths.best_model, device=device)
        val_acc, val_auroc, history = trainer.train(train_loader, val_loader, verbose=verbose)

        model.load_state_dict(torch.load(paths.best_model, weights_only=True))
        model.to('cpu')

    if verbose==1:
        print(f"Final - Val Acc: {val_acc:.4f}, Val AUROC: {val_auroc:.4f}")

    is_online = config['model_type'] == 'online_mlp' or config['model_type'] == 'online_combination'
    save_all_plots(model, history, val_loader, paths, is_online=is_online, feature_cols=feature_cols)

    if verbose:
        print(f"Plots saved to: {paths.run_dir}")

    return model, history

# uv run python .\scripts\train_model.py -d 강신길 -m online_mlp -t "[5,7]" -f v1 -td 5 --device cuda --verbose 1
# uv run python .\scripts\train_model.py -d 강신길 -m online_mlp -t "[5,7]" -f v1 -td 5 -vd 1 --device cuda --verbose 1
# uv run python .\scripts\train_model.py -d 강신길 -m online_combi_quad_only -t "[5,7]" -f v1 -td 5 --verbose 1

def main():
    from src.utils.arg_parser import train_model_parser
    parser = train_model_parser()
    args = parser.parse_args()

    train_model(args.driver_name,
                args.model_name,
                args.time_range,
                args.feature_version,
                args.train_downsample,
                args.val_downsample,
                device=args.device,
                verbose=args.verbose)

if __name__ == "__main__":
    main()