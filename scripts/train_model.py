import torch
from src.configs.utils import create_model, load_config
from src.configs.registries import MODELS
from src.model.base import RegressionModel
from src.utils import ExperimentPaths, prepare_training_data, convert_driver_name
from src.utils.trainer import BaseTrainer, RegressionTrainer
from src.utils.visualization import save_all_plots

def train_model(driver_name, model_type, model_name='base', time_range=None, train_downsample=1, val_downsample=1, device="cpu", tag=None, verbose=1):
    driver_name = convert_driver_name(driver_name)
    config = load_config(driver_name, model_type, model_name, verbose=verbose)
    model = create_model(driver_name, model_type, model_name, is_train=True, device=device, verbose=verbose)

    is_regression = issubclass(MODELS[config['model_type']], RegressionModel)

    train_loader, val_loader = prepare_training_data(driver_name, config, time_range, train_downsample, val_downsample)

    paths = ExperimentPaths(driver_name, model_type, model_name, time_range, tag)

    if is_regression:
        trainer = RegressionTrainer(model, config['trainer'], best_model_path=paths.best_model)
        val_acc, val_auroc, history = trainer.train(train_loader, val_loader, verbose=verbose)
    else:
        trainer = BaseTrainer(model, config['trainer'], best_model_path=paths.best_model, device=device)
        val_acc, val_auroc, history = trainer.train(train_loader, val_loader, verbose=verbose)

        model.load_state_dict(torch.load(paths.best_model, weights_only=True))
        model.to('cpu')

    if verbose > 0:
        print(f"  - Val Acc: {val_acc:.4f}, Val AUROC: {val_auroc:.4f}")

    is_online = config['model_type'] == 'online_mlp' or is_regression
    save_all_plots(model, history, val_loader, paths, is_online=is_online, is_regression=is_regression, feature_cols=config['features'], verbose=verbose)

    return model, history

# uv run python .\scripts\train_model.py -d 강신길 -mt online_mlp -mn base -t "[5,7]" -td 5 --device cuda --verbose 1
# uv run python .\scripts\train_model.py -d 강신길 -mt online_mlp -mn bo -t "[5,7]" -td 5 -vd 1 --device cuda --verbose 1
# uv run python .\scripts\train_model.py -d 강신길 -mt online_combi_quad_only -mn base -t "[5,7]" -td 5 --verbose 1

def main():
    from src.utils.arg_parser import train_model_parser
    parser = train_model_parser()
    args = parser.parse_args()

    train_model(args.driver_name,
                args.model_type,
                args.model_name,
                args.time_range,
                args.train_downsample,
                args.val_downsample,
                device=args.device,
                verbose=args.verbose)

if __name__ == "__main__":
    main()