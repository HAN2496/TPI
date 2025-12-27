from src.optimizer import create_optimizer
from src.utils import convert_driver_name, ExperimentPaths

from scripts.train_model import train_model
from scripts.train_model_kfold import train_model_kfold

def optimize_model(driver_name, model_type, time_range, downsample, n_splits,
                   n_trials=100, use_feature_selection=False, device="cpu", optimizer_type="bayesian", tag="optuna", verbose=1):
    driver_name = convert_driver_name(driver_name)
    optimizer = create_optimizer(driver_name, model_type, time_range, downsample, n_splits,
                                 use_feature_selection=use_feature_selection, device=device, optimizer_type=optimizer_type)

    if optimizer_type == "bayesian":
        study, optimized_model_name = optimizer.optimize(n_trials=n_trials, tag=tag)
    else:
        study, optimized_model_name = optimizer.optimize(tag=tag)

    if verbose > 0:
        print(f"\nOptimization completed!")
        if study:
            print(f"Best trial: {study.best_trial.number}")
            print(f"Best params: {study.best_params}")

        print(f"\nTraining final model with best configuration...")

    # train_model(driver_name, model_type, optimized_model_name, time_range, downsample, None,
    #             device=device, tag=tag, verbose=verbose)

    train_model_kfold(driver_name, model_type, optimized_model_name, time_range, downsample, None,
                device=device, verbose=verbose)

    return study

# uv run python .\scripts\optimize_model.py -d 강신길 -mt online_mlp -t "[5,7]" --downsample 5 --n-splits 5 --n-trials 500 --device cuda --verbose 1
# uv run python .\scripts\optimize_model.py -d 강신길 -mt online_mlp -t "[5,7]" --downsample 5 --n-splits 5 --n-trials 500 --device cuda --verbose 1 --use-feature-selection --optimizer-type exhaustive
def main():
    from src.utils.arg_parser import optimize_model_parser
    parser = optimize_model_parser()
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
        use_feature_selection=args.use_feature_selection,
        device=args.device,
        optimizer_type=args.optimizer_type,
        tag=tag,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()
