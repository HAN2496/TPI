from src.utils.optimizer import create_optimizer
from src.utils import convert_driver_name, ExperimentPaths

from scripts.train_model import train_model

def optimize_model(driver_name, model_name, time_range, feature_version, downsample,
                   n_trials=100, use_feature_selection=False, device="cpu", tag="optuna", verbose=1):
    driver_name = convert_driver_name(driver_name)
    optimizer = create_optimizer(driver_name, model_name, time_range, feature_version, downsample,
                                 use_feature_selection=use_feature_selection, device=device)
    study, new_model_name = optimizer.optimize(n_trials=n_trials, tag=tag)

    if verbose > 0:
        print(f"\nOptimization completed!")
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best params: {study.best_params}")

        print(f"\nTraining final model with best configuration...")

    train_model(driver_name, new_model_name, time_range, feature_version, downsample, None,
                device=device, tag=tag, verbose=verbose)

    return study

# uv run python .\scripts\optimize_model.py -d 강신길 -m online_mlp -t "[5,7]" -f v1 --downsample 5 --n-trials 500 --device cuda --verbose 1
def main():
    from src.utils.arg_parser import optimize_model_parser
    parser = optimize_model_parser()
    args = parser.parse_args()

    tag = args.tag
    if args.use_feature_selection:
        tag = f"{tag}_fs"

    optimize_model(
        args.driver_name,
        args.model_name,
        args.time_range,
        args.feature_version,
        args.downsample,
        n_trials=args.n_trials,
        use_feature_selection=args.use_feature_selection,
        device=args.device,
        tag=tag,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()
