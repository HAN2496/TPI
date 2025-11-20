from src.utils.optimizer import create_optimizer
from scripts.train_model import train_model

def optimize_model(driver_name, model_name, time_range, downsample, n_trials=100, device="cpu", tag="optuna", verbose=1):
    optimizer = create_optimizer(driver_name, model_name, time_range, downsample, device=device)
    study = optimizer.optimize(n_trials=n_trials, tag=tag)

    print(f"\nOptimization completed!")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best params: {study.best_params}")

    print(f"\nTraining final model with best configuration...")
    best_config_path = optimizer.base_config['feature_version']
    best_config_path = f"artifacts/{driver_name}/{model_name}/{best_config_path}_t{time_range[0]}-{time_range[1]}_{tag}/best_config.yaml"

    train_model(driver_name, model_name, time_range, downsample,
                device=device, tag=tag, config_path=best_config_path, verbose=1)

    return study

def main():
    from src.utils.arg_parser import optimize_model_parser
    parser = optimize_model_parser()
    args = parser.parse_args()

    optimize_model(
        args.driver_name,
        args.model_name,
        args.time_range,
        args.downsample,
        n_trials=args.n_trials,
        device=args.device,
        tag=args.tag,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()
