import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils.vpl_dataset import create_vpl_dataset, VPLDataset
from src.model.vpl.vae import VAEModel
from src.model.vpl.trainer import VPLTrainer, Annealer
from src.model.vpl.inference import infer_driver_latents, save_driver_latents, evaluate_test_drivers
from src.utils.paths import ExperimentPaths
from src.utils.data_loader import DatasetManager
from src.utils.visualization import plot_vpl_training_curves, plot_driver_latent_space, plot_vpl_test_evaluation
from sklearn.metrics import roc_auc_score


def load_features_yaml():
    """Load features from features.yaml"""
    import yaml
    features_path = Path(__file__).parent.parent / 'src' / 'configs' / 'features.yaml'
    with open(features_path, 'r', encoding='utf-8') as f:
        features_dict = yaml.safe_load(f)
    return features_dict


def main():
    parser = argparse.ArgumentParser(description='Train VPL model for multi-driver reward learning')
    parser.add_argument('--drivers', nargs='+', required=True,
                        help='List of driver names or "all" for all drivers (e.g., 강신길 박재일 이지환 or all)')
    parser.add_argument('--test-drivers', nargs='*', default=[],
                        help='Test drivers to exclude from training but include in visualization')
    parser.add_argument('--time_range', type=json.loads, default='[5,7]',
                        help='Time range as JSON list, e.g., "[5,7]"')
    parser.add_argument('--downsample', type=int, default=1,
                        help='Downsampling factor')
    parser.add_argument('--features', nargs='+', default=None,
                        help='List of features, or use --feature_version instead')
    parser.add_argument('--feature_version', default='v1', choices=['v0', 'v0_1', 'v1', 'v2'],
                        help='Feature version from features.yaml')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
                        help='Device to use')
    parser.add_argument('--set_len', type=int, default=1,
                        help='Number of episodes per query (annotation size)')

    # Model hyperparameters
    parser.add_argument('--latent_dim', type=int, default=32,
                        help='Latent dimension')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension')
    parser.add_argument('--kl_weight', type=float, default=1.0,
                        help='KL divergence weight')
    parser.add_argument('--reward_scaling', type=float, default=None,
                        help='Reward scaling factor (default: T)')

    # Training hyperparameters
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--test_size', type=float, default=0.1,
                        help='Validation set ratio')

    # Annealing
    parser.add_argument('--use_annealing', action='store_true',
                        help='Use KL annealing')
    parser.add_argument('--annealer_type', default='cosine', choices=['linear', 'cosine', 'logistic'],
                        help='Annealer schedule type')
    parser.add_argument('--annealer_cycles', type=int, default=4,
                        help='Number of annealing cycles')
    parser.add_argument('--annealer_baseline', type=float, default=0.0,
                        help='Baseline annealing weight')

    # Early stopping
    parser.add_argument('--early_stop', action='store_true',
                        help='Use early stopping')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--min_delta', type=float, default=3e-4,
                        help='Minimum delta for early stopping')

    parser.add_argument('--verbose', type=int, default=1,
                        help='Verbosity level')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. Load driver names
    if args.drivers == ['all']:
        dataset_manager = DatasetManager()
        all_drivers = list(dataset_manager.keys())
        train_drivers = [d for d in all_drivers if d not in args.test_drivers]
        print(f"Loading all drivers: {all_drivers}")
        print(f"Training drivers: {train_drivers}")
        if args.test_drivers:
            print(f"Test drivers (excluded from training): {args.test_drivers}")
    else:
        train_drivers = [d for d in args.drivers if d not in args.test_drivers]
        print(f"Training drivers: {train_drivers}")
        if args.test_drivers:
            print(f"Test drivers (excluded from training): {args.test_drivers}")

    if not train_drivers:
        raise ValueError("No training drivers available after excluding test drivers!")

    # 2. Load features
    if args.features:
        features = args.features
    else:
        features_dict = load_features_yaml()
        features = features_dict[args.feature_version]

    print(f"\nUsing {len(features)} features: {features}")

    # 3. Create VPL dataset (training only)
    print(f"\nCreating VPL training dataset...")
    train_dataset_dict, train_driver_map = create_vpl_dataset(
        driver_names=train_drivers,
        time_range=args.time_range,
        downsample=args.downsample,
        features=features,
        set_len=args.set_len
    )

    # 4. Create test driver dataset (for visualization only)
    test_dataset_dict = None
    test_driver_map = {}
    if args.test_drivers:
        print(f"\nCreating test driver dataset for visualization...")
        test_dataset_dict, test_driver_map = create_vpl_dataset(
            driver_names=args.test_drivers,
            time_range=args.time_range,
            downsample=args.downsample,
            features=features,
            set_len=args.set_len
        )

    # Combine datasets for inference (but only train on train_dataset)
    combined_dataset = {
        'observations': train_dataset_dict['observations'].copy(),
        'labels': train_dataset_dict['labels'].copy(),
        'driver_ids': train_dataset_dict['driver_ids'].copy()
    }
    combined_driver_map = train_driver_map.copy()

    if test_dataset_dict is not None:
        next_driver_id = max(combined_driver_map.keys()) + 1
        test_driver_ids_remapped = test_dataset_dict['driver_ids'] + next_driver_id

        combined_dataset['observations'] = np.concatenate([
            combined_dataset['observations'],
            test_dataset_dict['observations']
        ])
        combined_dataset['labels'] = np.concatenate([
            combined_dataset['labels'],
            test_dataset_dict['labels']
        ])
        combined_dataset['driver_ids'] = np.concatenate([
            combined_dataset['driver_ids'],
            test_driver_ids_remapped
        ])

        for old_id, driver_name in test_driver_map.items():
            new_id = old_id + next_driver_id
            combined_driver_map[new_id] = driver_name

    # 5. Train/val split (only on training data, driver ID 안 씀)
    n_total = len(train_dataset_dict['observations'])
    n_val = max(1, int(n_total * args.test_size))
    n_train = n_total - n_val

    indices = np.random.permutation(n_total)
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    train_dataset = VPLDataset({
        k: v[train_idx] for k, v in train_dataset_dict.items()
    })
    val_dataset = VPLDataset({
        k: v[val_idx] for k, v in train_dataset_dict.items()
    })

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset),
                           shuffle=False, num_workers=0, pin_memory=True)

    # 6. Calculate dimensions
    _, set_len, T, d = train_dataset_dict['observations'].shape
    encoder_input = set_len * (T * d + 1)
    decoder_input = d + args.latent_dim
    reward_scaling = args.reward_scaling if args.reward_scaling else T

    print(f"\nTraining dataset shape: {train_dataset_dict['observations'].shape}")
    if test_dataset_dict is not None:
        print(f"Test dataset shape: {test_dataset_dict['observations'].shape}")
    print(f"Combined dataset shape: {combined_dataset['observations'].shape}")
    print(f"Encoder input dim: {encoder_input}")
    print(f"Decoder input dim: {decoder_input}")
    print(f"set_len={set_len}, T={T} timesteps, d={d} features")
    print(f"Total {n_total} queries, {n_train} train, {n_val} val")

    # 7. Initialize VAE model
    annealer = None
    if args.use_annealing:
        annealer = Annealer(
            total_steps=args.epochs // args.annealer_cycles,
            shape=args.annealer_type,
            baseline=args.annealer_baseline,
            cyclical=args.annealer_cycles > 1
        )

    model = VAEModel(
        encoder_input=encoder_input,
        decoder_input=decoder_input,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        annotation_size=set_len,
        size_segment=T,
        kl_weight=args.kl_weight,
        annealer=annealer,
        reward_scaling=reward_scaling
    ).to(args.device)

    print(f"\nModel initialized with {sum(p.numel() for p in model.parameters())} parameters")

    # 8. Setup experiment paths
    if args.drivers == ['all']:
        tag = f"all_{len(train_drivers)}drivers"
    else:
        tag = '_'.join(train_drivers[:3])
        if len(train_drivers) > 3:
            tag += f'_plus{len(train_drivers)-3}'

    if args.test_drivers:
        tag += f"_test_{len(args.test_drivers)}"

    paths = ExperimentPaths(
        driver_name='multi_driver',
        model_type='vpl',
        model_name='base',
        time_range=args.time_range,
        tag=tag
    )
    paths.run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_dict = vars(args)
    config_dict['train_driver_map'] = train_driver_map
    if test_driver_map:
        config_dict['test_driver_map'] = test_driver_map
    config_dict['combined_driver_map'] = combined_driver_map
    with open(paths.run_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    # 9. Train
    config = {
        'learning_rate': args.lr,
        'weight_decay': 0.0,
        'epochs': args.epochs,
        'early_stop': args.early_stop,
        'patience': args.patience,
        'min_delta': args.min_delta
    }

    trainer = VPLTrainer(
        model, config,
        best_model_path=paths.best_model,
        device=args.device
    )

    print("\nStarting training...")
    history, val_accuracy = trainer.train(
        train_loader, val_loader,
        epochs=args.epochs,
        verbose=args.verbose
    )

    # 10. Save training history
    torch.save(history, paths.run_dir / 'history.pt')

    print(f"\nTraining complete!")
    print(f"Val Accuracy: {val_accuracy:.4f}")

    # 11. Infer driver latents (train + test drivers)
    print("\nInferring driver latents...")
    driver_latents = infer_driver_latents(
        model,
        combined_dataset,
        combined_driver_map,
        args.device
    )

    # Save driver latents
    save_driver_latents(driver_latents, paths.run_dir / 'driver_latents.pkl')
    print(f"Saved driver latents to: {paths.run_dir / 'driver_latents.pkl'}")

    # 12. Create visualizations
    print("\nCreating visualizations...")

    # Training curves
    plot_vpl_training_curves(history, save_path=paths.run_dir / 'training_curves.png')
    print(f"Saved training curves to: {paths.run_dir / 'training_curves.png'}")

    # Latent space visualization
    plot_driver_latent_space(driver_latents, save_path=paths.run_dir / 'latent_space.png')
    print(f"Saved latent space plot to: {paths.run_dir / 'latent_space.png'}")

    # 13. Evaluate test drivers
    if args.test_drivers and test_dataset_dict is not None:
        print("\n" + "="*60)
        print("Evaluating Test Drivers")
        print("="*60)

        # Load best model for evaluation
        model.load_state_dict(torch.load(paths.best_model, weights_only=True))
        model.eval()

        # Evaluate test drivers
        test_results = evaluate_test_drivers(model, test_dataset_dict, test_driver_map, args.device)

        # Compute AUROC for each test driver
        for driver_name, results in test_results.items():
            auroc = roc_auc_score(results['y_true'], results['y_probs'])
            test_results[driver_name]['auroc'] = auroc
            print(f"\n{driver_name}:")
            print(f"  Test AUROC: {auroc:.4f}")
            print(f"  Total episodes: {len(results['y_true'])}")

        # Save test evaluation plots
        print("\nCreating test evaluation visualizations...")
        plot_vpl_test_evaluation(test_results, paths.run_dir)
        print(f"Saved test evaluation plots to: {paths.run_dir}")

    print(f"\nAll results saved to: {paths.run_dir}")
    print(f"\nDriver latents summary:")
    for driver_name, latent_info in driver_latents.items():
        is_test = driver_name in args.test_drivers
        marker = " [TEST]" if is_test else ""
        print(f"  {driver_name}{marker}: z_mean shape = {latent_info['z_mean_point'].shape}")


if __name__ == '__main__':
    main()
