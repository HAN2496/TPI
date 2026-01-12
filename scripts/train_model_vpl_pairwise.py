import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils.vpl_pairwise_dataset import create_vpl_pairwise_dataset, VPLPairwiseDataset
from src.model.vpl.vae_pairwise import VAEPairwiseModel
from src.model.vpl.trainer_pairwise import VPLPairwiseTrainer
from src.model.vpl.trainer import Annealer
from src.model.vpl.inference import (
    infer_driver_latents_pairwise,
    save_driver_latents,
    infer_and_predict_from_binary_feedback
)
from src.utils.paths import ExperimentPaths
from src.utils.data_loader import DatasetManager
from src.utils.visualization import plot_vpl_training_curves, plot_driver_latent_space


def load_features_yaml():
    import yaml
    features_path = Path(__file__).parent.parent / 'src' / 'configs' / 'features.yaml'
    with open(features_path, 'r', encoding='utf-8') as f:
        features_dict = yaml.safe_load(f)
    return features_dict


def main():
    parser = argparse.ArgumentParser(description='Train VPL Pairwise model')
    parser.add_argument('--drivers', nargs='+', required=True,
                        help='List of driver names or "all" for all drivers')
    parser.add_argument('--test-drivers', nargs='*', default=[],
                        help='Test drivers to exclude from training')
    parser.add_argument('--time_range', type=json.loads, default='[5,7]',
                        help='Time range as JSON list')
    parser.add_argument('--downsample', type=int, default=1,
                        help='Downsampling factor')
    parser.add_argument('--features', nargs='+', default=None,
                        help='List of features, or use --feature_version instead')
    parser.add_argument('--feature_version', default='v1', choices=['v0', 'v0_1', 'v1', 'v2'],
                        help='Feature version from features.yaml')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
                        help='Device to use')
    parser.add_argument('--set_len', type=int, default=10,
                        help='Number of pairs per query (annotation size)')
    parser.add_argument('--max_pairs_per_true', type=int, default=10,
                        help='Max False episodes paired with each True episode')
    parser.add_argument('--tie_ratio', type=float, default=0.0,
                        help='Ratio of tie pairs to add (default 0.25 = 20%)')

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

    # Binary feedback inference
    parser.add_argument('--test_binary_inference', action='store_true',
                        help='Test binary feedback inference on test drivers')
    parser.add_argument('--z_estimation_ratio', type=float, default=0.5,
                        help='Ratio of episodes for z estimation (rest for prediction)')

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

    # 3. Create pairwise dataset
    print(f"\nCreating VPL pairwise dataset...")
    train_dataset_dict, train_driver_map = create_vpl_pairwise_dataset(
        driver_names=train_drivers,
        time_range=args.time_range,
        downsample=args.downsample,
        features=features,
        set_len=args.set_len,
        max_pairs_per_true=args.max_pairs_per_true,
        tie_ratio=args.tie_ratio
    )

    # 4. Create test driver dataset (pairwise format for inference)
    test_dataset_dict = None
    test_driver_map = {}
    if args.test_drivers:
        print(f"\nCreating test driver dataset (pairwise format for inference)...")
        test_dataset_dict, test_driver_map = create_vpl_pairwise_dataset(
            driver_names=args.test_drivers,
            time_range=args.time_range,
            downsample=args.downsample,
            features=features,
            set_len=args.set_len,
            max_pairs_per_true=args.max_pairs_per_true,
            tie_ratio=args.tie_ratio
        )

    # 5. Train/val split
    n_total = len(train_dataset_dict['obs_a'])
    n_val = max(1, int(n_total * args.test_size))
    n_train = n_total - n_val

    indices = np.random.permutation(n_total)
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    train_dataset = VPLPairwiseDataset({
        k: v[train_idx] for k, v in train_dataset_dict.items()
    })
    val_dataset = VPLPairwiseDataset({
        k: v[val_idx] for k, v in train_dataset_dict.items()
    })

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset),
                           shuffle=False, num_workers=0, pin_memory=True)

    # 6. Calculate dimensions
    _, set_len, T, d = train_dataset_dict['obs_a'].shape
    encoder_input = set_len * (2 * T * d + 1)  # Pairwise: 2 episodes + 1 preference
    decoder_input = d + args.latent_dim
    reward_scaling = args.reward_scaling if args.reward_scaling else T

    print(f"\nTraining dataset shape: obs_a={train_dataset_dict['obs_a'].shape}")
    print(f"Encoder input dim: {encoder_input} (set_len * (2 * T * d + 1) = {set_len} * (2 * {T} * {d} + 1))")
    print(f"Decoder input dim: {decoder_input}")
    print(f"set_len={set_len}, T={T} timesteps, d={d} features")
    print(f"Total {n_total} queries, {n_train} train, {n_val} val")

    # 7. Initialize VAE pairwise model
    annealer = None
    if args.use_annealing:
        annealer = Annealer(
            total_steps=args.epochs // args.annealer_cycles,
            shape=args.annealer_type,
            baseline=args.annealer_baseline,
            cyclical=args.annealer_cycles > 1
        )

    model = VAEPairwiseModel(
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
        model_type='vpl_pairwise',
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

    trainer = VPLPairwiseTrainer(
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

    # 11. Infer driver latents (test drivers using pairwise format)
    if test_dataset_dict:
        print("\nInferring test driver latents (using pairwise format)...")
        driver_latents = infer_driver_latents_pairwise(
            model, test_dataset_dict, test_driver_map, args.device
        )
        save_driver_latents(driver_latents, paths.run_dir / 'driver_latents.pkl')
        print(f"Saved driver latents to: {paths.run_dir / 'driver_latents.pkl'}")

    # 12. Create visualizations
    print("\nCreating visualizations...")

    # Training curves
    plot_vpl_training_curves(history, save_path=paths.run_dir / 'training_curves.png')
    print(f"Saved training curves to: {paths.run_dir / 'training_curves.png'}")

    # Latent space visualization
    if test_dataset_dict:
        plot_driver_latent_space(driver_latents, save_path=paths.run_dir / 'latent_space.png')
        print(f"Saved latent space plot to: {paths.run_dir / 'latent_space.png'}")

    print(f"\nAll results saved to: {paths.run_dir}")
    if test_dataset_dict:
        print(f"\nDriver latents summary:")
        for driver_name, latent_info in driver_latents.items():
            print(f"  {driver_name}: z_mean shape = {latent_info['z_mean_point'].shape}")

    # 13. Test binary feedback inference (if requested)
    if args.test_binary_inference and args.test_drivers:
        print("\n" + "=" * 70)
        print("Testing Binary Feedback Inference")
        print("=" * 70)

        from src.utils.utils import _load_dataset_sequences

        for test_driver in args.test_drivers:
            print(f"\nDriver: {test_driver}")
            print(f"{'='*70}")

            # Load binary feedback episodes
            X, y = _load_dataset_sequences(
                test_driver, args.time_range, args.downsample,
                {'features': features}
            )

            print(f"Loaded {len(X)} episodes ({(y==1).sum()} True, {(y==0).sum()} False)")

            # Run binary feedback inference
            print(f"\nSplitting: {args.z_estimation_ratio*100:.0f}% for z estimation, "
                  f"{(1-args.z_estimation_ratio)*100:.0f}% for reward prediction")

            z_estimated, predictions, pred_indices = infer_and_predict_from_binary_feedback(
                model, X, y,
                z_estimation_ratio=args.z_estimation_ratio,
                set_len=args.set_len,
                device=args.device
            )

            # Analyze results
            print(f"\nResults:")
            print(f"  Estimated z shape: {z_estimated.shape}")
            print(f"  Predictions shape: {predictions.shape}")

            mean_rewards = predictions.mean(axis=1)
            print(f"\nReward statistics:")
            print(f"  Mean: {mean_rewards.mean():.4f} ± {mean_rewards.std():.4f}")
            print(f"  Range: [{mean_rewards.min():.4f}, {mean_rewards.max():.4f}]")

            # Compare with actual labels
            actual_labels = y[pred_indices]
            true_rewards = mean_rewards[actual_labels == 1]
            false_rewards = mean_rewards[actual_labels == 0]

            if len(true_rewards) > 0 and len(false_rewards) > 0:
                print(f"\nLabel correlation:")
                print(f"  True episodes: {true_rewards.mean():.4f} ± {true_rewards.std():.4f}")
                print(f"  False episodes: {false_rewards.mean():.4f} ± {false_rewards.std():.4f}")
                print(f"  Difference: {true_rewards.mean() - false_rewards.mean():.4f}")

            # Classification metrics
            from sklearn.metrics import (
                roc_auc_score, roc_curve,
                accuracy_score, precision_score, recall_score, f1_score
            )
            import matplotlib.pyplot as plt

            print(f"\nClassification metrics (using rewards as scores):")

            # AUROC (threshold-free)
            if len(np.unique(actual_labels)) > 1:
                auroc = roc_auc_score(actual_labels, mean_rewards)
                print(f"  AUROC: {auroc:.4f}")

                # Plot ROC curve
                fpr, tpr, thresholds = roc_curve(actual_labels, mean_rewards)

                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUROC = {auroc:.4f})')
                plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random classifier')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate', fontsize=12)
                plt.ylabel('True Positive Rate', fontsize=12)
                plt.title(f'ROC Curve - {test_driver}\nBinary Feedback Inference', fontsize=14)
                plt.legend(loc="lower right", fontsize=11)
                plt.grid(alpha=0.3)
                plt.tight_layout()

                roc_save_path = paths.run_dir / f'roc_curve_{test_driver}.png'
                plt.savefig(roc_save_path, dpi=150)
                plt.close()
                print(f"  Saved ROC curve to: {roc_save_path}")
            else:
                print(f"  AUROC: N/A (only one class in prediction set)")
                auroc = None

            # Threshold-based metrics (use median reward as threshold)
            threshold = np.median(mean_rewards)
            pred_labels = (mean_rewards > threshold).astype(int)

            accuracy = accuracy_score(actual_labels, pred_labels)
            precision = precision_score(actual_labels, pred_labels, zero_division=0)
            recall = recall_score(actual_labels, pred_labels, zero_division=0)
            f1 = f1_score(actual_labels, pred_labels, zero_division=0)

            print(f"  Accuracy: {accuracy:.4f} (threshold={threshold:.4f})")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 Score: {f1:.4f}")

            # Plot reward distribution
            if len(true_rewards) > 0 and len(false_rewards) > 0:
                plt.figure(figsize=(10, 6))

                # Histogram
                plt.subplot(1, 2, 1)
                bins = np.linspace(mean_rewards.min(), mean_rewards.max(), 30)
                plt.hist(true_rewards, bins=bins, alpha=0.6, label='True episodes', color='green', edgecolor='black')
                plt.hist(false_rewards, bins=bins, alpha=0.6, label='False episodes', color='red', edgecolor='black')
                plt.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold={threshold:.3f}')
                plt.xlabel('Mean Reward', fontsize=11)
                plt.ylabel('Count', fontsize=11)
                plt.title('Reward Distribution', fontsize=12)
                plt.legend(fontsize=10)
                plt.grid(alpha=0.3)

                # Box plot
                plt.subplot(1, 2, 2)
                plt.boxplot([false_rewards, true_rewards], labels=['False', 'True'], patch_artist=True,
                           boxprops=dict(facecolor='lightblue', alpha=0.6),
                           medianprops=dict(color='red', linewidth=2))
                plt.ylabel('Mean Reward', fontsize=11)
                plt.title('Reward by Label', fontsize=12)
                plt.grid(alpha=0.3, axis='y')

                plt.suptitle(f'Binary Feedback Inference - {test_driver}', fontsize=14, y=1.02)
                plt.tight_layout()

                dist_save_path = paths.run_dir / f'reward_distribution_{test_driver}.png'
                plt.savefig(dist_save_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  Saved reward distribution to: {dist_save_path}")

            # Save results
            binary_results = {
                'driver_name': test_driver,
                'z_estimated': z_estimated,
                'predictions': predictions,
                'prediction_indices': pred_indices,
                'mean_rewards': mean_rewards,
                'actual_labels': actual_labels,
                'metrics': {
                    'auroc': auroc,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'threshold': threshold
                }
            }

            # Add ROC curve data if available
            if len(np.unique(actual_labels)) > 1:
                binary_results['roc_curve'] = {
                    'fpr': fpr,
                    'tpr': tpr,
                    'thresholds': thresholds
                }

            import pickle
            save_path = paths.run_dir / f'binary_inference_{test_driver}.pkl'
            with open(save_path, 'wb') as f:
                pickle.dump(binary_results, f)
            print(f"\nSaved binary inference results to: {save_path}")

        print("\n" + "=" * 70)
        print("Binary feedback inference testing complete!")
        print("=" * 70)


if __name__ == '__main__':
    main()
