"""
Explain trained TPI models using TimeSHAP.

This script:
1. Loads a trained model
2. Prepares background dataset from training data
3. Computes TimeSHAP explanations for test instances
4. Generates visualizations

Usage:
    uv run python scripts/explain_model.py -d 강신길 -mt online_lstm -mn base -t "[5,7]" --n_samples 5 --n_background 50
"""

import argparse
import os
import torch
import numpy as np
from pathlib import Path

from src.configs.utils import create_model, load_config
from src.configs.registries import MODELS
from src.model.base import RegressionModel
from src.utils import ExperimentPaths, prepare_training_data, convert_driver_name
from src.xai import TimeSHAPExplainer, create_background_dataset, plot_all_timeshap


def explain_model(
    driver_name,
    model_type,
    model_name='base',
    time_range=None,
    downsample=1,
    device='cpu',
    tag=None,
    n_samples=5,
    n_background=50,
    pruning_tol=0.01,
    verbose=1
):
    """
    Explain a trained model using TimeSHAP.

    Args:
        driver_name: Driver name
        model_type: Model type (online_lstm, offline_lstm, etc.)
        model_name: Model configuration name
        time_range: Time range for data
        downsample: Downsampling rate
        device: Device to run on
        tag: Experiment tag
        n_samples: Number of test samples to explain
        n_background: Number of background samples
        pruning_tol: Pruning tolerance for TimeSHAP
        verbose: Verbosity level
    """
    driver_name = convert_driver_name(driver_name)
    config = load_config(driver_name, model_type, model_name, verbose=verbose)
    paths = ExperimentPaths(driver_name, model_type, model_name, time_range, tag)

    # Check if model exists
    if not os.path.exists(paths.best_model):
        raise FileNotFoundError(f"Model not found at {paths.best_model}. Train the model first.")

    # Create empty model
    model = create_model(driver_name, model_type, model_name, is_train=True, device=device, verbose=verbose)
    is_regression = issubclass(MODELS[config['model_type']], RegressionModel)

    # Load trained weights
    if verbose > 0:
        print(f"Loading model weights from {paths.best_model}")

    if is_regression:
        # Load regression model
        state = torch.load(paths.best_model, weights_only=False, map_location='cpu')
        model.model = state['model']
        model.scaler = state['scaler']
        model.best_threshold = state.get('best_threshold', 0.5)
    else:
        # Load neural network model
        state_dict = torch.load(paths.best_model, weights_only=True, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

    # Prepare data
    if verbose > 0:
        print(f"Preparing data...")

    train_loader, val_loader = prepare_training_data(
        driver_name, config, time_range, downsample, downsample
    )

    # Extract numpy arrays from dataloaders
    train_X, train_y = [], []
    for X, y in train_loader:
        train_X.append(X.cpu().numpy())
        train_y.append(y.cpu().numpy())
    train_X = np.concatenate(train_X, axis=0)
    train_y = np.concatenate(train_y, axis=0)

    val_X, val_y = [], []
    for X, y in val_loader:
        val_X.append(X.cpu().numpy())
        val_y.append(y.cpu().numpy())
    val_X = np.concatenate(val_X, axis=0)
    val_y = np.concatenate(val_y, axis=0)

    if verbose > 0:
        print(f"  - Train: {train_X.shape}, Val: {val_X.shape}")

    # Create background dataset
    if verbose > 0:
        print(f"Creating background dataset ({n_background} samples)...")

    background_data = create_background_dataset(
        train_X,
        n_samples=n_background,
        strategy='random'
    )

    # Determine model type
    if 'online' in model_type.lower():
        xai_model_type = 'online'
    else:
        xai_model_type = 'offline'

    # Get feature names
    feature_names = config.get('features', None)

    # Create explainer
    if verbose > 0:
        print(f"Creating TimeSHAP explainer (model_type={xai_model_type})...")

    explainer = TimeSHAPExplainer(
        model,
        background_data,
        model_type=xai_model_type,
        device=device,
        feature_names=feature_names
    )

    # Create output directory
    explain_dir = paths.run_dir / 'explanations'
    explain_dir.mkdir(exist_ok=True, parents=True)

    if verbose > 0:
        print(f"Saving explanations to {explain_dir}")

    # Explain samples
    pruning_dict = {'tol': pruning_tol}

    for idx in range(min(n_samples, len(val_X))):
        if verbose > 0:
            print(f"\nExplaining sample {idx+1}/{n_samples} (label={val_y[idx]})...")

        instance = val_X[idx]  # (T, F)

        # Compute all explanations
        explanation = explainer.explain_instance(
            instance,
            pruning_dict=pruning_dict,
            verbose=(verbose > 1)
        )

        if verbose > 0:
            print(f"  - Prediction: {explanation['prediction']:.4f}")
            print(f"  - Top 3 important features: {np.argsort(explanation['feature_scores'])[::-1][:3]}")
            print(f"  - Top 3 important timesteps: {np.argsort(explanation['event_scores'])[::-1][:3]}")

        # Save plots
        sample_dir = explain_dir / f'sample_{idx}_label_{val_y[idx]}'
        sample_dir.mkdir(exist_ok=True)

        if verbose > 0:
            print(f"  - Generating plots...")

        figures = plot_all_timeshap(
            explanation,
            instance,
            feature_names=feature_names,
            save_dir=str(sample_dir)
        )

        # Save explanation data
        np.savez(
            sample_dir / 'explanation_data.npz',
            instance=instance,
            label=val_y[idx],
            prediction=explanation['prediction'],
            event_scores=explanation['event_scores'],
            feature_scores=explanation['feature_scores'],
            cell_scores=explanation['cell_scores'],
            feature_names=feature_names
        )

        if verbose > 0:
            print(f"  - Saved to {sample_dir}")

    if verbose > 0:
        print(f"\nExplanations complete! Results saved to {explain_dir}")

    return explainer, explain_dir


def main():
    parser = argparse.ArgumentParser(description='Explain TPI models using TimeSHAP')

    # Model arguments
    parser.add_argument('-d', '--driver_name', type=str, required=True,
                        help='Driver name')
    parser.add_argument('-mt', '--model_type', type=str, required=True,
                        help='Model type (e.g., online_lstm, offline_lstm)')
    parser.add_argument('-mn', '--model_name', type=str, default='base',
                        help='Model configuration name')
    parser.add_argument('-t', '--time_range', type=str, default=None,
                        help='Time range as string (e.g., "[5,7]")')
    parser.add_argument('--downsample', type=int, default=1,
                        help='Downsampling rate')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run on (cpu or cuda)')
    parser.add_argument('--tag', type=str, default=None,
                        help='Experiment tag')

    # Explanation arguments
    parser.add_argument('--n_samples', type=int, default=5,
                        help='Number of test samples to explain')
    parser.add_argument('--n_background', type=int, default=50,
                        help='Number of background samples for TimeSHAP')
    parser.add_argument('--pruning_tol', type=float, default=0.01,
                        help='Pruning tolerance for TimeSHAP')

    # General arguments
    parser.add_argument('--verbose', type=int, default=1,
                        help='Verbosity level (0, 1, 2)')

    args = parser.parse_args()

    # Parse time_range
    time_range = None
    if args.time_range:
        import ast
        time_range = ast.literal_eval(args.time_range)

    # Run explanation
    explain_model(
        driver_name=args.driver_name,
        model_type=args.model_type,
        model_name=args.model_name,
        time_range=time_range,
        downsample=args.downsample,
        device=args.device,
        tag=args.tag,
        n_samples=args.n_samples,
        n_background=args.n_background,
        pruning_tol=args.pruning_tol,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
