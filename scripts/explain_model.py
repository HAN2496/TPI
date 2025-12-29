import torch
import numpy as np
from src.configs.utils import load_config
from src.configs.registries import MODELS
from src.utils import convert_driver_name, ExperimentPaths
from src.utils.utils import _load_dataset_sequences
from src.xai import (
    create_explainer,
    compute_baselines,
    compute_feature_importance, compute_temporal_importance, compute_feature_time_matrix,
    plot_feature_importance, plot_temporal_importance,
    plot_feature_time_heatmap, plot_top_contributions
)

def explain_with_baseline(explainer, sample_indices, X, config, paths, baseline, baseline_name, verbose):
    if verbose:
        print(f"\n{'='*70}")
        if baseline.ndim == 3:
            print(f"Baseline: {baseline_name} ({baseline.shape[0]} samples from negative distribution)")
        else:
            print(f"Baseline: {baseline_name}")
        print(f"{'='*70}\n")

    for i, idx in enumerate(sample_indices):
        x_sample = X[idx]

        if verbose:
            print(f"{'='*60}")
            print(f"Sample {i+1}/{len(sample_indices)} (Dataset Index: {idx})")
            print(f"{'='*60}")

        try:
            attributions = explainer.explain_sample(x_sample, baseline=baseline)

            feature_importance = compute_feature_importance(attributions)
            temporal_importance = compute_temporal_importance(attributions)
            feature_time_matrix = compute_feature_time_matrix(attributions)

            if verbose:
                print(f"\nFeature Importance:")
                for f_idx in range(len(config['features'])):
                    print(f"  {config['features'][f_idx]:20s}: {feature_importance[f_idx]:+.6f}")

                print(f"\nTemporal Importance:")
                print(f"  Mean: {np.mean(temporal_importance):.6f}")
                print(f"  Max:  {np.max(temporal_importance):.6f} at t={np.argmax(temporal_importance)}")
                print(f"  Min:  {np.min(temporal_importance):.6f} at t={np.argmin(temporal_importance)}")

            plot_feature_importance(
                feature_importance, config['features'],
                save_path=paths.get(f'feature_importance_sample_{i}.png', create=True)
            )

            plot_temporal_importance(
                temporal_importance,
                save_path=paths.get(f'temporal_importance_sample_{i}.png', create=True)
            )

            plot_feature_time_heatmap(
                feature_time_matrix, config['features'],
                x_sample=x_sample,
                save_path=paths.get(f'feature_time_heatmap_sample_{i}.png', create=True)
            )

            plot_top_contributions(
                feature_time_matrix, config['features'], top_k=10,
                save_path=paths.get(f'top_contributions_sample_{i}.png', create=True)
            )

            if verbose:
                print(f"\nVisualizations saved to: {paths.run_dir}")
                print()

        except Exception as e:
            print(f"Error explaining sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

def main(driver_name, model_type, model_name, time_range, downsample,
         n_samples=5, baseline_type='all', n_conditional_samples=50,
         method='ig', device='cpu', verbose=1):

    driver_name = convert_driver_name(driver_name)
    config = load_config(driver_name, model_type, model_name, verbose=verbose)

    method_names = {'ig': 'IntegratedGradients', 'kernelshap': 'KernelShap'}

    if verbose:
        print(f"\n{'='*60}")
        print(f"Captum {method_names.get(method, method)} Explanation Analysis")
        print(f"Driver: {driver_name}")
        print(f"Model: {model_type}/{model_name}")
        print(f"Time range: {time_range}, Downsample: {downsample}")
        print(f"Method: {method}")
        print(f"Baseline type: {baseline_type}")
        print(f"{'='*60}\n")

    args = config['args'].copy()
    if 'input_dim' not in args:
        args['input_dim'] = len(config['features'])
    model = MODELS[config['model_type']](**args)
    model = model.to(device)

    train_paths = ExperimentPaths(driver_name, model_type, model_name, time_range, tag=None)
    model_path = train_paths.best_model

    if verbose:
        print(f"Loading model from: {model_path}")

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    X, y = _load_dataset_sequences(driver_name, time_range, downsample, config)

    if verbose:
        print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} timesteps, {X.shape[2]} features")
        print(f"Positive samples: {y.sum()}/{len(y)} ({100*y.sum()/len(y):.1f}%)")
        print(f"Negative samples: {(y==0).sum()}/{len(y)} ({100*(y==0).sum()/len(y):.1f}%)\n")

    baselines = compute_baselines(X, y, n_conditional_samples)

    if verbose:
        print("Computed baselines:")
        for name, baseline in baselines.items():
            if baseline is not None:
                if baseline.ndim == 3:
                    print(f"  - {name}: {baseline.shape[0]} samples, shape {baseline.shape}")
                else:
                    print(f"  - {name}: shape {baseline.shape}")
            else:
                print(f"  - {name}: None (no negative samples)")
        print()

    explainer = create_explainer(model, method=method, device=device)

    positive_indices = np.where(y == 1)[0]
    if len(positive_indices) == 0:
        print("Warning: No positive samples found in dataset!")
        return

    np.random.seed(42)
    sample_indices = np.random.choice(positive_indices, min(n_samples, len(positive_indices)), replace=False)

    if verbose:
        print(f"Selected sample indices: {sample_indices}\n")

    if baseline_type == 'all':
        baseline_names = ['zero', 'mean_all', 'mean_negative', 'conditional']
    else:
        baseline_names = [baseline_type]

    for baseline_name in baseline_names:
        baseline = baselines.get(baseline_name)
        if baseline is None:
            if verbose:
                print(f"Skipping {baseline_name} (not available)\n")
            continue

        paths = ExperimentPaths(driver_name, model_type, model_name, time_range,
                                tag=f'explanation_{baseline_name}', tag_as_subdir=True)

        explain_with_baseline(explainer, sample_indices, X, config, paths, baseline, baseline_name, verbose)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Baseline '{baseline_name}' explanation complete!")
            print(f"Results saved to: {paths.run_dir}")
            print(f"{'='*60}\n")

# IntegratedGradients (default)
# uv run python .\scripts\explain_model.py -d 강신길 -mt online_mlp -mn base -t "[5,7]" --downsample 5 --baseline all
# uv run python .\scripts\explain_model.py -d 강신길 -mt online_mlp -mn base -t "[5,7]" --downsample 5 --n-samples 10 --baseline mean_negative --device cuda

# KernelShap (coalition-based, more theoretically aligned with TimeSHAP)
# uv run python .\scripts\explain_model.py -d 강신길 -mt online_attention -mn bo -t "[5,8]" --downsample 1 --method kernelshap --baseline all

# Compare baselines
# uv run python .\scripts\explain_model.py -d 강신길 -mt online_mlp -mn base -t "[5,7]" --downsample 5 --n-samples 3 --baseline all -v 1
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Explain model predictions using Captum (IntegratedGradients or KernelShap)')
    parser.add_argument('-d', '--driver-name', required=True, help='Driver name')
    parser.add_argument('-mt', '--model-type', required=True, help='Model type')
    parser.add_argument('-mn', '--model-name', default='base', help='Model name')
    parser.add_argument('-t', '--time-range', type=eval, default=None, help='Time range as list [start, end]')
    parser.add_argument('--downsample', type=int, default=1, help='Downsample rate')
    parser.add_argument('--n-samples', type=int, default=5, help='Number of samples to explain')
    parser.add_argument('--baseline', type=str, default='all',
                        choices=['zero', 'mean_all', 'mean_negative', 'conditional', 'all'],
                        help='Baseline type: zero, mean_all, mean_negative, conditional, or all')
    parser.add_argument('--n-conditional-samples', type=int, default=50,
                        help='Number of samples for conditional imputation baseline (default: 50)')
    parser.add_argument('--method', type=str, default='ig',
                        choices=['ig', 'kernelshap'],
                        help='Explainer method: ig (IntegratedGradients) or kernelshap (KernelShap)')
    parser.add_argument('--device', default='cpu', help='Device (cpu or cuda)')
    parser.add_argument('-v', '--verbose', type=int, default=1, help='Verbosity level')

    args = parser.parse_args()

    main(args.driver_name, args.model_type, args.model_name,
         args.time_range, args.downsample,
         args.n_samples, args.baseline, args.n_conditional_samples,
         args.method, args.device, args.verbose)
