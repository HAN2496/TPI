import torch
import numpy as np
from src.configs.utils import load_config
from src.configs.registries import MODELS
from src.utils import convert_driver_name, ExperimentPaths
from src.utils.utils import _load_dataset_sequences
from src.utils.explainability import (
    create_explainer,
    compute_feature_importance, compute_temporal_importance, compute_feature_time_matrix,
    plot_feature_importance, plot_temporal_importance,
    plot_feature_time_heatmap, plot_top_contributions
)

def main(driver_name, model_type, model_name, time_range, downsample,
         n_samples=5, device='cpu', verbose=1):

    driver_name = convert_driver_name(driver_name)
    config = load_config(driver_name, model_type, model_name, verbose=verbose)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Captum IntegratedGradients Explanation Analysis")
        print(f"Driver: {driver_name}")
        print(f"Model: {model_type}/{model_name}")
        print(f"Time range: {time_range}, Downsample: {downsample}")
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
        print(f"Positive samples: {y.sum()}/{len(y)} ({100*y.sum()/len(y):.1f}%)\n")

    explainer = create_explainer(model, device)

    positive_indices = np.where(y == 1)[0]
    if len(positive_indices) == 0:
        print("Warning: No positive samples found in dataset!")
        return

    sample_indices = np.random.choice(positive_indices, min(n_samples, len(positive_indices)), replace=False)

    paths = ExperimentPaths(driver_name, model_type, model_name, time_range, tag='explanation')

    if verbose:
        print(f"Explaining {len(sample_indices)} positive samples...\n")

    for i, idx in enumerate(sample_indices):
        x_sample = X[idx]

        if verbose:
            print(f"{'='*60}")
            print(f"Sample {i+1}/{len(sample_indices)} (Dataset Index: {idx})")
            print(f"{'='*60}")

        try:
            attributions = explainer.explain_sample(x_sample, baseline=None)

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

    if verbose:
        print(f"{'='*60}")
        print(f"Explanation complete!")
        print(f"Results saved to: {paths.run_dir}")
        print(f"{'='*60}\n")

# uv run python .\scripts\explain_model.py -d 강신길 -mt online_mlp -mn base -t "[5,7]" --downsample 5
# uv run python .\scripts\explain_model.py -d 강신길 -mt online_mlp -mn base -t "[5,7]" --downsample 5 --n-samples 10 --device cuda
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Explain model predictions using Captum IntegratedGradients')
    parser.add_argument('-d', '--driver-name', required=True, help='Driver name')
    parser.add_argument('-mt', '--model-type', required=True, help='Model type')
    parser.add_argument('-mn', '--model-name', default='base', help='Model name')
    parser.add_argument('-t', '--time-range', type=eval, default=None, help='Time range as list [start, end]')
    parser.add_argument('--downsample', type=int, default=1, help='Downsample rate')
    parser.add_argument('--n-samples', type=int, default=5, help='Number of samples to explain')
    parser.add_argument('--device', default='cpu', help='Device (cpu or cuda)')
    parser.add_argument('-v', '--verbose', type=int, default=1, help='Verbosity level')

    args = parser.parse_args()

    main(args.driver_name, args.model_type, args.model_name,
         args.time_range, args.downsample,
         args.n_samples, args.device, args.verbose)
