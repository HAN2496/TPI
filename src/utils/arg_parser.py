import argparse
import ast


def base_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu/cuda) (mps if on Mac)')
    parser.add_argument('-v', '--verbose', type=int, default=0, choices=[0, 1, 2],
                       help='Verbosity level (0: silent, 1: verbose, 2: debug)')
    return parser

def train_model_parser():
    parser = base_parser()
    parser.add_argument('-n', '--driver-name', type=str, required=True, help='Name of the driver/configuration')
    parser.add_argument('-mt', '--model-type', type=str, required=True, help='Type of the model to train')
    parser.add_argument('-mn', '--model-name', type=str, required=True, help='Name of the model to train')
    parser.add_argument('-t', '--time-range', type=ast.literal_eval, default=[5, 7], help='Time range for training (e.g., "[5,7]")')
    parser.add_argument('-f', '--feature-version', type=str, default='v1', help='Feature version to use (e.g., "v1", "v2")')
    parser.add_argument('-d', '--downsample', type=int, default=5, help='Downsampling factor for training data')

    parser.add_argument('--n-splits', type=int, default=5, help='Number of folds')
    parser.add_argument('--test-ratio', type=float, default=0.1, help='Test set ratio (0.0-1.0, default: 0.1)')
    return parser

def optimize_model_parser():
    parser = base_parser()
    parser.add_argument('-n', '--driver-name', type=str, required=True, help='Name of the driver/configuration')
    parser.add_argument('-mt', '--model-type', type=str, required=True, help='Type of the model to optimize')
    parser.add_argument('-t', '--time-range', type=ast.literal_eval, default=[5, 7], help='Time range for optimization (e.g., "[5,7]")')
    parser.add_argument('-d', '--downsample', type=int, default=5, help='Downsampling factor for train and val data')

    parser.add_argument('--n-splits', type=int, default=5, help='Number of folds')
    parser.add_argument('--test-ratio', type=float, default=0.1, help='Test set ratio (0.0-1.0, default: 0.1)')

    parser.add_argument('-fs', '--use-feature-selection', action='store_true', help='Enable feature selection during optimization')
    parser.add_argument('--n-trials', type=int, default=100, help='Number of optimization trials')
    parser.add_argument('--optimizer-type', type=str, default='bayesian', choices=['bayesian', 'exhaustive'], help='Optimizer type')
    parser.add_argument('--tag', type=str, default='optuna', help='Tag for experiment identification')
    return parser