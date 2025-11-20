import argparse


def base_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu/cuda) (mps if on Mac)')
    parser.add_argument('-v', '--verbose', type=int, default=0, choices=[0, 1, 2],
                       help='Verbosity level (0: silent, 1: verbose, 2: debug)')
    return parser

def train_model_parser():
    parser = base_parser()
    parser.add_argument('-d', '--driver_name', type=str, required=True, help='Name of the driver/configuration')
    parser.add_argument('-m', '--model_name', type=str, required=True, help='Name of the model to train')
    parser.add_argument('-t', '--time_range', type=list, default=[5, 7], help='Time range for training (e.g., "0-100")')
    parser.add_argument('--downsample', type=int, default=5, help='Downsampling factor for the data')
    return parser

def optimize_model_parser():
    parser = base_parser()
    parser.add_argument('-d', '--driver_name', type=str, required=True, help='Name of the driver/configuration')
    parser.add_argument('-m', '--model_name', type=str, required=True, help='Name of the model to optimize')
    parser.add_argument('-t', '--time_range', type=list, default=[5, 7], help='Time range for optimization')
    parser.add_argument('--downsample', type=int, default=5, help='Downsampling factor for the data')
    parser.add_argument('-n', '--n_trials', type=int, default=100, help='Number of optimization trials')
    parser.add_argument('--tag', type=str, default='optuna', help='Tag for experiment identification')
    return parser