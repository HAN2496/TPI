import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.data_loader import DatasetManager
from src.utils.utils import _load_dataset_sequences

def create_vpl_datset_new(test_deriver_name, features, time_range, downsample, tie_ratio=0.0):
    manager = DatasetManager("datasets", downsample=downsample)
    all_driver_names = manager.keys()
    train_driver_names = [name for name in all_driver_names if name != test_deriver_name]

    config = {}
    config['features'] = features
    all_pairs = []
    for idx, train_driver_name in enumerate(train_driver_names):
        X, y = _load_dataset_sequences(train_driver_name, time_range, downsample, config)

        true_mask  = (y == 1)
        false_mask = (y == 0)

        true_episodes  = X[true_mask]
        false_episodes = X[false_mask]

        n_true  = len(true_episodes)
        n_false = len(false_episodes)

        # Phase 1: True vs False pairs
        for true_ep in true_episodes:
            for false_ep in false_episodes:
                all_pairs.append({
                    "driver_id": idx,
                    "driver_name": train_driver_name,
                    "observations": true_ep,
                    "observations_2": false_ep,
                    "labels": 1.0,
                })

        # Phase 2: Tie pairs
        true_tie_pairs = int(len(true_episodes) * tie_ratio)
        false_tie_pairs = int(len(false_episodes) * tie_ratio)

        for _ in range(true_tie_pairs):
            idx_a, idx_b = np.random.choice(n_true, size=2, replace=False)
            all_pairs.append({
                "driver_id": idx,
                "driver_name": train_driver_name,
                "observations": true_episodes[idx_a],
                "observations_2": true_episodes[idx_b],
                "labels": 0.5,
            })
        
        for _ in range(false_tie_pairs):
            idx_a, idx_b = np.random.choice(n_false, size=2, replace=False)
            all_pairs.append({
                "driver_id": idx,
                "driver_name": train_driver_name,
                "observations": false_episodes[idx_a],
                "observations_2": false_episodes[idx_b],
                "labels": 0.5,
            })
        
        print(f"Driver {train_driver_name}: {n_true} True, {n_false} False → "
              f"{n_true * n_false} primary + {true_tie_pairs + false_tie_pairs} tie = "
              f"{n_true * n_false + true_tie_pairs + false_tie_pairs} pairs")

    np.random.shuffle(all_pairs)

    

    return all_pairs



def create_vpl_pairwise_dataset(
    driver_names,
    time_range,
    downsample,
    features,
    set_len=10,
    max_pairs_per_true=10,
    tie_ratio=0.0,
    config=None
):
    """
    Create pairwise preference dataset from binary labels.

    Strategy:
    1. Within-driver True vs False pairs (primary)
    2. Same-label tie pairs (20% augmentation)
    3. Group into queries of size set_len

    Args:
        driver_names: List of driver names
        time_range: Tuple (start, end) in seconds
        downsample: Downsampling factor
        features: List of feature names
        set_len: Number of pairs per query (annotation size)
        max_pairs_per_true: Max False episodes paired with each True episode
        tie_ratio: Ratio of tie pairs to add (default 0.25 = 20% of primary)
        config: Additional config

    Returns:
        dataset: {
            'obs_a': (num_query, set_len, T, d),
            'obs_b': (num_query, set_len, T, d),
            'preferences': (num_query, set_len, 1),
            'driver_ids': (num_query,)
        }
        driver_map: {driver_id: driver_name}
    """
    all_pairs = []
    driver_map = {}

    total_primary_pairs = 0
    total_tie_pairs = 0

    for driver_id, driver_name in enumerate(driver_names):
        # Load driver episodes
        if config is None:
            config = {}
        config['features'] = features

        X, y = _load_dataset_sequences(driver_name, time_range, downsample, config)
        # X: (N, T, d), y: (N,) with 0/1 values

        true_mask = (y == 1)
        false_mask = (y == 0)

        true_episodes = X[true_mask]
        false_episodes = X[false_mask]

        n_true = len(true_episodes)
        n_false = len(false_episodes)

        if n_true == 0 or n_false == 0:
            print(f"Warning: Driver {driver_name} has no True or False episodes. Skipping.")
            continue

        driver_primary_pairs = 0
        driver_tie_pairs = 0

        # Phase 1: True vs False pairs
        for i in range(n_true):
            # Sample up to max_pairs_per_true False episodes
            n_samples = min(n_false, max_pairs_per_true)
            false_indices = np.random.choice(n_false, size=n_samples, replace=False)

            for j in false_indices:
                all_pairs.append({
                    'obs_a': true_episodes[i],
                    'obs_b': false_episodes[j],
                    'preference': 1.0,
                    'driver_id': driver_id
                })
                driver_primary_pairs += 1

        total_primary_pairs += driver_primary_pairs

        # Phase 2: Tie pairs
        num_tie_pairs = int(driver_primary_pairs * tie_ratio)

        # True vs True ties
        if n_true >= 2:
            for _ in range(num_tie_pairs // 2):
                idx_a, idx_b = np.random.choice(n_true, size=2, replace=False)
                all_pairs.append({
                    'obs_a': true_episodes[idx_a],
                    'obs_b': true_episodes[idx_b],
                    'preference': 0.5,
                    'driver_id': driver_id
                })
                driver_tie_pairs += 1

        # False vs False ties
        if n_false >= 2:
            for _ in range(num_tie_pairs // 2):
                idx_a, idx_b = np.random.choice(n_false, size=2, replace=False)
                all_pairs.append({
                    'obs_a': false_episodes[idx_a],
                    'obs_b': false_episodes[idx_b],
                    'preference': 0.5,
                    'driver_id': driver_id
                })
                driver_tie_pairs += 1

        total_tie_pairs += driver_tie_pairs

        print(f"Driver {driver_name}: {n_true} True, {n_false} False → "
              f"{driver_primary_pairs} primary + {driver_tie_pairs} tie = {driver_primary_pairs + driver_tie_pairs} pairs")

        driver_map[driver_id] = driver_name

    print(f"Total: {len(all_pairs)} pairs ({total_primary_pairs} primary + {total_tie_pairs} tie)")

    # Phase 3: Shuffle and group into queries
    np.random.shuffle(all_pairs)

    queries_obs_a = []
    queries_obs_b = []
    queries_preferences = []
    queries_driver_ids = []

    for i in range(0, len(all_pairs), set_len):
        query_pairs = all_pairs[i:i+set_len]

        if len(query_pairs) < set_len:
            break  # Skip incomplete query

        queries_obs_a.append(np.stack([p['obs_a'] for p in query_pairs]))
        queries_obs_b.append(np.stack([p['obs_b'] for p in query_pairs]))
        queries_preferences.append(np.array([p['preference'] for p in query_pairs]).reshape(-1, 1))
        queries_driver_ids.append(query_pairs[0]['driver_id'])  # Assume same driver

    num_queries = len(queries_obs_a)
    print(f"Created {num_queries} queries (set_len={set_len})")

    dataset = {
        'obs_a': np.array(queries_obs_a),              # (num_query, set_len, T, d)
        'obs_b': np.array(queries_obs_b),              # (num_query, set_len, T, d)
        'preferences': np.array(queries_preferences),  # (num_query, set_len, 1)
        'driver_ids': np.array(queries_driver_ids)     # (num_query,)
    }

    return dataset, driver_map


class VPLPairwiseDataset(Dataset):
    """
    PyTorch Dataset for pairwise preference training.
    """
    def __init__(self, dataset_dict):
        self.dataset = dataset_dict

    def __getitem__(self, idx):
        return {
            'obs_a': self.dataset['obs_a'][idx],           # (set_len, T, d)
            'obs_b': self.dataset['obs_b'][idx],           # (set_len, T, d)
            'preferences': self.dataset['preferences'][idx], # (set_len, 1)
            'driver_ids': self.dataset['driver_ids'][idx]  # scalar
        }

    def __len__(self):
        return len(self.dataset['obs_a'])
