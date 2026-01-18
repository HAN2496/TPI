import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.data_loader import DatasetManager
from src.utils.utils import _load_dataset_sequences

class PreferenceDataset(Dataset):
    def __init__(self, all_pairs, context_size):
        self.context_size = context_size
        self.queries = []

        # Group pairs into queries of context_size
        for i in range(0, len(all_pairs), context_size):
            query_pairs = all_pairs[i:i+context_size]

            if len(query_pairs) < context_size:
                original_size = len(query_pairs)
                padded_size = context_size - original_size
                print(f"  Query {len(self.queries)}: Padding {padded_size} pairs ({original_size} → {context_size})")

                # Pad last incomplete query by repeating samples
                while len(query_pairs) < context_size:
                    query_pairs.append(query_pairs[np.random.randint(len(query_pairs))])

            # Stack into query
            self.queries.append({
                'observations': np.stack([p['observations'] for p in query_pairs]),
                'observations_2': np.stack([p['observations_2'] for p in query_pairs]),
                'labels': np.stack([p['labels'] for p in query_pairs]),
            })

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return self.queries[idx]

def create_vpl_dataset_new_2(test_driver_names, features, time_range, downsample, tie_ratio=0.0):
    manager = DatasetManager("datasets", downsample=downsample)
    all_driver_names = manager.keys()

    config = {'features': features}

    train_pairs = []  # List of individual pair dicts
    test_driver_data = {}  # Dict of driver dicts (for easy lookup by name)

    for driver_name in all_driver_names:
        X, y = _load_dataset_sequences(driver_name, time_range, downsample, config)

        true_mask = (y == 1)
        false_mask = (y == 0)

        true_episodes = X[true_mask]
        false_episodes = X[false_mask]

        n_true = len(true_episodes)
        n_false = len(false_episodes)

        if n_true == 0 or n_false == 0:
            print(f"Warning: Driver {driver_name} has no True or False episodes. Skipping.")
            continue

        driver_obs = []
        driver_obs_2 = []
        driver_labels = []

        # Phase 1: True vs False pairs
        for true_ep in true_episodes:
            for false_ep in false_episodes:
                driver_obs.append(true_ep)
                driver_obs_2.append(false_ep)
                driver_labels.append(1.0)

        # Phase 2: Tie pairs
        true_tie_pairs = int(len(true_episodes) * tie_ratio)
        false_tie_pairs = int(len(false_episodes) * tie_ratio)
        for _ in range(true_tie_pairs):
            idx_a, idx_b = np.random.choice(n_true, size=2, replace=False)
            driver_obs.append(true_episodes[idx_a])
            driver_obs_2.append(true_episodes[idx_b])
            driver_labels.append(0.5)
        for _ in range(false_tie_pairs):
            idx_a, idx_b = np.random.choice(n_false, size=2, replace=False)
            driver_obs.append(false_episodes[idx_a])
            driver_obs_2.append(false_episodes[idx_b])
            driver_labels.append(0.5)

        # Convert to numpy arrays: (N, T, d)
        driver_obs = np.stack(driver_obs, axis=0)
        driver_obs_2 = np.stack(driver_obs_2, axis=0)
        driver_labels = np.array(driver_labels).reshape(-1, 1)  # (N, 1)

        if driver_name in test_driver_names:
            # For test data, keep as driver-grouped dict
            driver_data = {
                "observations": driver_obs,
                "observations_2": driver_obs_2,
                "labels": driver_labels,
                "driver_name": driver_name
            }
            test_driver_data[driver_name] = driver_data
            driver_type = "Test"
        else:
            # For train data, unpack into individual pairs
            for i in range(len(driver_obs)):
                train_pairs.append({
                    'observations': driver_obs[i],
                    'observations_2': driver_obs_2[i],
                    'labels': driver_labels[i]
                })
            driver_type = "Train"

        print(f"{driver_type} Driver {driver_name}: {n_true} True, {n_false} False -> "
              f"{n_true * n_false} primary + {true_tie_pairs + false_tie_pairs} tie = "
              f"{len(driver_labels)} pairs")

    return train_pairs, test_driver_data

def create_vpl_dataset_new(test_driver_name, features, time_range, downsample, tie_ratio=0.0):
    manager = DatasetManager("datasets", downsample=downsample)
    all_driver_names = manager.keys()
    train_driver_names = [name for name in all_driver_names if name != test_driver_name]

    config = {}
    config['features'] = features

    driver_datasets = []

    for train_driver_name in train_driver_names:
        X, y = _load_dataset_sequences(train_driver_name, time_range, downsample, config)

        true_mask  = (y == 1)
        false_mask = (y == 0)

        true_episodes  = X[true_mask]
        false_episodes = X[false_mask]

        n_true  = len(true_episodes)
        n_false = len(false_episodes)

        if n_true == 0 or n_false == 0:
            print(f"Warning: Driver {train_driver_name} has no True or False episodes. Skipping.")
            continue

        driver_obs = []
        driver_obs_2 = []
        driver_labels = []

        # Phase 1: True vs False pairs
        for true_ep in true_episodes:
            for false_ep in false_episodes:
                driver_obs.append(true_ep)
                driver_obs_2.append(false_ep)
                driver_labels.append(1.0)

        # Phase 2: Tie pairs
        true_tie_pairs = int(len(true_episodes) * tie_ratio)
        false_tie_pairs = int(len(false_episodes) * tie_ratio)

        for _ in range(true_tie_pairs):
            idx_a, idx_b = np.random.choice(n_true, size=2, replace=False)
            driver_obs.append(true_episodes[idx_a])
            driver_obs_2.append(true_episodes[idx_b])
            driver_labels.append(0.5)

        for _ in range(false_tie_pairs):
            idx_a, idx_b = np.random.choice(n_false, size=2, replace=False)
            driver_obs.append(false_episodes[idx_a])
            driver_obs_2.append(false_episodes[idx_b])
            driver_labels.append(0.5)

        # Convert to numpy arrays: (N, T, d)
        driver_obs = np.stack(driver_obs, axis=0)
        driver_obs_2 = np.stack(driver_obs_2, axis=0)
        driver_labels = np.array(driver_labels).reshape(-1, 1)  # (N, 1)

        driver_datasets.append({
            "observations": driver_obs,
            "observations_2": driver_obs_2,
            "labels": driver_labels,
            "driver_name": train_driver_name
        })

        print(f"Driver {train_driver_name}: {n_true} True, {n_false} False -> "
              f"{n_true * n_false} primary + {true_tie_pairs + false_tie_pairs} tie = "
              f"{len(driver_labels)} pairs")

    return driver_datasets


