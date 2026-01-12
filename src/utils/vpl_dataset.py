import numpy as np
import torch
from torch.utils.data import Dataset
from .utils import _load_dataset_sequences


def create_vpl_dataset(
    driver_names,
    time_range,
    downsample,
    features,
    set_len=10,
    config=None
):
    """
    Create VPL dataset by grouping episodes into queries.
    각 query는 한 driver의 set_len개 episodes.

    Args:
        driver_names: List of driver names
        time_range: Tuple (start, end) in seconds
        downsample: Downsampling factor
        features: List of feature names
        set_len: Number of episodes per query (annotation size)
        config: Optional config dict

    Returns:
        dataset: Dict with keys
            'observations': (num_query, set_len, T, d)
            'labels': (num_query, set_len, 1)
            'driver_ids': (num_query,)
        driver_map: Dict {driver_id: driver_name}
    """
    all_queries_obs = []
    all_queries_labels = []
    all_driver_ids = []
    driver_map = {}

    for driver_id, driver_name in enumerate(driver_names):
        X, y = _load_dataset_sequences(
            driver_name,
            time_range,
            downsample,
            {'features': features}
        )

        n_episodes = len(X)
        n_queries = n_episodes // set_len

        if n_queries == 0:
            print(f"Warning: Driver {driver_name} has only {n_episodes} episodes, "
                  f"which is less than set_len={set_len}. Skipping this driver.")
            continue

        for i in range(n_queries):
            start_idx = i * set_len
            end_idx = start_idx + set_len

            query_obs = X[start_idx:end_idx]
            query_labels = y[start_idx:end_idx].reshape(-1, 1)

            all_queries_obs.append(query_obs)
            all_queries_labels.append(query_labels)
            all_driver_ids.append(driver_id)

        driver_map[driver_id] = driver_name
        print(f"Driver {driver_name}: {n_episodes} episodes -> {n_queries} queries")

    dataset = {
        'observations': np.array(all_queries_obs),
        'labels': np.array(all_queries_labels),
        'driver_ids': np.array(all_driver_ids)
    }

    print(f"\nTotal dataset: {len(dataset['observations'])} queries")
    print(f"Shape: {dataset['observations'].shape}")

    return dataset, driver_map


class VPLDataset(Dataset):
    """
    PyTorch Dataset for VPL training.
    """
    def __init__(self, vpl_dataset):
        self.dataset = vpl_dataset

    def __getitem__(self, idx):
        return {
            'observations': self.dataset['observations'][idx],
            'labels': self.dataset['labels'][idx],
            'driver_ids': self.dataset['driver_ids'][idx]
        }

    def __len__(self):
        return len(self.dataset['observations'])
