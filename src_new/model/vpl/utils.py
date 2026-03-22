import torch
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict

from ...data.splits import load_sequences


class PreferenceDataset(Dataset):
    def __init__(self, pref_dataset):
        self.pref_dataset = pref_dataset

    def __len__(self):
        return len(self.pref_dataset["observations"])

    def __getitem__(self, idx):
        item = {
            "observations": self.pref_dataset["observations"][idx],
            "labels": self.pref_dataset["labels"][idx],
        }
        if "observations_2" in self.pref_dataset:
            item["observations_2"] = self.pref_dataset["observations_2"][idx]
        return item

    def get_driver_list(self):
        return np.unique(self.pref_dataset["driver_name"][:, 0])

    def get_driver_data(self, driver_name):
        drivers = self.pref_dataset["driver_name"][:, 0]
        indices = np.where(drivers == driver_name)[0]
        return {key: val[indices] for key, val in self.pref_dataset.items()}


def convert_to_pairwise(X, y, driver_name, context_size=64, tie_ratio=0.0):
    true_episodes = X[y == 1]
    false_episodes = X[y == 0]

    driver_obs, driver_obs_2, driver_labels = [], [], []

    for true_ep in true_episodes:
        for false_ep in false_episodes:
            driver_obs.append(true_ep)
            driver_obs_2.append(false_ep)
            driver_labels.append(1.0)

    n_tie = int((len(true_episodes) + len(false_episodes)) * tie_ratio)
    if n_tie > 0:
        if len(true_episodes) > 0:
            for i1, i2 in np.random.choice(len(true_episodes), size=(n_tie, 2)):
                driver_obs.append(true_episodes[i1])
                driver_obs_2.append(true_episodes[i2])
                driver_labels.append(0.5)
        if len(false_episodes) > 0:
            for i1, i2 in np.random.choice(len(false_episodes), size=(n_tie, 2)):
                driver_obs.append(false_episodes[i1])
                driver_obs_2.append(false_episodes[i2])
                driver_labels.append(0.5)

    driver_obs = np.stack(driver_obs)
    driver_obs_2 = np.stack(driver_obs_2)
    driver_labels = np.array(driver_labels).reshape(-1, 1)

    grouped_obs, grouped_obs_2, grouped_labels, grouped_names = [], [], [], []
    for i in range(0, len(driver_obs), context_size):
        end_idx = min(i + context_size, len(driver_obs))
        batch_obs = driver_obs[i:end_idx]
        batch_obs_2 = driver_obs_2[i:end_idx]
        batch_lbl = driver_labels[i:end_idx]
        current_len = len(batch_obs)
        if current_len < context_size:
            needed = context_size - current_len
            indices = np.random.randint(0, current_len, size=needed)
            batch_obs = np.concatenate([batch_obs, batch_obs[indices]], axis=0)
            batch_obs_2 = np.concatenate([batch_obs_2, batch_obs_2[indices]], axis=0)
            batch_lbl = np.concatenate([batch_lbl, batch_lbl[indices]], axis=0)
        grouped_obs.append(batch_obs)
        grouped_obs_2.append(batch_obs_2)
        grouped_labels.append(batch_lbl)
        grouped_names.append(np.array([driver_name] * context_size))

    return {
        'observations': np.stack(grouped_obs),
        'observations_2': np.stack(grouped_obs_2),
        'labels': np.stack(grouped_labels),
        'driver_name': np.stack(grouped_names),
    }


def convert_to_binary_context(X, y, driver_name, context_size=64, balanced=False):
    driver_obs = X
    driver_labels = y.reshape(-1)
    if len(driver_obs) == 0:
        return None

    pos_indices = np.where(driver_labels == 1)[0]
    neg_indices = np.where(driver_labels == 0)[0]

    if balanced and len(pos_indices) > 0 and len(neg_indices) > 0:
        pos_ratio = len(pos_indices) / len(driver_labels)
        n_pos_per_batch = max(1, min(int(np.round(context_size * pos_ratio)), context_size - 1))
        n_neg_per_batch = context_size - n_pos_per_batch
        num_batches = int(np.ceil(max(len(pos_indices) / n_pos_per_batch, len(neg_indices) / n_neg_per_batch)))

        grouped_obs, grouped_labels, grouped_names = [], [], []
        for _ in range(num_batches):
            batch_indices = np.random.permutation(np.concatenate([
                np.random.choice(pos_indices, n_pos_per_batch, replace=True),
                np.random.choice(neg_indices, n_neg_per_batch, replace=True),
            ]))
            grouped_obs.append(driver_obs[batch_indices])
            grouped_labels.append(driver_labels[batch_indices].reshape(-1, 1))
            grouped_names.append(np.array([driver_name] * context_size))

        return {
            'observations': np.stack(grouped_obs),
            'labels': np.stack(grouped_labels),
            'driver_name': np.stack(grouped_names),
        }

    driver_labels = driver_labels.reshape(-1, 1)
    grouped_obs, grouped_labels, grouped_names = [], [], []
    for i in range(0, len(driver_obs), context_size):
        end_idx = min(i + context_size, len(driver_obs))
        batch_obs = driver_obs[i:end_idx]
        batch_lbl = driver_labels[i:end_idx]
        current_len = len(batch_obs)
        if current_len < context_size:
            needed = context_size - current_len
            indices = np.random.randint(0, current_len, size=needed)
            batch_obs = np.concatenate([batch_obs, batch_obs[indices]], axis=0)
            batch_lbl = np.concatenate([batch_lbl, batch_lbl[indices]], axis=0)
        grouped_obs.append(batch_obs)
        grouped_labels.append(batch_lbl)
        grouped_names.append(np.array([driver_name] * context_size))

    return {
        'observations': np.stack(grouped_obs),
        'labels': np.stack(grouped_labels),
        'driver_name': np.stack(grouped_names),
    }


def create_vpl_dataset(train_driver_names, test_driver_names, features, time_range, downsample,
                       tie_ratio=0.0, context_size=64, normalize=False,
                       smooth=False, smooth_cutoff=12.0, smooth_order=2):
    target_drivers = sorted(set(train_driver_names + test_driver_names))

    print("Loading raw data to compute normalization stats...")
    raw_data_cache = {}
    all_observations = []

    for driver_name in target_drivers:
        X, y = load_sequences(driver_name, features, time_range, downsample,
                               smooth=smooth, smooth_cutoff=smooth_cutoff, smooth_order=smooth_order)
        if len(X[y == 1]) == 0 or len(X[y == 0]) == 0:
            raise ValueError(f"Driver {driver_name} must have both positive and negative samples for VPL training.")
        raw_data_cache[driver_name] = (X, y)
        if normalize:
            all_observations.append(X)

    mean, std = None, None
    if normalize and all_observations:
        print("Computing normalization stats...")
        concat_obs = np.concatenate(all_observations, axis=0)
        mean = np.mean(concat_obs, axis=(0, 1))
        std = np.std(concat_obs, axis=(0, 1)) + 1e-6

    train_queries = defaultdict(list)
    test_driver_data = {}

    print(f"Creating datasets with context_size={context_size}...")
    for driver_name in target_drivers:
        X_raw, y = raw_data_cache[driver_name]
        X = (X_raw - mean) / std if normalize else X_raw

        if driver_name in test_driver_names:
            test_driver_data[driver_name] = (X, y)
            print(f"  {driver_name}: Saved as raw data ({len(X)} episodes)")
            continue

        dataset_dict = convert_to_pairwise(X, y, driver_name, context_size, tie_ratio)
        print(f"  {driver_name}: {len(dataset_dict['observations'])} queries generated.")
        for key in ['observations', 'observations_2', 'labels', 'driver_name']:
            train_queries[key].extend(dataset_dict[key])

    train_driver_data = {k: raw_data_cache[k] for k in train_driver_names}
    train_dataset_dict = {k: np.stack(v) for k, v in train_queries.items() if v}
    return train_dataset_dict, train_driver_data, test_driver_data, (mean, std)


def compute_step_rewards(model, X, z_mean, device):
    N, T, d = X.shape
    obs = torch.from_numpy(X).float().to(device)
    z = torch.from_numpy(z_mean).float().to(device)
    z_expanded = z.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(N, 1, T, -1)
    with torch.no_grad():
        step_rewards = model.decode(obs.unsqueeze(1), z_expanded)
    return step_rewards.squeeze(1).squeeze(-1).cpu().numpy()
