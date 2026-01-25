import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
from torch.utils.data import Dataset
from collections import defaultdict
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from src.utils.utils import _load_dataset_sequences

class PreferenceDataset(Dataset):
    def __init__(self, pref_dataset):
        self.pref_dataset = pref_dataset

    def __len__(self):
        return len(self.pref_dataset["observations"])

    def __getitem__(self, idx):
        observations = self.pref_dataset["observations"][idx]
        labels = self.pref_dataset["labels"][idx]
        
        item = dict(observations=observations, labels=labels)

        if "observations_2" in self.pref_dataset:
            item["observations_2"] = self.pref_dataset["observations_2"][idx]

        if "driver_name" in self.pref_dataset:
            item["driver_name"] = self.pref_dataset["driver_name"][idx]
            
        return item

    def get_mode_data(self, batch_size):
        idxs = np.random.choice(range(len(self)), size=batch_size, replace=False)
        return dict(
            observations=self.pref_dataset["observations"][idxs],
            observations_2=self.pref_dataset["observations_2"][idxs],
        )

    def get_driver_list(self):
        # driver_name shape: (N_queries, Context_Size)
        all_names = self.pref_dataset["driver_name"][:, 0]
        return np.unique(all_names)

    def get_driver_data(self, driver_name):
        # Filter data for the given driver_name
        drivers = self.pref_dataset["driver_name"][:, 0]
        indices = np.where(drivers == driver_name)[0]

        subset = {}
        for key, val in self.pref_dataset.items():
            subset[key] = val[indices]
        return subset

def convert_to_pairwise(X, y, driver_name, context_size=64, tie_ratio=0.0):
    true_episodes = X[y == 1]
    false_episodes = X[y == 0]

    driver_obs, driver_obs_2, driver_labels = [], [], []

    # 1. True vs False
    for true_ep in true_episodes:
        for false_ep in false_episodes:
            driver_obs.append(true_ep)
            driver_obs_2.append(false_ep)
            driver_labels.append(1.0)

    # 2. Tie Pairs
    n_tie = int((len(true_episodes) + len(false_episodes)) * tie_ratio)
    if n_tie > 0:
        if len(true_episodes) > 0:
            true_indices = np.random.choice(len(true_episodes), size=(n_tie, 2))
            for i1, i2 in true_indices:
                driver_obs.append(true_episodes[i1])
                driver_obs_2.append(true_episodes[i2])
                driver_labels.append(0.5)

        if len(false_episodes) > 0:
            false_indices = np.random.choice(len(false_episodes), size=(n_tie, 2))
            for i1, i2 in false_indices:
                driver_obs.append(false_episodes[i1])
                driver_obs_2.append(false_episodes[i2])
                driver_labels.append(0.5)

    driver_obs = np.stack(driver_obs)
    driver_obs_2 = np.stack(driver_obs_2)
    driver_labels = np.array(driver_labels).reshape(-1, 1)

    # Context Grouping
    num_pairs = len(driver_obs)

    grouped_obs = []
    grouped_obs_2 = []
    grouped_labels = []
    grouped_names = []

    for i in range(0, num_pairs, context_size):
        end_idx = min(i + context_size, num_pairs)

        batch_obs = driver_obs[i:end_idx]
        batch_obs_2 = driver_obs_2[i:end_idx]
        batch_lbl = driver_labels[i:end_idx]

        current_len = len(batch_obs)
        if current_len < context_size:
            print(f"  Warning: Padding context from {current_len} to {context_size} for driver {driver_name}.")
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
        'driver_name': np.stack(grouped_names)
    }

def convert_to_binary_context(X, y, driver_name, context_size=64, balanced=False):
    driver_obs = X
    driver_labels = y.reshape(-1) # (N,)

    if len(driver_obs) == 0:
        return None

    pos_indices = np.where(driver_labels == 1)[0]
    neg_indices = np.where(driver_labels == 0)[0]

    # Check if stratified creation is possible
    if balanced and len(pos_indices) > 0 and len(neg_indices) > 0:
        # 1. Calculate Ratios
        total_count = len(driver_labels)
        pos_ratio = len(pos_indices) / total_count

        # 2. Determine counts per batch based on ratio
        n_pos_per_batch = int(np.round(context_size * pos_ratio))

        # Ensure at least 1 sample from each class if context_size allows
        n_pos_per_batch = max(1, min(n_pos_per_batch, context_size - 1))
        n_neg_per_batch = context_size - n_pos_per_batch

        # 3. Determine number of batches
        # We try to cover the larger class roughly once (Oversampling the smaller one implicitly)
        max_samples = max(len(pos_indices), len(neg_indices))
        # Normalized by the portion taken per batch
        max_batches_pos = len(pos_indices) / n_pos_per_batch
        max_batches_neg = len(neg_indices) / n_neg_per_batch

        # Use the max possible batches to utilize data fully (similar to previous logic)
        num_batches = int(np.ceil(max(max_batches_pos, max_batches_neg)))

        grouped_obs = []
        grouped_labels = []
        grouped_names = []

        # Create batches
        for _ in range(num_batches):
            p_idxs = np.random.choice(pos_indices, n_pos_per_batch, replace=True)
            n_idxs = np.random.choice(neg_indices, n_neg_per_batch, replace=True)

            batch_indices = np.concatenate([p_idxs, n_idxs])
            np.random.shuffle(batch_indices) # Mix them within the context

            batch_obs = driver_obs[batch_indices]
            batch_lbl = driver_labels[batch_indices].reshape(-1, 1)

            grouped_obs.append(batch_obs)
            grouped_labels.append(batch_lbl)
            grouped_names.append(np.array([driver_name] * context_size))

        return {
            'observations': np.stack(grouped_obs),
            'labels': np.stack(grouped_labels),
            'driver_name': np.stack(grouped_names)
        }

    # Fallback: Original Sequential Logic (if not balanced or missing class)
    driver_labels = driver_labels.reshape(-1, 1)
    num_samples = len(driver_obs)

    grouped_obs = []
    grouped_labels = []
    grouped_names = []

    for i in range(0, num_samples, context_size):
        end_idx = min(i + context_size, num_samples)

        batch_obs = driver_obs[i:end_idx]
        batch_lbl = driver_labels[i:end_idx]

        current_len = len(batch_obs)

        if current_len < context_size:
            # Padding
            needed = context_size - current_len
            indices = np.random.randint(0, current_len, size=needed)

            batch_obs = np.concatenate([batch_obs, batch_obs[indices]], axis=0)
            batch_lbl = np.concatenate([batch_lbl, batch_lbl[indices]], axis=0)

        grouped_obs.append(batch_obs)
        grouped_labels.append(batch_lbl)
        grouped_names.append(np.array([driver_name] * context_size))

    return {
        'observations': np.stack(grouped_obs), # (N_queries, context, T, F)
        'labels': np.stack(grouped_labels),    # (N_queries, context, 1)
        'driver_name': np.stack(grouped_names) # (N_queries, context)
    }



def create_vpl_dataset(train_driver_names, test_driver_names, features, time_range, downsample,
                       tie_ratio=0.0, context_size=64, normalize=False):
    target_drivers = sorted(list(set(train_driver_names + test_driver_names)))

    config = {'features': features}

    # Step 1: Load data and Normalization
    print(f"Loading raw data to compute normalization stats...")
    raw_data_cache = {}
    all_observations = []

    for driver_name in target_drivers:
        X, y = _load_dataset_sequences(driver_name, time_range, downsample, config)

        if len(X[y==1]) == 0 or len(X[y==0]) == 0:
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
        print(f"  Mean: {mean}\n  Std : {std}")

    # Step 2: Create Pairwise Preference Dataset
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
            
        num_queries = len(dataset_dict['observations'])
        print(f"  {driver_name}: {num_queries} queries generated.")

        for key in ['observations', 'observations_2', 'labels', 'driver_name']:
            train_queries[key].extend(dataset_dict[key])

    train_dataset_dict = {k: np.stack(v) for k, v in train_queries.items() if v}
        
    return train_dataset_dict, test_driver_data, (mean, std)


def log_metrics(metrics, epoch, logger):
    processed = {k: np.mean(v) if isinstance(v, list) else v for k, v in metrics.items()}
    logger.log_metrics(processed, step=epoch)

def prefix_metrics(metrics, prefix):
    return {
        '{}/{}'.format(prefix, key): value for key, value in metrics.items()
    }

def get_labels(seg_reward_1, seg_reward_2):
    sum_r_t_1 = np.sum(seg_reward_1, axis=-1)
    sum_r_t_2 = np.sum(seg_reward_2, axis=-1)
    binary_label = (sum_r_t_1 > sum_r_t_2).reshape(-1, 1).astype(np.float32)
    return binary_label


def get_latent(batch, env, reward_model, mode, num_samples):
    # obs_dim = env.reward_observation_space.shape[0]
    obs1 = batch["observations"]
    obs2 = batch["observations_2"]
    obs_dim = obs1.shape[-1]
    seg_reward_1 = env.compute_reward(obs1.reshape(-1, reward_model.size_segment, obs_dim), mode)
    seg_reward_2 = env.compute_reward(obs2.reshape(-1, reward_model.size_segment, obs_dim), mode)

    seg_reward_1 = seg_reward_1.reshape(
        num_samples, reward_model.annotation_size, reward_model.size_segment, -1
    )
    seg_reward_2 = seg_reward_2.reshape(
        num_samples, reward_model.annotation_size, reward_model.size_segment, -1
    )

    labels = get_labels(seg_reward_1, seg_reward_2)
    device = next(reward_model.parameters()).device
    obs1 = torch.from_numpy(obs1).float().to(device)
    obs2 = torch.from_numpy(obs2).float().to(device)
    labels = torch.from_numpy(labels).float().to(device)
    with torch.no_grad():
        mean, _ = reward_model.encode(obs1, obs2, labels)
    return mean.cpu().numpy()

def get_posterior(env, reward_model, dataset, mode, num_samples):
    batch, num_samples = dataset.get_mode_data(num_samples)
    return get_latent(batch, env, reward_model, mode, num_samples)


def get_all_posterior(env, reward_model, dataset, num_samples):
    means = []
    for mode in range(env.get_num_modes()):
        means.append(get_posterior(env, reward_model, dataset, mode, num_samples))
    return np.stack(means, axis=0)


def get_test_latent(model, driver_dataset, device):
    obs1 = torch.from_numpy(driver_dataset['observations']).float().to(device)
    obs2 = torch.from_numpy(driver_dataset['observations_2']).float().to(device)
    labels = torch.from_numpy(driver_dataset['labels']).float().to(device)

    obs1 = obs1.unsqueeze(0)
    obs2 = obs2.unsqueeze(0)
    labels = labels.unsqueeze(0)

    with torch.no_grad():
        mean, _ = model.encode(obs1, obs2, labels)

    return mean.squeeze(0).cpu().numpy()


def compute_step_rewards(model, X, z_mean, device):
    N, T, d = X.shape

    obs = torch.from_numpy(X).float().to(device)
    z = torch.from_numpy(z_mean).float().to(device)

    z_expanded = z.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    z_expanded = z_expanded.expand(N, 1, T, -1)

    obs = obs.unsqueeze(1)

    with torch.no_grad():
        step_rewards = model.decode(obs, z_expanded)

    step_rewards = step_rewards.squeeze(1).squeeze(-1)

    return step_rewards.cpu().numpy()


def plot_latents(env, reward_model, dataset):
    if reward_model.flow_prior:
        fig, axs = plt.subplots(1, 2, figsize=(10, 8))
        ax1 = axs[0]
        ax2 = axs[1]
    else:
        fig, axs = plt.subplots(1, 1, figsize=(10, 8))
        ax1 = axs

    modes_n = env.get_num_modes()
    latents = get_all_posterior(env, reward_model, dataset, 128)
    for mode_n in range(modes_n):
        z = latents[mode_n]
        #X_embedded = PCA(n_components=2).fit_transform(z)
        X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=10).fit_transform(z)
        ax1.scatter(X_embedded[:, 0], X_embedded[:, 1], c=f"C{mode_n}")

        if reward_model.flow_prior:
            transformed_z = (
                reward_model.flow(
                    torch.from_numpy(z)
                    .float()
                    .to(next(reward_model.parameters()).device)
                )[0]
                .detach()
                .cpu()
                .numpy()
            )
            X_embedded = PCA(n_components=2).fit_transform(transformed_z)
            ax2.scatter(X_embedded[:, 0], X_embedded[:, 1], c=f"C{mode_n}")
    ax1.set_title("Latent embeddings")
    if reward_model.flow_prior:
        ax2.set_title("Transformed latent embeddings")
    plt.close(fig)
    return dict(latent_plot=wandb.Image(fig))    