import torch
import numpy as np
import pickle
from pathlib import Path


@torch.no_grad()
def infer_driver_latents(model, dataset, driver_map, device='cpu'):
    """
    For each driver, encode their queries to get latent z.

    Args:
        model: Trained VAEModel
        dataset: VPL dataset dict
        driver_map: Dict {driver_id: driver_name}
        device: Device to use

    Returns:
        {driver_name: {
            'mean': ndarray (num_queries, latent_dim),
            'logvar': ndarray (num_queries, latent_dim),
            'z_mean_point': ndarray (latent_dim,) - average latent
        }}
    """
    model.eval()
    driver_latents = {}

    for driver_id, driver_name in driver_map.items():
        mask = (dataset['driver_ids'] == driver_id)
        obs = torch.tensor(dataset['observations'][mask]).float().to(device)
        labels = torch.tensor(dataset['labels'][mask]).float().to(device)

        mean, logvar = model.encode(obs, labels)

        driver_latents[driver_name] = {
            'mean': mean.cpu().numpy(),
            'logvar': logvar.cpu().numpy(),
            'z_mean_point': mean.cpu().numpy().mean(axis=0)
        }

    return driver_latents


def save_driver_latents(driver_latents, save_path):
    """Save driver latents to pickle file"""
    with open(save_path, 'wb') as f:
        pickle.dump(driver_latents, f)


def load_driver_latents(load_path):
    """Load driver latents from pickle file"""
    with open(load_path, 'rb') as f:
        return pickle.load(f)


@torch.no_grad()
def compute_reward_for_episode(model, episode_data, z_latent, device='cpu'):
    """
    Given episode (T, d) and latent z, compute reward r(obs, z).

    Args:
        model: Trained VAEModel
        episode_data: Episode observations (T, d)
        z_latent: Latent variable (latent_dim,)
        device: Device to use

    Returns:
        rewards: (T,) rewards per time step
    """
    model.eval()
    obs = torch.tensor(episode_data).float().to(device)
    z = torch.tensor(z_latent).float().to(device)

    if z.ndim == 1:
        z = z.unsqueeze(0)

    z_expanded = z.repeat(obs.shape[0], 1)
    obs_z = torch.cat([obs, z_expanded], dim=-1)
    rewards = model.Decoder(obs_z).squeeze(-1).cpu().numpy()

    return rewards


@torch.no_grad()
def infer_driver_latents_pairwise(model, dataset, driver_map, device='cpu'):
    """
    For each driver, encode their pairwise queries to get latent z.

    Args:
        model: Trained VAEPairwiseModel
        dataset: Pairwise VPL dataset dict
        driver_map: Dict {driver_id: driver_name}
        device: Device to use

    Returns:
        {driver_name: {
            'mean': ndarray (num_queries, latent_dim),
            'logvar': ndarray (num_queries, latent_dim),
            'z_mean_point': ndarray (latent_dim,) - average latent
        }}
    """
    model.eval()
    driver_latents = {}

    for driver_id, driver_name in driver_map.items():
        mask = (dataset['driver_ids'] == driver_id)
        obs_a = torch.tensor(dataset['obs_a'][mask]).float().to(device)
        obs_b = torch.tensor(dataset['obs_b'][mask]).float().to(device)
        preferences = torch.tensor(dataset['preferences'][mask]).float().to(device)

        mean, logvar = model.encode_pairwise(obs_a, obs_b, preferences)

        driver_latents[driver_name] = {
            'mean': mean.cpu().numpy(),
            'logvar': logvar.cpu().numpy(),
            'z_mean_point': mean.cpu().numpy().mean(axis=0)
        }

    return driver_latents


@torch.no_grad()
def evaluate_test_drivers(model, dataset, driver_map, device='cpu'):
    """
    Evaluate model on test drivers to get predictions for all queries.

    Args:
        model: Trained VAEModel
        dataset: VPL dataset dict with test driver data
        driver_map: Dict {driver_id: driver_name}
        device: Device to use

    Returns:
        results: Dict {driver_name: {'y_true': array, 'y_probs': array}}
    """
    model.eval()
    results = {}

    for driver_id, driver_name in driver_map.items():
        mask = (dataset['driver_ids'] == driver_id)
        obs = torch.tensor(dataset['observations'][mask]).float().to(device)
        labels = torch.tensor(dataset['labels'][mask]).float().to(device)

        # Encode and sample z
        mean, log_var = model.encode(obs, labels)
        z = model.reparameterization(mean, torch.exp(0.5 * log_var))

        # Expand z for decoding
        z = z.repeat((1, model.annotation_size * model.size_segment)).view(
            -1, model.annotation_size, model.size_segment, z.shape[1]
        )

        # Decode to get rewards
        r = model.decode(obs, z)

        # Compute probabilities
        r_hat = r.sum(axis=2) / model.scaling
        p_hat = torch.sigmoid(r_hat).view(-1)

        # Flatten labels
        y_true_flat = labels.cpu().numpy().reshape(-1)
        y_probs_flat = p_hat.cpu().numpy()

        results[driver_name] = {
            'y_true': y_true_flat,
            'y_probs': y_probs_flat
        }

    return results


def compare_driver_rewards(model, episode, driver_latents, device='cpu'):
    """
    For a given episode, compute rewards using each driver's latent.

    Args:
        model: Trained VAEModel
        episode: Episode observations (T, d)
        driver_latents: Dict from infer_driver_latents
        device: Device to use

    Returns:
        {driver_name: (T,) rewards}
    """
    results = {}
    for driver_name, latent_info in driver_latents.items():
        z = latent_info['z_mean_point']
        rewards = compute_reward_for_episode(model, episode, z, device)
        results[driver_name] = rewards

    return results


@torch.no_grad()
def estimate_z_from_binary_feedback(model, episodes, labels, set_len=10, device='cpu'):
    """
    Estimate latent z from binary feedback episodes.

    Strategy: Convert binary episodes to pairwise format and encode.
    - Create True vs False pairs (primary signal)
    - Create True vs True and False vs False ties (for robustness)

    Args:
        model: Trained VAEPairwiseModel
        episodes: (N, T, d) - binary labeled episodes
        labels: (N,) - 0 or 1 binary labels
        set_len: Number of pairs per query for encoding
        device: Device to use

    Returns:
        z_mean: (latent_dim,) - estimated driver latent
    """
    model.eval()

    true_mask = (labels == 1)
    false_mask = (labels == 0)

    true_episodes = episodes[true_mask]
    false_episodes = episodes[false_mask]

    n_true = len(true_episodes)
    n_false = len(false_episodes)

    if n_true == 0 or n_false == 0:
        raise ValueError(f"Need both True and False episodes. Got {n_true} True, {n_false} False")

    # Create pairwise comparisons
    all_pairs = []

    # Phase 1: True vs False pairs (primary)
    max_pairs_per_true = min(n_false, 10)
    for i in range(n_true):
        n_samples = min(n_false, max_pairs_per_true)
        false_indices = np.random.choice(n_false, size=n_samples, replace=False)

        for j in false_indices:
            all_pairs.append({
                'obs_a': true_episodes[i],
                'obs_b': false_episodes[j],
                'preference': 1.0
            })

    # Phase 2: Tie pairs (20%)
    num_primary = len(all_pairs)
    num_tie = int(num_primary * 0.25)

    # True vs True
    if n_true >= 2:
        for _ in range(num_tie // 2):
            idx_a, idx_b = np.random.choice(n_true, size=2, replace=False)
            all_pairs.append({
                'obs_a': true_episodes[idx_a],
                'obs_b': true_episodes[idx_b],
                'preference': 0.5
            })

    # False vs False
    if n_false >= 2:
        for _ in range(num_tie // 2):
            idx_a, idx_b = np.random.choice(n_false, size=2, replace=False)
            all_pairs.append({
                'obs_a': false_episodes[idx_a],
                'obs_b': false_episodes[idx_b],
                'preference': 0.5
            })

    # Group into queries
    np.random.shuffle(all_pairs)

    all_means = []

    for i in range(0, len(all_pairs), set_len):
        query_pairs = all_pairs[i:i+set_len]

        if len(query_pairs) < set_len:
            break

        # Stack into batch
        obs_a = torch.tensor(np.stack([p['obs_a'] for p in query_pairs])).float().to(device).unsqueeze(0)
        obs_b = torch.tensor(np.stack([p['obs_b'] for p in query_pairs])).float().to(device).unsqueeze(0)
        prefs = torch.tensor(np.array([p['preference'] for p in query_pairs]).reshape(-1, 1)).float().to(device).unsqueeze(0)

        # Encode
        mean, logvar = model.encode_pairwise(obs_a, obs_b, prefs)
        all_means.append(mean.cpu().numpy())

    # Average all query means
    z_mean = np.concatenate(all_means, axis=0).mean(axis=0)

    return z_mean


@torch.no_grad()
def predict_rewards_with_z(model, episodes, z_latent, device='cpu'):
    """
    Predict rewards for episodes using estimated latent z.

    Uses decoder as reward model: r(obs, z)

    Args:
        model: Trained VAEPairwiseModel
        episodes: (N, T, d) - episodes to predict rewards for
        z_latent: (latent_dim,) - estimated driver latent
        device: Device to use

    Returns:
        rewards: (N, T) - predicted rewards for each timestep
    """
    model.eval()

    all_rewards = []

    for episode in episodes:
        rewards = compute_reward_for_episode(model, episode, z_latent, device)
        all_rewards.append(rewards)

    return np.array(all_rewards)


@torch.no_grad()
def infer_and_predict_from_binary_feedback(
    model, episodes, labels,
    z_estimation_ratio=0.5,
    set_len=10,
    device='cpu'
):
    """
    Two-stage inference from binary feedback:
    1. Estimate latent z from subset of episodes
    2. Predict rewards for remaining episodes using decoder

    This allows using pairwise-trained model with binary feedback by:
    - Using encoder to estimate driver-specific z
    - Using decoder as reward model r(obs, z)

    Args:
        model: Trained VAEPairwiseModel
        episodes: (N, T, d) - binary labeled episodes
        labels: (N,) - 0 or 1 binary labels
        z_estimation_ratio: Fraction of data for z estimation (default 0.5)
        set_len: Number of pairs per query for encoding
        device: Device to use

    Returns:
        z_estimated: (latent_dim,) - estimated driver latent
        predictions: (N_pred, T) - predicted rewards for remaining episodes
        prediction_indices: Indices of episodes used for prediction
    """
    n_total = len(episodes)
    n_z_estimation = max(2, int(n_total * z_estimation_ratio))

    # Shuffle indices
    indices = np.random.permutation(n_total)
    z_indices = indices[:n_z_estimation]
    pred_indices = indices[n_z_estimation:]

    # Split data
    z_episodes = episodes[z_indices]
    z_labels = labels[z_indices]

    pred_episodes = episodes[pred_indices]

    print(f"Estimating z from {len(z_episodes)} episodes...")
    z_estimated = estimate_z_from_binary_feedback(
        model, z_episodes, z_labels, set_len, device
    )

    print(f"Predicting rewards for {len(pred_episodes)} episodes...")
    predictions = predict_rewards_with_z(
        model, pred_episodes, z_estimated, device
    )

    return z_estimated, predictions, pred_indices
