import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def log_metrics(metrics, epoch, logger):
    # Compute mean for list values
    processed_metrics = {}
    for key, val in metrics.items():
        if isinstance(val, list):
            processed_metrics[key] = np.mean(val)
        else:
            processed_metrics[key] = val
    # Log using ExperimentLogger.log_metrics
    logger.log_metrics(processed_metrics, step=epoch)

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
    