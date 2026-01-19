import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
from torch.utils.data import Dataset
from collections import defaultdict
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from src.utils.data_loader import DatasetManager
from src.utils.utils import _load_dataset_sequences

class PreferenceDataset(Dataset):
    def __init__(self, pref_dataset):
        self.pref_dataset = pref_dataset

    def __len__(self):
        return len(self.pref_dataset["observations"])

    def __getitem__(self, idx):
        observations = self.pref_dataset["observations"][idx]
        observations_2 = self.pref_dataset["observations_2"][idx]
        labels = self.pref_dataset["labels"][idx]
        
        # driver_name이 있다면 함께 반환 (없으면 무시)
        item = dict(
            observations=observations, 
            observations_2=observations_2, 
            labels=labels
        )
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
        """데이터셋에 포함된 고유한 드라이버 이름 리스트를 반환"""
        if "driver_name" not in self.pref_dataset:
            return []
        
        # driver_name shape: (N_queries, Context_Size)
        # 쿼리 내부(Context)는 동일한 드라이버라고 가정하므로, 
        # 첫 번째 컬럼[:, 0]만 봐도 충분합니다.
        all_names = self.pref_dataset["driver_name"][:, 0]
        return np.unique(all_names)

    # [추가된 메서드]
    def get_driver_data(self, driver_name):
        """특정 드라이버의 데이터만 필터링하여 딕셔너리로 반환"""
        if "driver_name" not in self.pref_dataset:
            return None
            
        # 1. 해당 드라이버의 인덱스 찾기
        # (모든 쿼리의 첫 번째 데이터 포인트의 드라이버 이름을 확인)
        drivers = self.pref_dataset["driver_name"][:, 0]
        indices = np.where(drivers == driver_name)[0]
        
        if len(indices) == 0:
            return None
            
        # 2. 해당 인덱스의 데이터만 슬라이싱하여 반환
        subset = {}
        for key, val in self.pref_dataset.items():
            subset[key] = val[indices]
            
        return subset

def create_vpl_dataset(train_driver_names, test_driver_names, features, time_range, downsample, tie_ratio=0.0, context_size=64, normalize=True):
    manager = DatasetManager("datasets", downsample=downsample)
    available_drivers = set(manager.keys())
    target_drivers = sorted(list(set(train_driver_names + test_driver_names)))
    
    config = {'features': features}

    # Step 1: Load data and Normalization
    print(f"Loading raw data to compute normalization stats...")
    raw_data_cache = {}
    all_observations = []

    for driver_name in target_drivers:
        if driver_name not in available_drivers:
            continue
            
        X, y = _load_dataset_sequences(driver_name, time_range, downsample, config)
        
        # 유효 데이터 체크
        if len(X[y==1]) == 0 or len(X[y==0]) == 0:
            print(f"Skipping {driver_name} for stats (insufficient data)")
            continue
            
        raw_data_cache[driver_name] = (X, y)
        all_observations.append(X)
    
    if not all_observations:
        raise ValueError("No valid data found for any driver!")

    # 전체 데이터에 대해 Mean, Std 계산
    concat_obs = np.concatenate(all_observations, axis=0) # (Total_N, T, D)
    # Feature 차원(axis=2)을 제외한 나머지 차원에 대해 평균 계산
    # shape: (D,)
    mean = np.mean(concat_obs, axis=(0, 1))
    std = np.std(concat_obs, axis=(0, 1)) + 1e-6 # 0으로 나누기 방지

    print("Data Normalization Stats:")
    print(f"  Mean: {mean}")
    print(f"  Std : {std}")

def convert_to_pairwise(X, y, driver_name, context_size=64, tie_ratio=0.0):
    """
    Raw Sequences (X, y)를 받아서 Pairwise Preference Dataset 포맷으로 변환
    """
    true_episodes = X[y == 1]
    false_episodes = X[y == 0]

    # Pair 생성
    driver_obs, driver_obs_2, driver_labels = [], [], []

    # (1) True vs False
    for true_ep in true_episodes:
        for false_ep in false_episodes:
            driver_obs.append(true_ep)
            driver_obs_2.append(false_ep)
            driver_labels.append(1.0)
    
    # (2) Tie Pairs
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

    if not driver_obs: # Pair가 하나도 안 만들어진 경우
        return None

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
            # 부족하면 랜덤하게 채움
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


def create_vpl_dataset(train_driver_names, test_driver_names, features, time_range, downsample, tie_ratio=0.0, context_size=64, normalize=True):
    manager = DatasetManager("datasets", downsample=downsample)
    available_drivers = set(manager.keys())
    target_drivers = sorted(list(set(train_driver_names + test_driver_names)))
    
    config = {'features': features}

    # Step 1: Load data and Normalization
    print(f"Loading raw data to compute normalization stats...")
    raw_data_cache = {}
    all_observations = []

    for driver_name in target_drivers:
        if driver_name not in available_drivers:
            continue
            
        X, y = _load_dataset_sequences(driver_name, time_range, downsample, config)
        
        # 유효 데이터 체크
        if len(X[y==1]) == 0 or len(X[y==0]) == 0:
            print(f"Skipping {driver_name} for stats (insufficient data)")
            continue
            
        raw_data_cache[driver_name] = (X, y)
        all_observations.append(X)
    
    if not all_observations:
        raise ValueError("No valid data found for any driver!")

    # 전체 데이터에 대해 Mean, Std 계산
    concat_obs = np.concatenate(all_observations, axis=0) # (Total_N, T, D)
    mean = np.mean(concat_obs, axis=(0, 1))
    std = np.std(concat_obs, axis=(0, 1)) + 1e-6 

    print("Data Normalization Stats:")
    print(f"  Mean: {mean}")
    print(f"  Std : {std}")

    # [Step 2] 정규화 적용 및 데이터셋 생성
    train_queries = defaultdict(list)
    test_driver_data = {}

    print(f"Creating datasets with context_size={context_size}...")

    for driver_name in target_drivers:
        if driver_name not in raw_data_cache:
            continue

        X_raw, y = raw_data_cache[driver_name]
        
        # ★ 정규화 (Standardization) ★
        X = (X_raw - mean) / std

        # Test Driver: Raw Data (X, y) 저장 (추후 Split하여 사용)
        if driver_name in test_driver_names:
            test_driver_data[driver_name] = (X, y)
            print(f"  {driver_name}: Saved as raw data ({len(X)} episodes)")
            continue

        # Train Driver: Pairwise 변환
        dataset_dict = convert_to_pairwise(X, y, driver_name, context_size, tie_ratio)
        
        if dataset_dict is None:
            print(f"  Skipping {driver_name}: No pairs generated.")
            continue
            
        num_queries = len(dataset_dict['observations'])
        print(f"  {driver_name}: {num_queries} queries generated.")

        train_queries['observations'].extend(dataset_dict['observations'])
        train_queries['observations_2'].extend(dataset_dict['observations_2'])
        train_queries['labels'].extend(dataset_dict['labels'])
        train_queries['driver_name'].extend(dataset_dict['driver_name'])

    train_dataset_dict = {}
    if train_queries['observations']:
        train_dataset_dict['observations'] = np.stack(train_queries['observations'])
        train_dataset_dict['observations_2'] = np.stack(train_queries['observations_2'])
        train_dataset_dict['labels'] = np.stack(train_queries['labels'])
        train_dataset_dict['driver_name'] = np.stack(train_queries['driver_name'])
        
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