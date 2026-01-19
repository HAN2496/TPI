import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from src.utils.data_loader import DatasetManager
from src.utils.utils import _load_dataset_sequences

class PreferenceDataset(Dataset):
    def __init__(self, pref_dataset):
        self.pref_dataset = pref_dataset

    def __len__(self):
        return len(self.pref_dataset["observations"])

    def __getitem__(self, idx):
        item = {
            'observations': self.pref_dataset["observations"][idx],
            'observations_2': self.pref_dataset["observations_2"][idx],
            'labels': self.pref_dataset["labels"][idx]
        }
        if "driver_name" in self.pref_dataset:
            item['driver_name'] = self.pref_dataset["driver_name"][idx]
        return item

    def get_driver_data(self, target_name):
        if "driver_name" not in self.pref_dataset:
            return None
        all_names = self.pref_dataset["driver_name"][:, 0] 
        indices = np.where(all_names == target_name)[0]
        
        if len(indices) == 0:
            return None
            
        return {
            'observations': self.pref_dataset["observations"][indices],
            'observations_2': self.pref_dataset["observations_2"][indices],
            'labels': self.pref_dataset["labels"][indices],
            'driver_name': target_name
        }


def create_vpl_dataset(train_driver_names, test_driver_names, features, time_range, downsample, tie_ratio=0.0, context_size=64):
    manager = DatasetManager("datasets", downsample=downsample)
    available_drivers = set(manager.keys())
    target_drivers = sorted(list(set(train_driver_names + test_driver_names)))
    
    config = {'features': features}

    # ---------------------------------------------------------
    # [Step 1] 데이터 로드 및 정규화(Normalization) 통계량 계산
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # [Step 2] 정규화 적용 및 데이터셋 생성
    # ---------------------------------------------------------
    train_queries = defaultdict(list)
    test_driver_data = {}

    print(f"Creating datasets with context_size={context_size}...")

    for driver_name in target_drivers:
        if driver_name not in raw_data_cache:
            continue

        X_raw, y = raw_data_cache[driver_name]
        
        # ★ 정규화 (Standardization) ★
        # Broadcasting: (N, T, D) - (D,)
        X = (X_raw - mean) / std

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
            true_indices = np.random.choice(len(true_episodes), size=(n_tie, 2))
            for i1, i2 in true_indices:
                driver_obs.append(true_episodes[i1])
                driver_obs_2.append(true_episodes[i2])
                driver_labels.append(0.5)
            
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
                needed = context_size - current_len
                indices = np.random.randint(0, current_len, size=needed)
                
                batch_obs = np.concatenate([batch_obs, batch_obs[indices]], axis=0)
                batch_obs_2 = np.concatenate([batch_obs_2, batch_obs_2[indices]], axis=0)
                batch_lbl = np.concatenate([batch_lbl, batch_lbl[indices]], axis=0)
            
            grouped_obs.append(batch_obs)
            grouped_obs_2.append(batch_obs_2)
            grouped_labels.append(batch_lbl)
            grouped_names.append(np.array([driver_name] * context_size))

        if driver_name in test_driver_names:
            test_driver_data[driver_name] = {
                'observations': np.stack(grouped_obs),
                'observations_2': np.stack(grouped_obs_2),
                'labels': np.stack(grouped_labels),
                'driver_name': np.stack(grouped_names)
            }
        elif driver_name in train_driver_names:
            train_queries['observations'].extend(grouped_obs)
            train_queries['observations_2'].extend(grouped_obs_2)
            train_queries['labels'].extend(grouped_labels)
            train_queries['driver_name'].extend(grouped_names)

        print(f"  {driver_name}: {num_pairs} pairs -> {len(grouped_obs)} queries")

    train_dataset_dict = {}
    if train_queries:
        train_dataset_dict['observations'] = np.stack(train_queries['observations'])
        train_dataset_dict['observations_2'] = np.stack(train_queries['observations_2'])
        train_dataset_dict['labels'] = np.stack(train_queries['labels'])
        train_dataset_dict['driver_name'] = np.stack(train_queries['driver_name'])
        
    return train_dataset_dict, test_driver_data

# --- Utils for inference (기존과 동일하게 유지) ---
def compute_step_rewards(model, X, z_mean, device):
    N, T, d = X.shape
    obs = torch.from_numpy(X).float().to(device)
    z = torch.from_numpy(z_mean).float().to(device)
    z_expanded = z.view(1, 1, 1, -1).expand(N, 1, T, -1)
    obs = obs.unsqueeze(1) 
    with torch.no_grad():
        step_rewards = model.decode(obs, z_expanded)
    return step_rewards.squeeze(1).squeeze(-1).cpu().numpy()