import os
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

from src.vmc.env.plant import ErideEnv
from src.vmc.controller import PController
from src.vmc.oracle.oracle import OracleBuilder

# --- Configuration ---
DATASET_DIR = "datasets/vmc"
OS_EXISTS = os.path.exists(DATASET_DIR)
if not OS_EXISTS:
    os.makedirs(DATASET_DIR)

# Feature selection for the dataset
# We will save these features for each step
FEATURE_KEYS = [
    # State
    "dz_com", "dtheta", "dz_us_f", "dz_us_r", "dx_com", 
    "z_com", "theta", "z_us_f", "z_us_r", "x_com",
    # Acceleration
    "ddz_com", "ddtheta", "ddz_us_f", "ddz_us_r", "ddx_com",
    # Control
    "u_eride"
]

def get_feature_vector(env_info):
    """Extracts the selected features from the environment info dictionary."""
    # env_info contains 'state', 'state_ddot', 'u_eride', etc.
    # We flatten them into a single vector
    state = env_info['state'] # Merged dict of state, ddot, dddot, u_eride
    
    vec = []
    for key in FEATURE_KEYS:
        if key in state:
            vec.append(state[key])
        elif key == "u_eride": # explicit check if not in state dict
            vec.append(state.get("u_eride", 0.0))
        else:
            vec.append(0.0)
    return np.array(vec, dtype=np.float32)

def generate_trajectory(env, controller, seed=None):
    """Runs a single episode and returns the trajectory (features)."""
    obs, info = env.reset(seed=seed)
    controller.reset()
    
    trajectory = []
    
    # Initial state
    trajectory.append(get_feature_vector(info))
    
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        u = controller(env.obs_dict, env.state)
        
        obs, reward, terminated, truncated, info = env.step([u])
        trajectory.append(get_feature_vector(info))
        
    return np.array(trajectory) # (T, F)

def create_synthetic_drivers():
    drivers = {}
    drivers['Comfort_strong'] = OracleBuilder().add_dtheta_threshold(weight=0.001, threshold=0.0, exp_scale=15.0).build()    
    drivers['Comfort_weak'] = OracleBuilder().add_dtheta_threshold(weight=0.001, threshold=0.0, exp_scale=10.0).build()    
    drivers['Aggressive_weak'] = OracleBuilder().add_pitch(weight=0.05).add_longitudinal(weight=0.1).build()    
    drivers['Aggressive_strong'] = OracleBuilder().add_pitch(weight=0.01).add_longitudinal(weight=0.1).build()
    return drivers

def process_single_pair(args):
    """하나의 에피소드 쌍을 생성하고 데이터를 반환합니다."""
    name, oracle, seed = args
    
    # Controller 설정 (랜덤)
    kp_a = float(np.random.uniform(30, 300))
    controller_a = PController(kp=kp_a)
    env_a = ErideEnv(controller_a)

    kp_b = float(np.random.uniform(30, 300))
    controller_b = PController(kp=kp_b)
    env_b = ErideEnv(controller_b)
    
    # Trajectory 생성
    traj_a = generate_trajectory(env_a, controller_a, seed=seed)
    traj_b = generate_trajectory(env_b, controller_b, seed=seed)
    
    # 길이 맞춤 (둘 중 짧은 쪽으로)
    min_len = min(len(traj_a), len(traj_b))
    traj_a = traj_a[:min_len]
    traj_b = traj_b[:min_len]
    
    # Labeling & Reward
    def to_dict(traj):
        return {k: traj[:, i] for i, k in enumerate(FEATURE_KEYS)}
    
    dict_a = to_dict(traj_a)
    dict_b = to_dict(traj_b)
    
    r_a = oracle.calculate_episode_reward(dict_a)
    r_b = oracle.calculate_episode_reward(dict_b)
    label = oracle.compare(dict_a, dict_b)
    
    return traj_a, traj_b, label, name, (r_a, r_b), (kp_a, kp_b)

def collect_data(num_pairs_per_driver, num_workers):
    drivers = create_synthetic_drivers()
    
    # 1. 작업 목록 생성 (Flatten tasks)
    tasks = []
    for name, oracle in drivers.items():
        for i in range(num_pairs_per_driver):
            tasks.append((name, oracle, i)) # (Driver, Oracle, Seed)
    
    print(f"Generating {len(tasks)} pairs using {num_workers}/{cpu_count()} cores...")

    # 2. 병렬 처리 실행
    # imap을 사용하여 tqdm과 연동
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(process_single_pair, tasks), total=len(tasks)))
    
    # 3. 데이터 정리 (Unzip results)
    # results는 [(traj_a, traj_b, label, name, rewards, params), ...] 형태임
    obs_a_list, obs_b_list, labels_list, names_list, rewards_list, params_list = zip(*results)

    # 4. 전체 데이터셋 최소 길이로 맞춤 (배치 처리를 위해 필수)
    lengths = [len(x) for x in obs_a_list]
    global_min_len = min(lengths)
    print(f"Trimming all trajectories to global min length: {global_min_len}")
    
    dataset = {
        'observations': np.stack([x[:global_min_len] for x in obs_a_list]),
        'observations_2': np.stack([x[:global_min_len] for x in obs_b_list]),
        'labels': np.array(labels_list, dtype=np.float32).reshape(-1, 1),
        'driver_name': np.array(names_list).reshape(-1, 1),
        'rewards': np.array(rewards_list, dtype=np.float32),
        'params': np.array(params_list, dtype=np.float32)
    }

    return dataset

def visualize_dataset(dataset, save_dir="artifacts/vmc_vpl/datasets"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"Visualizing dataset to {save_dir}...")
    
    driver_names = np.unique(dataset['driver_name'])
    
    # 1. Label Statistics
    stats_path = os.path.join(save_dir, "label_stats.txt")
    print("Computing label statistics...")
    with open(stats_path, "w") as f:
        f.write("Driver Label Statistics:\n")
        print("Driver Label Statistics:")
        
        for name in driver_names:
            mask = (dataset['driver_name'][:, 0] == name)
            labels = dataset['labels'][mask].flatten()
            
            n_total = len(labels)
            n_0 = np.sum(labels == 0)
            n_1 = np.sum(labels == 1)
            ratio_0 = n_0 / n_total * 100
            ratio_1 = n_1 / n_total * 100
            
            stat_line = f"  {name}: Total={n_total}, Label 0={n_0} ({ratio_0:.1f}%), Label 1={n_1} ({ratio_1:.1f}%)"
            f.write(stat_line + "\n")
            print(stat_line)
            
    # 2. KP vs Reward Scatter (Single Column)
    fig, axs = plt.subplots(len(driver_names), 1, figsize=(8, 4 * len(driver_names)), constrained_layout=True)
    if len(driver_names) == 1: axs = [axs] 

    for i, name in enumerate(driver_names):
        mask = (dataset['driver_name'][:, 0] == name)
        params = dataset['params'][mask].flatten() # [kp_a, kp_b, kp_a, ...]
        rewards = dataset['rewards'][mask].flatten()
        
        ax = axs[i]
        ax.scatter(params, rewards, alpha=0.7, s=10)
        ax.set_title(f"Kp vs Reward ({name})")
        ax.set_xlabel("Kp (P-Gain)")
        ax.set_ylabel("Reward")
        ax.grid(True, alpha=0.3)
        
    plt.savefig(os.path.join(save_dir, "kp_analysis.png"))
    plt.close()
    
    # 3. Reward Distribution Histogram
    fig, axs = plt.subplots(len(driver_names), 1, figsize=(8, 4 * len(driver_names)), constrained_layout=True)
    if len(driver_names) == 1: axs = [axs]
    
    for i, name in enumerate(driver_names):
        mask = (dataset['driver_name'][:, 0] == name)
        rewards = dataset['rewards'][mask].flatten()
        
        ax = axs[i]
        ax.hist(rewards, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_title(f"Reward Distribution ({name})")
        ax.set_xlabel("Reward")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(save_dir, "reward_dist.png"))
    plt.close()
    
    # 4. Average Trajectory Comparison (Label 0 vs Label 1)
    feature_indices = {
        'dtheta': FEATURE_KEYS.index('dtheta'),
        'ddx_com': FEATURE_KEYS.index('ddx_com'),
        'u_eride': FEATURE_KEYS.index('u_eride')
    }
    
    for name in driver_names:
        mask = (dataset['driver_name'][:, 0] == name)
        labels = dataset['labels'][mask].flatten()
        
        obs_a = dataset['observations'][mask] # (N, T, F)
        obs_b = dataset['observations_2'][mask]
        
        # Collect trajectories based on preference
        # Label 1: A is better -> Preferred: A, Non-Preferred: B
        # Label 0: B is better -> Preferred: B, Non-Preferred: A
        
        pref_trajs = []
        non_pref_trajs = []
        
        for idx, lbl in enumerate(labels):
            if lbl == 1:
                pref_trajs.append(obs_a[idx])
                non_pref_trajs.append(obs_b[idx])
            else:
                pref_trajs.append(obs_b[idx])
                non_pref_trajs.append(obs_a[idx])
                
        if not pref_trajs: continue
        
        pref_trajs = np.stack(pref_trajs)
        non_pref_trajs = np.stack(non_pref_trajs)
        
        fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        t = np.arange(pref_trajs.shape[1])
        
        for i, (feat_name, feat_idx) in enumerate(feature_indices.items()):
            # Calculate Mean & Std
            mu_p = pref_trajs[:, :, feat_idx].mean(axis=0)
            std_p = pref_trajs[:, :, feat_idx].std(axis=0)
            
            mu_np = non_pref_trajs[:, :, feat_idx].mean(axis=0)
            std_np = non_pref_trajs[:, :, feat_idx].std(axis=0)
            
            # Plot Preferred
            ax[i].plot(t, mu_p, label="Preferred (Mean)", color='blue')
            ax[i].fill_between(t, mu_p - std_p, mu_p + std_p, color='blue', alpha=0.1)
            
            # Plot Non-Preferred
            ax[i].plot(t, mu_np, label="Non-Preferred (Mean)", color='red', linestyle='--')
            ax[i].fill_between(t, mu_np - std_np, mu_np + std_np, color='red', alpha=0.1)
            
            ax[i].set_ylabel(feat_name)
            ax[i].grid(True, alpha=0.3)
            
        ax[0].legend()
        ax[0].set_title(f"Average Trajectory Comparison - {name}")
        ax[-1].set_xlabel("Time Step")
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"traj_comparison_avg_{name}.png"))
        plt.close()

    # 5. Reward Difference vs Label Probability (Oracle Check)
    plt.figure(figsize=(8, 6))
    
    all_diffs = []
    all_labels = []
    
    for name in driver_names:
        mask = (dataset['driver_name'][:, 0] == name)
        r_a = dataset['rewards'][mask][:, 0]
        r_b = dataset['rewards'][mask][:, 1]
        lbls = dataset['labels'][mask].flatten()
        
        diffs = r_a - r_b
        all_diffs.extend(diffs)
        all_labels.extend(lbls)
        
    all_diffs = np.array(all_diffs)
    all_labels = np.array(all_labels)
    
    # Binning
    bins = np.linspace(all_diffs.min(), all_diffs.max(), 30)
    bin_indices = np.digitize(all_diffs, bins)
    
    bin_means = []
    prob_means = []
    
    for i in range(1, len(bins)):
        in_bin = all_labels[bin_indices == i]
        if len(in_bin) > 0:
            bin_means.append((bins[i-1] + bins[i]) / 2)
            prob_means.append(in_bin.mean())
            
    plt.scatter(all_diffs, all_labels, alpha=0.05, label='Samples', color='gray', s=5)
    plt.plot(bin_means, prob_means, 'r-o', label='Empirical Prob', linewidth=2)
    
    # Theoretical Sigmoid
    x = np.linspace(all_diffs.min(), all_diffs.max(), 100)
    y = 1 / (1 + np.exp(-x))
    plt.plot(x, y, 'b--', label='Sigmoid (Theoretical)')
    
    plt.title("Oracle Check: Reward Diff vs P(Label=1)")
    plt.xlabel("Reward Difference (R_a - R_b)")
    plt.ylabel("Probability (Label=1)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(save_dir, "oracle_check.png"))
    plt.close()

    print("Visualization complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_pairs", type=int, default=5000, help="Number of pairs per driver")
    parser.add_argument("--num_workers", type=int, default=6, help="Number of parallel workers")
    args = parser.parse_args()

    dataset = collect_data(args.num_pairs, args.num_workers)
    
    save_path = os.path.join(DATASET_DIR, f"vmc_pairwise_dataset_{args.num_pairs}.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"Dataset saved to {save_path}")
    print(f"Shapes:")
    print(f"  Obs: {dataset['observations'].shape}")
    print(f"  Obs2: {dataset['observations_2'].shape}")
    print(f"  Labels: {dataset['labels'].shape}")
    print(f"  Drivers: {dataset['driver_name'].shape}")
    
    visualize_dataset(dataset)
