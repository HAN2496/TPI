import os

import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

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

def collect_data(num_pairs_per_driver=1000):
    drivers = create_synthetic_drivers()
    
    dataset = {
        'observations': [],
        'observations_2': [],
        'labels': [],
        'driver_name': [],
        'rewards': [], # Store calculated rewards for visualization
        'params': []   # Store controller params (kp_a, kp_b)
    }
    
    print(f"Generating {num_pairs_per_driver} pairs for {len(drivers)} drivers...")
    
    for name, oracle in drivers.items():
        print(f"  Processing Driver: {name}")
        
        for i in tqdm(range(num_pairs_per_driver)):
            # 1. Setup Context (Seed)
            seed = i
            # seed = np.random.randint(0, 100000)
            
            # 2. Controller A
            kp_a = float(np.random.uniform(30, 300))
            controller_a = PController(kp=kp_a)
            env_a = ErideEnv(controller_a)

            # 3. Controller B
            kp_b = float(np.random.uniform(30, 300))
            controller_b = PController(kp=kp_b)
            env_b = ErideEnv(controller_b)
            
            # 4. Generate Trajectories
            traj_a = generate_trajectory(env_a, controller_a, seed=seed)
            traj_b = generate_trajectory(env_b, controller_b, seed=seed)
            
            # Truncate to min length
            min_len = min(len(traj_a), len(traj_b))
            traj_a = traj_a[:min_len]
            traj_b = traj_b[:min_len]
            
            # 5. Labeling & Reward Calculation
            def to_dict(traj):
                return {k: traj[:, i] for i, k in enumerate(FEATURE_KEYS)}
            
            dict_a = to_dict(traj_a)
            dict_b = to_dict(traj_b)
            
            # Calculate explicit rewards for metadata/visualization
            r_a = oracle.calculate_episode_reward(dict_a)
            r_b = oracle.calculate_episode_reward(dict_b)
            
            label = oracle.compare(dict_a, dict_b) # 1 if A > B, 0 if B > A
            
            # 6. Store
            dataset['observations'].append(traj_a)
            dataset['observations_2'].append(traj_b)
            dataset['labels'].append([float(label)])
            dataset['driver_name'].append(name)
            dataset['rewards'].append((r_a, r_b))
            dataset['params'].append((kp_a, kp_b))

    # Convert to numpy arrays
    lengths = [len(x) for x in dataset['observations']]
    min_len = min(lengths)
    print(f"  Trimming all trajectories to min length: {min_len}")
    
    dataset['observations'] = np.stack([x[:min_len] for x in dataset['observations']])
    dataset['observations_2'] = np.stack([x[:min_len] for x in dataset['observations_2']])
    dataset['labels'] = np.array(dataset['labels'], dtype=np.float32)
    dataset['driver_name'] = np.array(dataset['driver_name']).reshape(-1, 1)
    dataset['rewards'] = np.array(dataset['rewards'], dtype=np.float32)
    dataset['params'] = np.array(dataset['params'], dtype=np.float32)

    return dataset

def visualize_dataset(dataset, save_dir="artifacts/vmc/datasets/plots"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"Visualizing dataset to {save_dir}...")
    
    driver_names = np.unique(dataset['driver_name'])
    
    # 2. KP vs Reward Scatter
    fig, axs = plt.subplots(len(driver_names), 2, figsize=(16, 4 * len(driver_names)), constrained_layout=True)
    if len(driver_names) == 1: axs = [axs] # Ensure 2D array-like access if single driver
    
    # Feature index for dtheta
    dtheta_idx = FEATURE_KEYS.index('dtheta')

    for i, name in enumerate(driver_names):
        mask = (dataset['driver_name'][:, 0] == name)
        params = dataset['params'][mask].flatten() # [kp_a, kp_b, kp_a, ...]
        rewards = dataset['rewards'][mask].flatten()
        
        # Calculate Mean Squared Pitch Rate for each trajectory
        obs_a = dataset['observations'][mask]
        obs_b = dataset['observations_2'][mask]
        
        # Calculate mean(dtheta^2) for each trajectory
        pitch_cost_a = np.mean(obs_a[:, :, dtheta_idx]**2, axis=1)
        pitch_cost_b = np.mean(obs_b[:, :, dtheta_idx]**2, axis=1)
        
        pitch_costs = np.concatenate([pitch_cost_a, pitch_cost_b]) # Same order as params flatten
        
        # Plot 1: Kp vs Reward
        ax_rew = axs[i][0] if len(driver_names) > 1 else axs[0][0]
        ax_rew.scatter(params, rewards, alpha=0.3, s=10)
        ax_rew.set_title(f"Kp vs Reward ({name})")
        ax_rew.set_xlabel("Kp (P-Gain)")
        ax_rew.set_ylabel("Reward")
        ax_rew.grid(True, alpha=0.3)
        
        # Plot 2: Kp vs Pitch Cost
        ax_cost = axs[i][1] if len(driver_names) > 1 else axs[0][1]
        ax_cost.scatter(params, pitch_costs, alpha=0.3, s=10, color='orange')
        ax_cost.set_title(f"Kp vs Mean Pitch Cost (dtheta^2) ({name})")
        ax_cost.set_xlabel("Kp (P-Gain)")
        ax_cost.set_ylabel("Mean dtheta^2")
        ax_cost.set_yscale('log') # Log scale for cost as it can vary widely
        ax_cost.grid(True, alpha=0.3)
        
    plt.savefig(os.path.join(save_dir, "kp_analysis.png"))
    plt.close()

    # 3. Trajectory Examples (Good vs Bad)
    # Plot feature trajectories for high reward vs low reward examples
    feature_indices = {
        'dtheta': FEATURE_KEYS.index('dtheta'),
        'ddx_com': FEATURE_KEYS.index('ddx_com'),
        'u_eride': FEATURE_KEYS.index('u_eride')
    }
    
    for name in driver_names:
        mask = (dataset['driver_name'][:, 0] == name)
        driver_indices = np.where(mask)[0]
        
        subset_rewards = dataset['rewards'][mask] # (N_subset, 2)
        
        # Best
        best_idx_flat = np.argmax(subset_rewards)
        best_pair_idx = driver_indices[best_idx_flat // 2]
        best_is_b = best_idx_flat % 2
        
        # Worst
        worst_idx_flat = np.argmin(subset_rewards)
        worst_pair_idx = driver_indices[worst_idx_flat // 2]
        worst_is_b = worst_idx_flat % 2
        
        best_traj = dataset['observations_2'][best_pair_idx] if best_is_b else dataset['observations'][best_pair_idx]
        worst_traj = dataset['observations_2'][worst_pair_idx] if worst_is_b else dataset['observations'][worst_pair_idx]
        
        best_kp = dataset['params'][best_pair_idx][best_is_b]
        worst_kp = dataset['params'][worst_pair_idx][worst_is_b]
        
        fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        t = np.arange(len(best_traj))
        
        for i, (feat_name, feat_idx) in enumerate(feature_indices.items()):
            ax[i].plot(t, best_traj[:, feat_idx], label=f"Best (Rew={np.max(subset_rewards):.2f}, Kp={best_kp:.1f})", color='tab:green')
            ax[i].plot(t, worst_traj[:, feat_idx], label=f"Worst (Rew={np.min(subset_rewards):.2f}, Kp={worst_kp:.1f})", color='tab:red', alpha=0.7)
            ax[i].set_ylabel(feat_name)
            ax[i].grid(True, alpha=0.3)
            
        ax[0].legend()
        ax[0].set_title(f"Trajectory Comparison - {name}")
        ax[-1].set_xlabel("Time Step")
        
        plt.savefig(os.path.join(save_dir, f"traj_example_{name}.png"))
        plt.close()
    
    print("Visualization complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_pairs", type=int, default=5000, help="Number of pairs per driver")
    args = parser.parse_args()

    dataset = collect_data(num_pairs_per_driver=args.num_pairs)
    
    save_path = os.path.join(DATASET_DIR, "vmc_pairwise_dataset.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"Dataset saved to {save_path}")
    print(f"Shapes:")
    print(f"  Obs: {dataset['observations'].shape}")
    print(f"  Obs2: {dataset['observations_2'].shape}")
    print(f"  Labels: {dataset['labels'].shape}")
    print(f"  Drivers: {dataset['driver_name'].shape}")
    
    visualize_dataset(dataset)
