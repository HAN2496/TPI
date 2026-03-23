import os
import json
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from src.model.vpl.vae import VAEModel
from src.model.vpl.trainer import VPLTrainer, Annealer
from src.model.vpl.utils import PreferenceDataset, compute_step_rewards
from src.utils.logger import ExperimentLogger

# --- Configuration ---
FLAGS = {
    # Data
    "dataset_name": 5000,
    "test_driver_name": "Aggressive_weak",
    "train_driver_names": ["Comfort_strong", "Comfort_weak", "Aggressive_strong"],
    "features": ["ddx_com", "dtheta"], # Selected features

    # Preprocessing
    "downsample": 20,      # Reduce 1000 steps to ~50
    "context_size": 50,    # Number of pairs per context batch
    "val_split": 0.1,      # Validation split from training drivers

    # Model
    "hidden_dim": 64,
    "batch_size": 128,
    "latent_dim": 8,
    "kl_weight": 0.001,
    "flow_prior": False,
    "use_annealing": False,
    "annealer_baseline": 0.0,
    "annealer_type": "cosine",
    "annealer_cycles": 4,
    "reward_scaling": "T", # 1 or T
    
    # Training
    "batch_size": 32,
    "n_epochs": 300,
    "lr": 3e-4,
    "weight_decay": 0.0,
    "early_stop": False,
    "patience": 15,
    "min_delta": 1e-4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    # Inference
    "eval_freq": 10,
    "timestamp": "test",   # None: new training, "test": debug training, else: inference only

    # Preprocessing
    "normalize": True,
}

# Must match generate_vmc_datasets.py
FEATURE_KEYS = [
    # State
    "dz_com", "dtheta", "dz_us_f", "dz_us_r", "dx_com", 
    "z_com", "theta", "z_us_f", "z_us_r", "x_com",
    # Acceleration
    "ddz_com", "ddtheta", "ddz_us_f", "ddz_us_r", "ddx_com",
    # Control
    "u_eride"
]

def collate_fn(batch):
    return {
        'observations': torch.stack([torch.from_numpy(item['observations']).float() for item in batch]),
        'observations_2': torch.stack([torch.from_numpy(item['observations_2']).float() for item in batch]),
        'labels': torch.stack([torch.from_numpy(item['labels']).float() for item in batch]),
        'driver_name': [item.get('driver_name') for item in batch]
    }

def get_feature_indices(selected_features):
    """Returns indices of selected features."""
    indices = []
    for f in selected_features:
        if f in FEATURE_KEYS:
            indices.append(FEATURE_KEYS.index(f))
        else:
            print(f"Warning: Feature '{f}' not found in dataset keys.")
    return indices

def group_data_by_driver(raw_data, selected_features=None):
    """Groups flat data by driver name and filters features."""
    grouped = defaultdict(lambda: defaultdict(list))
    
    feature_indices = None
    if selected_features:
        feature_indices = get_feature_indices(selected_features)
        print(f"Selected features: {selected_features} -> Indices: {feature_indices}")
    
    n_samples = len(raw_data['labels'])
    for i in range(n_samples):
        d_name = raw_data['driver_name'][i][0]
        
        obs = raw_data['observations'][i]
        obs2 = raw_data['observations_2'][i]
        
        # Filter features if indices are computed
        if feature_indices is not None:
            obs = obs[:, feature_indices]
            obs2 = obs2[:, feature_indices]
            
        grouped[d_name]['observations'].append(obs)
        grouped[d_name]['observations_2'].append(obs2)
        grouped[d_name]['labels'].append(raw_data['labels'][i])
        
    # Convert to numpy
    final_grouped = {}
    for d_name, data in grouped.items():
        final_grouped[d_name] = {k: np.stack(v) for k, v in data.items()}
        
    return final_grouped

def create_context_dataset(grouped_data, driver_names, context_size, downsample):
    """
    Chunks driver data into contexts.
    Returns a dictionary suitable for PreferenceDataset.
    """
    out = defaultdict(list)
    
    for d_name in driver_names:
        if d_name not in grouped_data:
            print(f"Warning: Driver {d_name} not found in dataset.")
            continue
            
        data = grouped_data[d_name]
        n_pairs = len(data['labels'])
        
        # Shuffle
        perm = np.random.permutation(n_pairs)
        
        # Downsample features: (N, T, F) -> (N, T//ds, F)
        obs = data['observations'][perm][:, ::downsample, :]
        obs2 = data['observations_2'][perm][:, ::downsample, :]
        lbl = data['labels'][perm]
        
        # Chunk into contexts
        for i in range(0, n_pairs, context_size):
            if i + context_size > n_pairs:
                break # Drop last incomplete batch
                
            batch_obs = obs[i:i+context_size]
            batch_obs2 = obs2[i:i+context_size]
            batch_lbl = lbl[i:i+context_size]
            
            out['observations'].append(batch_obs)
            out['observations_2'].append(batch_obs2)
            out['labels'].append(batch_lbl)
            out['driver_name'].append(np.array([d_name] * context_size))
            
    if not out:
        raise ValueError("No data created. Check driver names or dataset.")
        
    return {k: np.stack(v) for k, v in out.items()}

def estimate_latent(model, observations, observations_2, labels, device):
    """Infer z from a context batch."""
    model.eval()
    
    # Add batch dim if needed (1, Context, T, F)
    if observations.ndim == 3:
        obs1 = torch.from_numpy(observations).float().to(device).unsqueeze(0)
        obs2 = torch.from_numpy(observations_2).float().to(device).unsqueeze(0)
        y = torch.from_numpy(labels).float().to(device).unsqueeze(0)
    else:
        obs1 = torch.from_numpy(observations).float().to(device)
        obs2 = torch.from_numpy(observations_2).float().to(device)
        y = torch.from_numpy(labels).float().to(device)

    with torch.no_grad():
        mean, _ = model.encode(obs1, obs2, y)
        
    return mean.cpu().numpy().flatten() # (Latent,)

def adapt_and_evaluate(model, driver_data, driver_name, FLAGS):
    """
    Adapt to a new driver using a small query set and evaluate on the rest.
    """
    print(f"\nEvaluating adaptation for {driver_name}...")

    # Data is dict: obs, obs2, labels (N, T, F)
    n_total = len(driver_data['labels'])
    indices = np.random.permutation(n_total)

    # Use context_size for query (must match model's expected input dimension)
    context_size = FLAGS['context_size']
    query_idx = indices[:context_size]
    eval_idx = indices[context_size:]

    # Downsample
    ds = FLAGS['downsample']

    # Prepare Query Context
    q_obs = driver_data['observations'][query_idx][:, ::ds, :]
    q_obs2 = driver_data['observations_2'][query_idx][:, ::ds, :]
    q_lbl = driver_data['labels'][query_idx]

    # Infer z
    z_mean = estimate_latent(model, q_obs, q_obs2, q_lbl, FLAGS['device'])
    print(f"  Inferred z: {z_mean[:4]}...")
    
    # Evaluate
    # We predict preference for single items? Or pairs?
    # VPL decode gives reward R(tau). 
    # For pairs, we check if R(tau1) > R(tau2) matches label.
    
    eval_obs = driver_data['observations'][eval_idx][:, ::ds, :]
    eval_obs2 = driver_data['observations_2'][eval_idx][:, ::ds, :]
    eval_lbl = driver_data['labels'][eval_idx].flatten() # (N,)
    
    print(f"  Eval Stats: Total {len(eval_lbl)} | Label 0: {(eval_lbl==0).sum()} | Label 1: {(eval_lbl==1).sum()}")

    # Compute rewards
    # compute_step_rewards handles (N, T, F) and z (D,) -> returns (N, T)
    r1_seq = compute_step_rewards(model, eval_obs, z_mean, FLAGS['device'])
    r2_seq = compute_step_rewards(model, eval_obs2, z_mean, FLAGS['device'])
    
    r1 = r1_seq.sum(axis=1)
    r2 = r2_seq.sum(axis=1)
    
    # Prediction: 1 if r1 > r2, else 0
    preds = (r1 > r2).astype(float)
    
    acc = accuracy_score(eval_lbl, preds)
    # For AUROC, we need probabilities. P(1>2) = sigmoid(r1 - r2)
    probs = 1 / (1 + np.exp(-(r1 - r2))) # Assuming temperature=1 implicit in model? 
    # Model output is usually unscaled logits, VAE uses sigmoid(r1-r2) in forward.
    
    try:
        auroc = roc_auc_score(eval_lbl, probs)
    except:
        auroc = 0.5
        
    print(f"  Result: Accuracy = {acc:.4f}, AUROC = {auroc:.4f}")
    return acc, auroc, r1, r2, eval_lbl

def plot_history(metrics, save_dir, warmup_epochs=0):
    """Plot training history."""
    print(f"Plotting training history (excluding first {warmup_epochs} epochs)...")

    metric_groups = {}
    for key in metrics.keys():
        if '/' in key:
            prefix, name = key.split('/', 1)
        else:
            prefix, name = 'misc', key

        if name not in metric_groups:
            metric_groups[name] = []
        metric_groups[name].append(key)

    n_metrics = len(metric_groups)
    cols = 2
    rows = (n_metrics + 1) // cols
    if rows * cols < n_metrics:
        rows += 1

    plt.figure(figsize=(12, 4 * rows))

    for idx, (name, keys) in enumerate(metric_groups.items(), 1):
        plt.subplot(rows, cols, idx)
        for key in keys:
            values = metrics[key]
            if len(values) > warmup_epochs:
                plot_values = values[warmup_epochs:]
                plot_epochs = range(warmup_epochs + 1, warmup_epochs + 1 + len(plot_values))
            else:
                plot_values = values
                plot_epochs = range(1, len(values) + 1)

            if 'train' in key:
                color = 'tab:blue'
                label = 'Train'
            elif 'eval' in key:
                color = 'tab:orange'
                label = 'Eval'
            else:
                color = None
                label = key

            plt.plot(plot_epochs, plot_values, label=label, color=color, alpha=0.8)

        plt.title(f'{name.replace("_", " ").capitalize()}')
        plt.xlabel('Epochs')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'history_all.png')
    plt.close()


def visualize_latent_space(model, grouped_data, train_drivers, test_driver, device, save_path, downsample, context_size):
    """Visualize latent space using t-SNE and PCA."""
    print("Visualizing Latent Space...")

    latents = []
    labels = []

    for d_name in train_drivers + [test_driver]:
        if d_name not in grouped_data:
            continue

        data = grouped_data[d_name]
        n_pairs = len(data['labels'])

        # Process in context batches
        ds = downsample
        obs = data['observations'][:, ::ds, :]
        obs2 = data['observations_2'][:, ::ds, :]
        lbl = data['labels']

        for i in range(0, n_pairs, context_size):
            if i + context_size > n_pairs:
                break

            batch_obs = obs[i:i+context_size]
            batch_obs2 = obs2[i:i+context_size]
            batch_lbl = lbl[i:i+context_size]

            z = estimate_latent(model, batch_obs, batch_obs2, batch_lbl, device)
            latents.append(z)

            tag = "(Test)" if d_name == test_driver else "(Train)"
            labels.append(f"{d_name} {tag}")

    if not latents:
        print("No latents to visualize.")
        return

    latents = np.stack(latents)
    labels = np.array(labels)
    unique_labels = np.unique(labels)

    perplexities = [5, 10, 15]
    n_cols = len(perplexities) + 1

    fig, axs = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))

    def plot_scatter(ax, data, title):
        for label in unique_labels:
            mask = (labels == label)
            marker = '*' if "(Test)" in label else 'o'
            s = 100 if "(Test)" in label else 30
            alpha = 1.0 if "(Test)" in label else 0.7
            ax.scatter(data[mask, 0], data[mask, 1], label=label, marker=marker, s=s, alpha=alpha)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    # t-SNE plots
    for i, perp in enumerate(perplexities):
        actual_perp = min(perp, len(latents) - 1)
        if actual_perp < 2:
            continue
        tsne = TSNE(n_components=2, random_state=42, perplexity=actual_perp)
        z_tsne = tsne.fit_transform(latents)
        plot_scatter(axs[i], z_tsne, f"t-SNE (Perp={actual_perp})")

    # PCA plot
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(latents)
    var_ratio = pca.explained_variance_ratio_
    plot_scatter(axs[-1], z_pca, f"PCA (Var: [{var_ratio[0]:.2f}, {var_ratio[1]:.2f}])")

    axs[-1].legend(loc='upper right', bbox_to_anchor=(1.3, 1))

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Latent plot saved to {save_path}")


def plot_reward_distribution(r1, r2, labels, driver_name, save_path):
    """Plot reward distribution for pairs."""
    print(f"Plotting reward distribution for {driver_name}...")
    print(f"  Labels unique values: {np.unique(labels)}")

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: R1 vs R2 scatter
    colors = ['blue' if l == 0 else 'red' for l in labels]
    axs[0].scatter(r1, r2, c=colors, alpha=0.5, s=20)
    axs[0].plot([min(r1.min(), r2.min()), max(r1.max(), r2.max())],
                [min(r1.min(), r2.min()), max(r1.max(), r2.max())], 'k--', alpha=0.5)
    axs[0].set_xlabel('R(τ1)')
    axs[0].set_ylabel('R(τ2)')
    axs[0].set_title('Reward Comparison')
    axs[0].grid(True, alpha=0.3)

    # Plot 2: R1 - R2 histogram
    diff = r1 - r2
    axs[1].hist(diff[labels == 0], bins=20, alpha=0.5, label='Label 0', color='blue', density=True)
    axs[1].hist(diff[labels == 1], bins=20, alpha=0.5, label='Label 1', color='red', density=True)
    axs[1].axvline(x=0, color='k', linestyle='--', alpha=0.5)
    axs[1].set_xlabel('R(τ1) - R(τ2)')
    axs[1].set_ylabel('Density')
    axs[1].set_title('Reward Difference Distribution')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    # Plot 3: Predicted probability
    probs = 1 / (1 + np.exp(-diff))
    idx = np.arange(len(probs))
    mask_0 = (labels == 0)
    mask_1 = (labels == 1)
    axs[2].scatter(idx[mask_0], probs[mask_0], c='blue', alpha=0.5, label='Label 0', s=15)
    axs[2].scatter(idx[mask_1], probs[mask_1], c='red', alpha=0.5, label='Label 1', s=15)
    axs[2].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    axs[2].set_xlabel('Sample Index')
    axs[2].set_ylabel('P(τ1 > τ2)')
    axs[2].set_title('Preference Prediction')
    axs[2].set_ylim(-0.05, 1.05)
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)

    plt.suptitle(f'{driver_name} - Evaluation Results', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Reward plot saved to {save_path}")


def main():
    # Setup Logger with timestamp handling
    os.makedirs("artifacts/vmc_vpl", exist_ok=True)
    if FLAGS["timestamp"] == "test":
        timestamp = "test"
        log_dir = Path("artifacts/vmc_vpl") / timestamp
        is_training = True
    elif FLAGS["timestamp"] is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = Path("artifacts/vmc_vpl") / timestamp
        is_training = True
    else:
        timestamp = FLAGS["timestamp"]
        log_dir = Path("artifacts/vmc_vpl") / timestamp
        is_training = False
        print(f"Inference Mode: Loading model from {log_dir}")

    logger = ExperimentLogger(log_dir=log_dir, experiment_name="vmc_vpl_train", add_timestamp=False)
    print(f"Experiment Directory: {logger.log_dir}")

    if is_training:
        with open(logger.log_dir / "flags.json", "w", encoding='utf-8') as f:
            json.dump(FLAGS, f, indent=4, ensure_ascii=False)
    
    # 1. Load Data
    dataset_path = f"datasets/vmc/vmc_pairwise_dataset_{FLAGS['dataset_name']}.pkl"
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'rb') as f:
        raw_data = pickle.load(f)
    
    # 1.1 Feature Selection & Normalization
    feature_indices = get_feature_indices(FLAGS['features'])
    print(f"Selected features: {FLAGS['features']} -> Indices: {feature_indices}")
    
    # Select features immediately
    raw_data['observations'] = raw_data['observations'][:, :, feature_indices]
    raw_data['observations_2'] = raw_data['observations_2'][:, :, feature_indices]
    
    if FLAGS['normalize']:
        print("Normalizing data (StandardScaler)...")
        # Compute stats on all data (obs and obs2) to ensure consistent scaling
        all_obs = np.concatenate([raw_data['observations'], raw_data['observations_2']], axis=0)
        # Flatten to (N*T, F) for computing mean/std per feature
        mean = np.mean(all_obs, axis=(0, 1))
        std = np.std(all_obs, axis=(0, 1))
        std[std < 1e-6] = 1.0 # Prevent div by zero
        
        print(f"  Mean: {mean}")
        print(f"  Std : {std}")
        
        raw_data['observations'] = (raw_data['observations'] - mean) / std
        raw_data['observations_2'] = (raw_data['observations_2'] - mean) / std
        
    grouped_data = group_data_by_driver(raw_data, selected_features=None)
    
    # --- Debug: Print Label Statistics ---
    print("\nDataset Label Statistics:")
    for d_name, data in grouped_data.items():
        lbls = data['labels'].flatten()
        n_0 = (lbls == 0).sum()
        n_1 = (lbls == 1).sum()
        n_tie = (lbls == 0.5).sum()
        print(f"  {d_name}: Total {len(lbls)} | Label 0: {n_0} ({n_0/len(lbls):.1%}) | Label 1: {n_1} ({n_1/len(lbls):.1%}) | Tie: {n_tie}")
    print("-" * 30)

    # 2. Create Train Dataset (Contexts)
    print("Creating training contexts...")
    train_dict = create_context_dataset(
        grouped_data, FLAGS['train_driver_names'], 
        FLAGS['context_size'], FLAGS['downsample']
    )
    
    # Split validation contexts (simple split of batches)
    n_train_batches = len(train_dict['labels'])
    n_val = int(n_train_batches * FLAGS['val_split'])
    n_train = n_train_batches - n_val
    
    indices = np.random.permutation(n_train_batches)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    def subset(d, idx): return {k: v[idx] for k, v in d.items()}
    
    train_dataset = PreferenceDataset(subset(train_dict, train_idx))
    val_dataset = PreferenceDataset(subset(train_dict, val_idx))
    
    train_loader = DataLoader(train_dataset, batch_size=FLAGS['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=FLAGS['batch_size'], shuffle=False, collate_fn=collate_fn)
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # 3. Initialize Model
    # Get shape from a sample
    sample_obs = train_dict['observations'][0] # (Context, T, F)
    ctx_size, T, F = sample_obs.shape
    print(f"Input Shape: Context={ctx_size}, Time={T}, Features={F}")
    
    # Encoder Input Dim calculation (Flattened Context)
    # (2 * T * F + 1) per pair * Context_Size
    pair_dim = 2 * T * F + 1
    encoder_input = ctx_size * pair_dim
    decoder_input = F + FLAGS['latent_dim']
    
    annealer = None
    if FLAGS['use_annealing']:
        annealer = Annealer(
            total_steps=FLAGS['n_epochs'] // FLAGS['annealer_cycles'],
            shape=FLAGS['annealer_type'],
            cyclical=FLAGS['annealer_cycles'] > 1
        )

    model = VAEModel(
        encoder_input=encoder_input,
        decoder_input=decoder_input,
        latent_dim=FLAGS['latent_dim'],
        hidden_dim=FLAGS['hidden_dim'],
        annotation_size=ctx_size,
        size_segment=T,
        kl_weight=FLAGS['kl_weight'],
        flow_prior=FLAGS['flow_prior'],
        annealer=annealer
    ).to(FLAGS['device'])
    
    # 4. Train
    if is_training:
        trainer = VPLTrainer(model, logger, FLAGS)
        print("Starting training...")
        metrics, _ = trainer.train(train_loader, val_loader)

        # Plot training history
        plot_history(metrics, logger.log_dir, warmup_epochs=10)
    else:
        print("Skipping training (Inference Mode)...")

    # 5. Evaluate on Test Driver
    print("\n" + "="*50 + "\nStarting Evaluation\n" + "="*50)
    model_path = logger.log_dir / "best_model.pt"
    print(f"Loading best model from {model_path}")
    model.load_state_dict(torch.load(model_path))

    if FLAGS['test_driver_name'] in grouped_data:
        acc, auroc, r1, r2, eval_lbl = adapt_and_evaluate(
            model, grouped_data[FLAGS['test_driver_name']], FLAGS['test_driver_name'], FLAGS
        )

        # Plot reward distribution
        plot_reward_distribution(
            r1, r2, eval_lbl, FLAGS['test_driver_name'],
            save_path=logger.log_dir / f"rewards_{FLAGS['test_driver_name']}.png"
        )

        # Visualize latent space
        visualize_latent_space(
            model, grouped_data,
            FLAGS['train_driver_names'], FLAGS['test_driver_name'],
            FLAGS['device'],
            save_path=logger.log_dir / "latent_space.png",
            downsample=FLAGS['downsample'],
            context_size=FLAGS['context_size']
        )
    else:
        print(f"Test driver {FLAGS['test_driver_name']} not in dataset.")

    print("\nAll Done.")

if __name__ == "__main__":
    main()
