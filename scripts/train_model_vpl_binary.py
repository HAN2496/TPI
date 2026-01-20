import os
import json
import torch
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.model.vpl.vae_binary import VAEModel
from src.model.vpl.trainer import VPLTrainerBinary, Annealer
from src.model.vpl.utils import PreferenceDataset, compute_step_rewards, convert_to_binary_context
from src.utils.logger import ExperimentLogger

# --- Configuration ---
FLAGS = {
    # Data
    "features": ["IMU_VerAccelVal", "Bounce_rate_6D", "Pitch_rate_6D", "IMU_LongAccelVal"],
    "test_driver_name": "강신길",
    "train_driver_names": ["김진명", "김태근", "조현석", "한규택", "박재일", "이지환"],
    "time_range": (5, 7),
    "downsample": 5,
    "context_size": 8,
    "val_size": 0.1,
    'normalize': False, # False, True
    "balanced": True,

    "num_workers": 0,
    "pin_memory": False,

    # Inference
    "query_ratio": 0.3, # Ratio of data used for querying z
    "timestamp": "test", # If not None, load model from this timestamp and skip training

    # Model
    "hidden_dim": 64,
    "batch_size": 128,
    "latent_dim": 8,
    "kl_weight": 20.0,
    "flow_prior": False,
    "use_annealing": True,
    "annealer_baseline": 0.0,
    "annealer_type": "cosine",
    "annealer_cycles": 4,
    "reward_scaling": "T", # 1 or T

    # Training
    "early_stop": False,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "n_epochs": 500,
    "eval_freq": 10,
    "warmup_epochs": 10,

    "device": "cuda", # cuda, cpu
}

def collate_fn(batch):
    return {
        'observations': torch.stack([torch.from_numpy(item['observations']).float() for item in batch]),
        'labels': torch.stack([torch.from_numpy(item['labels']).float() for item in batch]),
        'driver_name': [item.get('driver_name') for item in batch]
    }

def estimate_latent(model, dataset, device):
    model.eval()
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    all_means = []
    all_logvars = []
    
    with torch.no_grad():
        for batch in loader:
            obs = batch['observations'].to(device)
            labels = batch['labels'].to(device)

            # Encoder Output: (B, latent_dim)
            # Binary model encode takes 2 arguments
            mean, log_var = model.encode(obs, labels)
            all_means.append(mean.cpu().numpy())
            all_logvars.append(log_var.cpu().numpy())

    if not all_means:
        return np.zeros(model.latent_dim), np.empty((0, model.latent_dim)), np.empty((0, model.latent_dim))

    all_means = np.concatenate(all_means, axis=0)
    all_logvars = np.concatenate(all_logvars, axis=0)

    # Global representation (Average of means)
    z_global = np.mean(all_means, axis=0)

    return z_global, all_means, all_logvars

def create_binary_dataset(train_driver_names, test_driver_names, features, time_range, downsample,
                          context_size=64, normalize=False, balanced=False):
    target_drivers = sorted(list(set(train_driver_names + test_driver_names)))
    config = {'features': features}

    # Step 1: Load data and Normalization
    print(f"Loading raw data to compute normalization stats...")
    raw_data_cache = {}
    all_observations = []

    for driver_name in target_drivers:
        from src.utils.utils import _load_dataset_sequences 
        
        X, y = _load_dataset_sequences(driver_name, time_range, downsample, config)
        
        # Check valid labels
        if len(X[y==1]) == 0 or len(X[y==0]) == 0:
             print(f"Warning: Driver {driver_name} might be missing positive or negative samples.")

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

    # Step 2: Create Binary Context Dataset
    train_queries = defaultdict(list)
    test_driver_data = {}

    print(f"Creating binary context datasets with context_size={context_size}...")

    for driver_name in target_drivers:
        X_raw, y = raw_data_cache[driver_name]

        X = (X_raw - mean) / std if normalize else X_raw

        if driver_name in test_driver_names:
            test_driver_data[driver_name] = (X, y)
            print(f"  {driver_name}: Saved as raw data ({len(X)} episodes)")
            continue

        # Shuffle data to ensure mixed labels in each context batch
        perm = np.random.permutation(len(X))
        X = X[perm]
        y = y[perm]

        # Use the new binary converter
        dataset_dict = convert_to_binary_context(X, y, driver_name, context_size, balanced=balanced)

        if dataset_dict is None:
            continue

        num_queries = len(dataset_dict['observations'])
        print(f"  {driver_name}: {num_queries} context batches generated.")

        for key in ['observations', 'labels', 'driver_name']:
            train_queries[key].extend(dataset_dict[key])

    train_dataset_dict = {k: np.stack(v) for k, v in train_queries.items() if v}
        
    return train_dataset_dict, test_driver_data, (mean, std)

def evaluate_driver(model, driver_name, X_raw, y_raw, query_ratio, device, logger):
    print(f"Evaluating rewards for {driver_name}...")
    
    # 1. Split Data
    X_query, X_eval, y_query, y_eval = train_test_split(
        X_raw, y_raw, train_size=query_ratio, random_state=42, stratify=y_raw
    )

    print(f"  Split: Query({len(X_query)}) / Eval({len(X_eval)})")

    # 2. Inference (Get z using Query Set)
    print("  Converting Query Set to Binary Context...")
    query_dict = convert_to_binary_context(
        X_query, y_query, driver_name, 
        context_size=FLAGS["context_size"], balanced=FLAGS["balanced"]
    )

    query_dataset = PreferenceDataset(query_dict)
    z_mean, _, _ = estimate_latent(model, query_dataset, device)
    print(f"  Inferred z: {z_mean.shape}")

    # 3. Compute Rewards (using Eval Set)
    # Binary model decode also works same way for single observation batch
    step_rewards = compute_step_rewards(model, X_eval, z_mean, device)

    # Mean reward per episode (scaled)
    mean_rewards = step_rewards.mean(axis=1)

    # 4. AUROC & Statistics
    auroc = roc_auc_score(y_eval, mean_rewards)
    print(f"  AUROC (Eval Set): {auroc:.4f}")

    # Plot ROC
    from src.utils.visualization import plot_roc_curve
    plot_roc_curve(
        y_eval, mean_rewards, 
        save_path=logger.log_dir / f"roc_{driver_name}.png",
        title=f"ROC Curve - {driver_name} (Eval Set)"
    )

    # 5. Plot Step Rewards (Trajectory)
    from src.utils.visualization import plot_test_step_rewards
    plot_test_step_rewards(
        step_rewards, y_eval, driver_name, n_samples=5,
        save_path=logger.log_dir / f"rewards_{driver_name}.png"
    )

    return mean_rewards, query_dataset, X_query, y_query
def visualize_latent_space(model, train_dataset, test_driver_results, device, save_path):
    print("Visualizing Latent Space...")
    
    latents = []
    labels = []
    
    # 1. Collect Train Drivers Data
    if hasattr(train_dataset, 'get_driver_list'):
        for name in train_dataset.get_driver_list():
            data_dict = train_dataset.get_driver_data(name)
            if data_dict is None: continue
            
            dset = PreferenceDataset(data_dict)
            _, query_means, _ = estimate_latent(model, dset, device)
            
            latents.append(query_means)
            labels.extend([f"{name} (Train)"] * len(query_means))
            
    # 2. Collect Test Drivers Data
    for name, dataset in test_driver_results.items():
        if dataset is None: continue
        _, query_means, _ = estimate_latent(model, dataset, device)
        
        latents.append(query_means)
        labels.extend([f"{name} (Test)"] * len(query_means))
        
    if not latents:
        print("No latents to visualize.")
        return

    # 3. Visualization Setup
    latents = np.concatenate(latents, axis=0)
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    
    use_flow = getattr(model, 'flow_prior', False)
    perplexities = [10, 20, 30]
    
    n_rows = 2 if use_flow else 1
    # 3 t-SNEs + 1 PCA = 4 Columns
    n_cols = len(perplexities) + 1 
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    
    # Ensure axs is 2D array [row, col]
    if n_rows == 1:
        axs = np.expand_dims(axs, axis=0) # (1, 4)
    if n_cols == 1:
        axs = np.expand_dims(axs, axis=1) # Should not happen here

    # --- Helper Function for Scatter Plot ---
    def plot_scatter(ax, data, title):
        for label in unique_labels:
            mask = (labels == label)
            
            # Distinct style for Test data
            if "(Test)" in label:
                marker = '*'
                s = 100
                alpha = 1.0
                edgecolors = 'black'
                zorder = 10 # Draw on top
            else:
                marker = 'o'
                s = 30
                alpha = 0.7
                edgecolors = 'none'
                zorder = 1
                
            ax.scatter(
                data[mask, 0], data[mask, 1], 
                label=label, 
                marker=marker,
                s=s,
                alpha=alpha, 
                edgecolors=edgecolors,
                zorder=zorder
            )
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    # --- 1. Original Latent Space (Row 0) ---
    print("  Computing t-SNE & PCA for Original Latent Space...")
    
    # A. t-SNEs
    for i, perp in enumerate(perplexities):
        actual_perp = min(perp, len(latents) - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=actual_perp)
        z_tsne = tsne.fit_transform(latents)
        plot_scatter(axs[0, i], z_tsne, f"t-SNE (Original)\nPerp={actual_perp}")
    
    # B. PCA
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(latents)
    var_ratio = pca.explained_variance_ratio_
    plot_scatter(axs[0, 3], z_pca, f"PCA (Original)\nVar: [{var_ratio[0]:.2f}, {var_ratio[1]:.2f}]")

    # --- 2. Flow Transformed Latent Space (Row 1) ---
    if use_flow:
        print("  Computing t-SNE & PCA for Flow-Transformed Latent Space...")
        z_tensor = torch.from_numpy(latents).float().to(device)
        with torch.no_grad():
            z_transformed, _ = model.flow(z_tensor)
            z_transformed = z_transformed.cpu().numpy()
            
        # A. t-SNEs
        for i, perp in enumerate(perplexities):
            actual_perp = min(perp, len(latents) - 1)
            tsne = TSNE(n_components=2, random_state=42, perplexity=actual_perp)
            z_trans_tsne = tsne.fit_transform(z_transformed)
            plot_scatter(axs[1, i], z_trans_tsne, f"t-SNE (Flow)\nPerp={actual_perp}")
            
        # B. PCA
        pca_flow = PCA(n_components=2)
        z_trans_pca = pca_flow.fit_transform(z_transformed)
        var_ratio_flow = pca_flow.explained_variance_ratio_
        plot_scatter(axs[1, 3], z_trans_pca, f"PCA (Flow)\nVar: [{var_ratio_flow[0]:.2f}, {var_ratio_flow[1]:.2f}]")

    # Add legend to the last plot only
    axs[0, -1].legend(loc='upper right', bbox_to_anchor=(1.3, 1))

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Latent plot saved to {save_path}")

    # --- Statistics Printout ---
    print("std per dim:", latents.std(axis=0))
    corr = np.corrcoef(latents.T)
    print("corr:\n", np.round(corr, 3))

    # Check Full PCA (up to 4 dims or latent_dim)
    pca_full = PCA(n_components=min(4, latents.shape[1])).fit(latents)
    print("Full PCA explained variance ratio:", np.round(pca_full.explained_variance_ratio_, 4))

    # --- 1. Effective Rank ---
    sv = pca_full.singular_values_
    p = sv / np.sum(sv)
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p))
    effective_rank = np.exp(entropy)
    print(f"Effective Rank: {effective_rank:.4f} (Max possible: {min(4, latents.shape[1])})")

    # --- 2. Silhouette Score ---
    from sklearn.metrics import silhouette_score
    driver_labels = [label.split(' (')[0] for label in labels]
    if len(set(driver_labels)) > 1:
        score = silhouette_score(latents, driver_labels)
        print(f"Driver Silhouette Score: {score:.4f} (-1 to 1, higher is better separation)")
    else:
        print("Driver Silhouette Score: N/A (Only 1 driver found)")

def plot_history(metrics, save_dir, warmup_epochs=0):
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

            style = '-'
            if 'train' in key:
                color = 'tab:blue'
                label = 'Train'
            elif 'eval' in key:
                color = 'tab:orange'
                label = 'Eval'
            else:
                color = None
                label = key
            
            plt.plot(plot_epochs, plot_values, label=label, linestyle=style, color=color, alpha=0.8)
            
        plt.title(f'{name.replace("_", " ").capitalize()}')
        plt.xlabel('Epochs')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(save_dir / 'history_all.png')
    plt.close()

def plot_preference_predictions(model, X, y, z_mean, driver_name, device, save_path):
    """
    Enhanced Plotting:
    1. Histogram of Rewards (Label 0 vs Label 1)
    2. Scatter Plot of Probability (P(Good))
    """
    print(f"Plotting pointwise preference predictions for {driver_name}...")
    model.eval()

    batch_size = 32
    num_samples = len(X)
    all_rewards = []
    all_probs = []

    z_fixed = torch.from_numpy(z_mean).float().to(device)

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_X = X[i : i+batch_size]
            obs = torch.from_numpy(batch_X).float().to(device).unsqueeze(1)
            B, _, T, _ = obs.shape
            z_batch = z_fixed.view(1, 1, 1, -1).expand(B, 1, T, -1)

            r_seq = model.decode(obs, z_batch)
            r_total = r_seq.sum(dim=2) # (B, 1)

            if hasattr(model, 'scaling'):
                r_total = r_total / model.scaling

            probs = torch.sigmoid(r_total) # (B, 1)
            
            all_rewards.append(r_total.cpu().numpy().flatten())
            all_probs.append(probs.cpu().numpy().flatten())

    all_rewards = np.concatenate(all_rewards)
    all_probs = np.concatenate(all_probs)
    
    # --- Plot 1: Histogram of Rewards ---
    plt.figure(figsize=(10, 6))
    plt.hist(all_rewards[y==0], bins=30, alpha=0.5, label='Label 0 (Bad)', color='blue', density=True)
    plt.hist(all_rewards[y==1], bins=30, alpha=0.5, label='Label 1 (Good)', color='red', density=True)
    plt.title(f"Reward Distribution - {driver_name}\n(Raw Rewards)")
    plt.xlabel("Reward Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(str(save_path).replace("scatter_", "hist_"))
    plt.close()
    
    # --- Plot 2: Scatter of Probabilities ---
    plt.figure(figsize=(10, 6))
    idx = np.arange(num_samples)
    mask_0 = (y == 0)
    mask_1 = (y == 1)

    plt.scatter(idx[mask_0], all_probs[mask_0], c='blue', alpha=0.5, label='Label 0 (Bad)', s=15)
    plt.scatter(idx[mask_1], all_probs[mask_1], c='red', alpha=0.5, label='Label 1 (Good)', s=15)

    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.title(f"Pointwise Preference Prediction - {driver_name}")
    plt.xlabel("Sample Index")
    plt.ylabel("Predicted Probability")
    plt.legend()
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Plots saved to {save_path} and hist_ version")

def main():
    os.makedirs("artifacts/vpl/binary", exist_ok=True)
    if FLAGS["timestamp"] == "test":
        timestamp = "test"
        log_dir = Path("artifacts/vpl/binary") / timestamp
        is_training = True
    elif FLAGS["timestamp"] is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = Path("artifacts/vpl/binary") / timestamp
        is_training = True
    else:
        timestamp = FLAGS["timestamp"]
        log_dir = Path("artifacts/vpl/binary") / timestamp
        is_training = False
        print(f"Inference Mode: Loading model from {log_dir}")
    
    logger = ExperimentLogger(
        log_dir=log_dir,
        experiment_name="vpl_binary_experiment",
        add_timestamp=False 
    )
    
    print(f"Experiment Directory: {logger.log_dir}")
    
    if is_training:
        with open(logger.log_dir / "flags.json", "w", encoding='utf-8') as f:
            json.dump(FLAGS, f, indent=4, ensure_ascii=False)

    # 1. Dataset Creation (Binary)
    print("Creating datasets...")
    train_dict_all, test_driver_raw_data, (data_mean, data_std) = create_binary_dataset(
        train_driver_names=FLAGS["train_driver_names"],
        test_driver_names=[FLAGS["test_driver_name"]],
        features=FLAGS["features"],
        time_range=FLAGS["time_range"],
        downsample=FLAGS["downsample"],
        context_size=FLAGS["context_size"],
        normalize=FLAGS["normalize"],
        balanced=FLAGS["balanced"]
    )

    # Train / Validation Split (Query Level)
    n_total_queries = len(train_dict_all['observations'])
    n_val = int(n_total_queries * FLAGS["val_size"])
    n_train = n_total_queries - n_val
    
    permuted_indices = np.random.permutation(n_total_queries)
    train_indices = permuted_indices[:n_train]
    val_indices = permuted_indices[n_train:]
    
    print(f"\nSplitting Dataset (Total {n_total_queries} queries):")
    print(f"  Train: {n_train} queries")
    print(f"  Val  : {n_val} queries")

    def subset_dict(data_dict, indices):
        return {k: v[indices] for k, v in data_dict.items()}

    train_subset = subset_dict(train_dict_all, train_indices)
    val_subset = subset_dict(train_dict_all, val_indices)

    train_dataset = PreferenceDataset(train_subset)
    val_dataset = PreferenceDataset(val_subset)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=FLAGS["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=FLAGS["num_workers"],
        pin_memory=FLAGS["pin_memory"]
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=FLAGS["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=FLAGS["num_workers"],
        pin_memory=FLAGS["pin_memory"]
    )

    # 3. Model Initialization
    sample = next(iter(train_loader))
    B, Nmax, T, d = sample['observations'].shape
    print(f"Observation Shape: Batch {B}, Nmax {Nmax}, Time {T}, Dim {d}")

    # Binary Encoder Input: (T*d + 1)
    # Context flattened: Nmax * (T*d + 1)
    single_dim = T * d + 1
    encoder_input = Nmax * single_dim 
    decoder_input = d + FLAGS["latent_dim"]
    
    print(f"\nModel Input Dim: {encoder_input} (Flattened Context)")

    annealer = None
    if FLAGS["use_annealing"]:
        annealer = Annealer(
            total_steps=FLAGS["n_epochs"] // FLAGS["annealer_cycles"],
            shape=FLAGS["annealer_type"],
            cyclical=FLAGS["annealer_cycles"] > 1
        )
    reward_scaling = 1 if FLAGS["reward_scaling"] == 1 else T
    model = VAEModel(
        encoder_input=encoder_input,
        decoder_input=decoder_input,
        latent_dim=FLAGS["latent_dim"],
        hidden_dim=FLAGS["hidden_dim"],
        annotation_size=Nmax,
        size_segment=T,
        kl_weight=FLAGS["kl_weight"],
        flow_prior=FLAGS["flow_prior"],
        annealer=annealer,
        reward_scaling=reward_scaling
    ).to(FLAGS["device"])

    # 4. Training (Use VPLTrainerBinary)
    if is_training:
        trainer = VPLTrainerBinary(model, logger, FLAGS)
        print("Starting training...")

        metrics, _ = trainer.train(train_loader, val_loader) 
        plot_history(metrics, logger.log_dir, warmup_epochs=FLAGS["warmup_epochs"])
    else:
        print("Skipping training (Inference Mode)...")

    model_path = logger.log_dir / "best_model.pt"
    print(f"Loading best model from {model_path}")
    model.load_state_dict(torch.load(model_path))

    # 5. Evaluation & Visualization
    print("\n" + "="*50 + "\nStarting Evaluation\n" + "="*50)
    test_results_for_viz = {}

    # 5-1. Test Driver Analysis
    test_name = FLAGS["test_driver_name"]
    X_raw, y_raw = test_driver_raw_data[test_name]

    mean_rewards, query_dataset, X_query, y_query = evaluate_driver(
        model, test_name, X_raw, y_raw, 
        FLAGS["query_ratio"], FLAGS["device"], logger
    )

    z_mean, _, _ = estimate_latent(model, query_dataset, FLAGS["device"])

    plot_preference_predictions(
        model, X_query, y_query, z_mean, test_name, 
        FLAGS["device"], 
        save_path=logger.log_dir / f"scatter_pred_{test_name}.png"
    )

    test_results_for_viz[test_name] = query_dataset

    # 5-2. All Drivers Latent Visualization
    visualize_latent_space(
        model, train_dataset, test_results_for_viz, 
        FLAGS["device"], 
        save_path=logger.log_dir / f"latent_space_tsne.png"
    )

    print("\nAll Done.")

if __name__ == "__main__":
    main()
