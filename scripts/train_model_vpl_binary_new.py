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
from src.model.vpl.utils import PreferenceDataset, DynamicPreferenceDataset, compute_step_rewards, convert_to_binary_context
from src.utils.logger import ExperimentLogger

# --- Configuration ---
FLAGS = {
    # Data
    "features": ["IMU_VerAccelVal", "Bounce_rate_6D", "Pitch_rate_6D", "IMU_LongAccelVal"],
    "test_driver_name": "강신길",
    "train_driver_names": ["김진명", "김태근", "조현석", "한규택", "박재일", "이지환"],
    "time_range": (5, 7),
    "downsample": 1,
    "context_size": 32,
    "val_size": 0.1,
    'normalize': True, # Normalize is crucial for VAE
    "balanced": False,

    "num_workers": 0,
    "pin_memory": False,

    # Inference
    "query_ratio": 0.5, # Ratio of data used for querying z
    "timestamp": None, # If not None, load model from this timestamp and skip training

    # Model
    "hidden_dim": 128,
    "batch_size": 32, # Number of drivers (contexts) per batch
    "latent_dim": 8,
    "kl_weight": 0.1,
    "flow_prior": True,
    "use_annealing": False,
    "annealer_baseline": 0.0,
    "annealer_type": "cosine",
    "annealer_cycles": 4,

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
    # batch is a list of dicts (from __getitem__)
    # each dict has 'observations': (C, T, F), 'labels': (C, 1)
    
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

def prepare_dynamic_data(train_driver_names, test_driver_names, features, time_range, downsample, normalize=True):
    """
    Loads data and returns:
    1. driver_data_map for DynamicPreferenceDataset (Training)
    2. test_driver_data for Evaluation
    3. (mean, std) for normalization
    """
    target_drivers = sorted(list(set(train_driver_names + test_driver_names)))
    config = {'features': features}

    print(f"Loading raw data...")
    raw_data_cache = {}
    all_observations = []

    # 1. Load All Data
    for driver_name in target_drivers:
        from src.utils.utils import _load_dataset_sequences 
        X, y = _load_dataset_sequences(driver_name, time_range, downsample, config)
        
        # Check valid labels
        if len(X[y==1]) == 0 or len(X[y==0]) == 0:
             print(f"Warning: Driver {driver_name} might be missing positive or negative samples.")

        raw_data_cache[driver_name] = (X, y)
        if normalize:
            all_observations.append(X)

    # 2. Compute Normalization Stats
    mean, std = None, None
    if normalize and all_observations:
        print("Computing normalization stats...")
        concat_obs = np.concatenate(all_observations, axis=0)
        mean = np.mean(concat_obs, axis=(0, 1))
        std = np.std(concat_obs, axis=(0, 1)) + 1e-6 
        print(f"  Mean: {mean}\n  Std : {std}")

    # 3. Organize Data
    train_data_map = {}
    test_driver_data = {}

    for driver_name in target_drivers:
        X_raw, y = raw_data_cache[driver_name]
        
        # Apply Normalization
        X = (X_raw - mean) / std if normalize else X_raw

        if driver_name in test_driver_names:
            test_driver_data[driver_name] = (X, y)
            print(f"  {driver_name}: Saved for Testing ({len(X)} episodes)")
        else:
            # For training, just store the array. Dataset will sample dynamically.
            train_data_map[driver_name] = {'observations': X, 'labels': y}
            print(f"  {driver_name}: Saved for Training ({len(X)} episodes)")

    return train_data_map, test_driver_data, (mean, std)

def evaluate_driver(model, driver_name, X_raw, y_raw, query_ratio, device, logger):
    print(f"Evaluating rewards for {driver_name}...")
    
    # 1. Split Data
    X_query, X_eval, y_query, y_eval = train_test_split(
        X_raw, y_raw, train_size=query_ratio, random_state=42, stratify=y_raw
    )

    print(f"  Split: Query({len(X_query)}) / Eval({len(X_eval)})")

    # 2. Inference (Get z using Query Set)
    print("  Converting Query Set to Binary Context...")
    # NOTE: Evaluation still uses static context conversion for consistency
    query_dict = convert_to_binary_context(
        X_query, y_query, driver_name, 
        context_size=FLAGS["context_size"], balanced=FLAGS["balanced"]
    )

    query_dataset = PreferenceDataset(query_dict)
    z_mean, _, _ = estimate_latent(model, query_dataset, device)
    print(f"  Inferred z: {z_mean.shape}")

    # 3. Compute Rewards (using Eval Set)
    step_rewards = compute_step_rewards(model, X_eval, z_mean, device)
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

    # 5. Plot Step Rewards
    from src.utils.visualization import plot_test_step_rewards
    plot_test_step_rewards(
        step_rewards, y_eval, driver_name, n_samples=5,
        save_path=logger.log_dir / f"rewards_{driver_name}.png"
    )

    return mean_rewards, query_dataset, X_query, y_query

def visualize_latent_space(model, train_data_map, test_driver_results, device, save_path):
    print("Visualizing Latent Space...")
    
    latents = []
    labels = []
    
    # 1. Collect Train Drivers Data (Sampled from map)
    # Since we don't have static queries, we generate some for visualization
    context_size = FLAGS["context_size"]
    
    for name, data in train_data_map.items():
        X, y = data['observations'], data['labels']
        
        # Create a temporary static dataset for visualization (e.g., 5 contexts)
        # We use convert_to_binary_context but we should shuffle first to mimic training
        perm = np.random.permutation(len(X))
        X_shuffled = X[perm]
        y_shuffled = y[perm]
        
        # Limit to a few contexts to avoid clutter
        limit = context_size * 5 
        if len(X_shuffled) > limit:
            X_shuffled = X_shuffled[:limit]
            y_shuffled = y_shuffled[:limit]

        temp_dict = convert_to_binary_context(X_shuffled, y_shuffled, name, context_size, balanced=FLAGS["balanced"])
        if temp_dict:
            dset = PreferenceDataset(temp_dict)
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
    n_rows = 2 if use_flow else 1
    n_cols = 2 
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 6 * n_rows))
    if n_rows == 1:
        axs = np.array([axs])

    # Helper
    def plot_scatter(ax, data, title):
        for label in unique_labels:
            mask = (labels == label)
            ax.scatter(data[mask, 0], data[mask, 1], label=label, alpha=0.6)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # --- 1. Original Latent Space ---
    print("  Computing t-SNE & PCA for Original Latent Space...")
    perplexity = min(30, len(latents) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    
    z_tsne = tsne.fit_transform(latents)
    plot_scatter(axs[0, 0], z_tsne, "t-SNE (Original)")
    
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(latents)
    var_ratio = pca.explained_variance_ratio_
    plot_scatter(axs[0, 1], z_pca, f"PCA (Original)\nVar: {var_ratio}")

    # --- 2. Flow Transformed ---
    if use_flow:
        print("  Computing t-SNE & PCA for Flow-Transformed Latent Space...")
        z_tensor = torch.from_numpy(latents).float().to(device)
        with torch.no_grad():
            z_transformed, _ = model.flow(z_tensor)
            z_transformed = z_transformed.cpu().numpy()
            
        z_trans_tsne = tsne.fit_transform(z_transformed)
        plot_scatter(axs[1, 0], z_trans_tsne, "t-SNE (Flow Transformed)")
        
        pca_flow = PCA(n_components=2)
        z_trans_pca = pca_flow.fit_transform(z_transformed)
        var_ratio_flow = pca_flow.explained_variance_ratio_
        plot_scatter(axs[1, 1], z_trans_pca, f"PCA (Flow)\nVar: {var_ratio_flow}")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Latent plot saved to {save_path}")

def plot_history(metrics, save_dir, warmup_epochs=0):
    print(f"Plotting training history...")
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
            plot_values = values[warmup_epochs:] if len(values) > warmup_epochs else values
            plot_epochs = range(len(metrics[key]) - len(plot_values) + 1, len(metrics[key]) + 1)
            
            label = 'Train' if 'train' in key else ('Eval' if 'eval' in key else key)
            plt.plot(plot_epochs, plot_values, label=label, alpha=0.8)
            
        plt.title(name)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(save_dir / 'history_all.png')
    plt.close()

def plot_preference_predictions(model, X, y, z_mean, driver_name, device, save_path):
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
    plt.legend()
    plt.savefig(str(save_path).replace("scatter_", "hist_"))
    plt.close()
    
    # --- Plot 2: Scatter of Probabilities ---
    plt.figure(figsize=(10, 6))
    idx = np.arange(num_samples)
    mask_0 = (y == 0)
    mask_1 = (y == 1)

    plt.scatter(idx[mask_0], all_probs[mask_0], c='blue', alpha=0.5, label='Label 0 (Bad)', s=15)
    plt.scatter(idx[mask_1], all_probs[mask_1], c='red', alpha=0.5, label='Label 1 (Good)', s=15)
    plt.axhline(y=0.5, color='gray', linestyle='--')
    plt.title(f"Pointwise Prediction - {driver_name}")
    plt.ylim(-0.05, 1.05)
    plt.savefig(save_path)
    plt.close()
    print(f"  Plots saved to {save_path}")

def main():
    os.makedirs("artifacts/vpl/binary_dynamic", exist_ok=True)
    if FLAGS["timestamp"] == "test":
        timestamp = "test"
        log_dir = Path("artifacts/vpl/binary_dynamic") / timestamp
        is_training = True
    elif FLAGS["timestamp"] is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = Path("artifacts/vpl/binary_dynamic") / timestamp
        is_training = True
    else:
        timestamp = FLAGS["timestamp"]
        log_dir = Path("artifacts/vpl/binary_dynamic") / timestamp
        is_training = False
        print(f"Inference Mode: Loading model from {log_dir}")
    
    logger = ExperimentLogger(
        log_dir=log_dir,
        experiment_name="vpl_binary_dynamic",
        add_timestamp=False 
    )
    
    print(f"Experiment Directory: {logger.log_dir}")
    
    if is_training:
        with open(logger.log_dir / "flags.json", "w", encoding='utf-8') as f:
            json.dump(FLAGS, f, indent=4, ensure_ascii=False)

    # 1. Dataset Creation (Dynamic)
    print("Creating dynamic datasets...")
    train_data_map, test_driver_data, (data_mean, data_std) = prepare_dynamic_data(
        train_driver_names=FLAGS["train_driver_names"],
        test_driver_names=[FLAGS["test_driver_name"]],
        features=FLAGS["features"],
        time_range=FLAGS["time_range"],
        downsample=FLAGS["downsample"],
        normalize=FLAGS["normalize"]
    )

    # For Validation, we need a static dataset to measure consistent progress
    # We can hold out a portion of train_data_map or just use it all for 'train' 
    # and use Dynamic sampling for 'validation' but fixed seed? 
    # Better approach: Split train_data_map into train/val at driver level? No, data level.
    # For now, let's use DynamicDataset for both, but we rely on random sampling for Val too.
    # Ideally, Val should be fixed. Let's create a fixed validation set from train_data_map.
    
    print("Creating Fixed Validation Set...")
    val_queries = defaultdict(list)
    
    # We will extract 10% of data from train_data_map for validation
    train_data_map_final = {}
    
    for driver, data in train_data_map.items():
        X, y = data['observations'], data['labels']
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=FLAGS["val_size"], stratify=y, random_state=42)
        
        # Update Train Map
        train_data_map_final[driver] = {'observations': X_train, 'labels': y_train}
        
        # Create Val Contexts (Static)
        # Shuffle Val data
        perm = np.random.permutation(len(X_val))
        X_val = X_val[perm]
        y_val = y_val[perm]
        
        val_dict = convert_to_binary_context(X_val, y_val, driver, FLAGS["context_size"], balanced=FLAGS["balanced"])
        if val_dict:
            for key in ['observations', 'labels', 'driver_name']:
                val_queries[key].extend(val_dict[key])
                
    # Build Datasets
    train_dataset = DynamicPreferenceDataset(
        train_data_map_final, 
        context_size=FLAGS["context_size"], 
        mode='binary'
    )
    
    val_dataset_dict = {k: np.stack(v) for k, v in val_queries.items() if v}
    val_dataset = PreferenceDataset(val_dataset_dict)
    
    print(f"Train Dataset: Dynamic Sampling from {len(train_data_map_final)} drivers.")
    print(f"Val Dataset: {len(val_dataset)} static queries.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=FLAGS["batch_size"],
        shuffle=True, # Shuffle drivers
        collate_fn=collate_fn,
        num_workers=FLAGS["num_workers"],
        pin_memory=FLAGS["pin_memory"]
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=FLAGS["batch_size"], # Can use same batch size
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=FLAGS["num_workers"],
        pin_memory=FLAGS["pin_memory"]
    )

    # 3. Model Initialization
    # Sample from Train Loader to check dims
    sample = next(iter(train_loader))
    B, Nmax, T, d = sample['observations'].shape # B=Batch(Drivers), Nmax=Context

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
        reward_scaling=T
    ).to(FLAGS["device"])

    # 4. Training
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
    X_raw, y_raw = test_driver_data[test_name]

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
        model, train_data_map_final, test_results_for_viz, 
        FLAGS["device"], 
        save_path=logger.log_dir / "latent_space_tsne.png"
    )

    print("\nAll Done.")

if __name__ == "__main__":
    main()
