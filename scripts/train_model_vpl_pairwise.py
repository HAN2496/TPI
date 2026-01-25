import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.model.vpl.vae import VAEModel
from src.model.vpl.trainer import VPLTrainer, Annealer
from src.model.vpl.utils import PreferenceDataset, create_vpl_dataset, compute_step_rewards, convert_to_pairwise
from src.utils.logger import ExperimentLogger

# --- Configuration ---
FLAGS = {
    # Data
    "features": ["IMU_VerAccelVal", "Bounce_rate_6D", "Pitch_rate_6D", "IMU_LongAccelVal"],
    "test_driver_name": "강신길",
    "train_driver_names": ["김진명", "김태근", "조현석", "한규택", "박재일", "이지환"],
    "time_range": (5, 7),
    "downsample": 5,
    "tie_ratio": 0.0,
    "context_size": 20,
    "val_size": 0.1,
    'normalize': True,

    "num_workers": 0,
    "pin_memory": False,

    # Inference
    "query_ratio": 0.3, # Ratio of data used for querying z
    "timestamp": None, # If not None or "test", load model from this timestamp and skip training

    # Model
    "hidden_dim": 128,
    "batch_size": 32,
    "latent_dim": 4,
    "kl_weight": 0.005,
    "flow_prior": False,
    "use_annealing": True,
    "annealer_baseline": 0.0,
    "annealer_type": "cosine",
    "annealer_cycles": 4,

    # Training
    "early_stop": False,
    "lr": 3e-4,
    "weight_decay": 0.0,
    "n_epochs": 500,
    "eval_freq": 10,
    "warmup_epochs": 10,

    "device": "cuda", # cuda, cpu
}

def collate_fn(batch):
    return {
        'observations': torch.stack([torch.from_numpy(item['observations']).float() for item in batch]),
        'observations_2': torch.stack([torch.from_numpy(item['observations_2']).float() for item in batch]),
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
            obs1 = batch['observations'].to(device)
            obs2 = batch['observations_2'].to(device)
            labels = batch['labels'].to(device)

            # Encoder Output: (B, latent_dim)
            mean, log_var = model.encode(obs1, obs2, labels)
            all_means.append(mean.cpu().numpy())
            all_logvars.append(log_var.cpu().numpy())

    if not all_means:
        return np.zeros(model.latent_dim), np.empty((0, model.latent_dim)), np.empty((0, model.latent_dim))

    all_means = np.concatenate(all_means, axis=0)
    all_logvars = np.concatenate(all_logvars, axis=0)

    # Global representation (Average of means)
    z_global = np.mean(all_means, axis=0)

    return z_global, all_means, all_logvars

def evaluate_driver(model, driver_name, X_raw, y_raw, query_ratio, device, logger):
    """
    Test Driver evaluation process:
    1. Split: Raw Data -> Query Set (for z inference) / Eval Set (for reward evaluation)
    2. Inference: Query Set -> Pairwise -> z inference
    3. Evaluation: Eval Set -> Reward calculation -> AUROC
    """
    print(f"Evaluating rewards for {driver_name}...")
    
    # 1. Split Data
    X_query, X_eval, y_query, y_eval = train_test_split(
        X_raw, y_raw, train_size=query_ratio, random_state=42, stratify=y_raw
    )

    print(f"  Split: Query({len(X_query)}) / Eval({len(X_eval)})")

    # 2. Inference (Get z using Query Set)
    print("  Converting Query Set to Pairwise...")
    query_dict = convert_to_pairwise(
        X_query, y_query, driver_name, 
        context_size=FLAGS["context_size"], 
        tie_ratio=FLAGS["tie_ratio"]
    )

    query_dataset = PreferenceDataset(query_dict)
    z_mean, _, _ = estimate_latent(model, query_dataset, device)
    print(f"  Inferred z: {z_mean.shape}")

    # 3. Compute Rewards (using Eval Set)
    # step_rewards: (N_episodes, T)
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
    """모든 드라이버의 Latent Space(t-SNE) 시각화 (Flow Prior 지원)"""
    print("Visualizing Latent Space...")
    
    latents = []
    labels = []
    
    # 1. Collect Train Drivers Data
    for name in train_dataset.get_driver_list():
            data_dict = train_dataset.get_driver_data(name)
            
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

    # 3. t-SNE & Plot
    latents = np.concatenate(latents, axis=0)
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    
    # Setup t-SNE
    perplexity = min(30, len(latents) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    
    # Determine Plot Layout (1 or 2 subplots)
    use_flow = model.flow_prior
    n_cols = 2 if use_flow else 1
    fig, axs = plt.subplots(1, n_cols, figsize=(10 * n_cols, 8))

    if n_cols == 1:
        axs = [axs] # Make iterable
    
    # --- Plot 1: Original Latent Space ---
    print("  Computing t-SNE for Original Latent Space...")
    z_embedded = tsne.fit_transform(latents)
    
    for label in unique_labels:
        mask = (labels == label)
        axs[0].scatter(z_embedded[mask, 0], z_embedded[mask, 1], label=label, alpha=0.6)
        
    axs[0].set_title("Latent Space (Original)")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    
    # --- Plot 2: Flow Transformed (if applicable) ---
    if use_flow:
        print("  Computing t-SNE for Flow-Transformed Latent Space...")
        # Apply Flow
        z_tensor = torch.from_numpy(latents).float().to(device)
        with torch.no_grad():
            # flow() returns (z_k, log_det_jac)
            z_transformed, _ = model.flow(z_tensor)
            z_transformed = z_transformed.cpu().numpy()
            
        z_trans_embedded = tsne.fit_transform(z_transformed)
        
        for label in unique_labels:
            mask = (labels == label)
            axs[1].scatter(z_trans_embedded[mask, 0], z_trans_embedded[mask, 1], label=label, alpha=0.6)
            
        axs[1].set_title("Latent Space (Flow Transformed)")
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Latent plot saved to {save_path}")

def plot_history(metrics, save_dir, warmup_epochs=0):
    """학습 History 그래프 (모든 Metrics 동적 시각화) - Warmup 이후부터 표시"""
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
            plot_values = values[warmup_epochs:]
            plot_epochs = range(warmup_epochs + 1, warmup_epochs + 1 + len(plot_values))

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
    print(f"Plotting pointwise preference predictions for {driver_name}...")
    model.eval()

    # X: (N, T, F), y: (N,)
    batch_size = 32
    num_samples = len(X)
    all_probs = []

    z_fixed = torch.from_numpy(z_mean).float().to(device) # (D,)

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_X = X[i : i+batch_size]

            # Prepare Obs: (B, 1, T, F) - creating 'annotation' dimension of 1
            obs = torch.from_numpy(batch_X).float().to(device).unsqueeze(1)
            B, _, T, _ = obs.shape

            # Expand z to match obs: (B, 1, T, D)
            z_batch = z_fixed.view(1, 1, 1, -1).expand(B, 1, T, -1)

            # Decode -> Reward
            # VAEModel decode returns reward sequence (B, 1, T)
            r_seq = model.decode(obs, z_batch)
            
            # Sum over time to get Episode Reward
            r_total = r_seq.sum(dim=2) # (B, 1)

            # Scale (if model has scaling factor)
            if hasattr(model, 'scaling'):
                r_total = r_total / model.scaling

            # Sigmoid (Pointwise Probability: P(Good|x))
            probs = torch.sigmoid(r_total) # (B, 1)
            all_probs.append(probs.cpu().numpy().flatten())

    all_probs = np.concatenate(all_probs)
    
    plt.figure(figsize=(10, 6))
    idx = np.arange(num_samples)
    
    # y can be 0 or 1
    mask_0 = (y == 0)
    mask_1 = (y == 1)

    plt.scatter(idx[mask_0], all_probs[mask_0], c='blue', alpha=0.5, label='Label 0 (Bad)', s=15)
    plt.scatter(idx[mask_1], all_probs[mask_1], c='red', alpha=0.5, label='Label 1 (Good)', s=15)

    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.title(f"Pointwise Preference Prediction - {driver_name}\n(P(Good|x, z_global))")
    plt.xlabel("Sample Index")
    plt.ylabel("Predicted Probability")
    plt.legend()
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Scatter plot saved to {save_path}")

def main():
    os.makedirs("artifacts/vpl/pairwise", exist_ok=True)
    if FLAGS["timestamp"] == "test":
        timestamp = "test"
        log_dir = Path("artifacts/vpl/pairwise") / timestamp
        is_training = True
    elif FLAGS["timestamp"] is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = Path("artifacts/vpl/pairwise") / timestamp
        is_training = True
    else:
        timestamp = FLAGS["timestamp"]
        log_dir = Path("artifacts/vpl/pairwise") / timestamp
        is_training = False
        print(f"Inference Mode: Loading model from {log_dir}")
    
    logger = ExperimentLogger(
        log_dir=log_dir,
        experiment_name="vpl_experiment",
        add_timestamp=False 
    )
    
    print(f"Experiment Directory: {logger.log_dir}")
    
    if is_training:
        with open(logger.log_dir / "flags.json", "w", encoding='utf-8') as f:
            json.dump(FLAGS, f, indent=4, ensure_ascii=False)
    
    # 1. Dataset Creation
    print("Creating datasets...")
    train_dict_all, test_driver_raw_data, (data_mean, data_std) = create_vpl_dataset(
        train_driver_names=FLAGS["train_driver_names"],
        test_driver_names=[FLAGS["test_driver_name"]],
        features=FLAGS["features"],
        time_range=FLAGS["time_range"],
        downsample=FLAGS["downsample"],
        tie_ratio=FLAGS["tie_ratio"],
        context_size=FLAGS["context_size"],
        normalize=FLAGS["normalize"]
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
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=FLAGS["num_workers"],
        pin_memory=FLAGS["pin_memory"]
    )

    # 3. Model Initialization
    sample = next(iter(train_loader))
    B, Nmax, T, d = sample['observations'].shape

    pair_dim = 2 * T * d + 1
    encoder_input = Nmax * pair_dim 
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
        learned_prior=FLAGS["flow_prior"],
        annealer=annealer,
        reward_scaling=T
    ).to(FLAGS["device"])

    # 4. Training
    if is_training:
        trainer = VPLTrainer(model, logger, FLAGS)
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
        save_path=logger.log_dir / "latent_space_tsne.png"
    )

    print("\nAll Done.")

if __name__ == "__main__":
    main()