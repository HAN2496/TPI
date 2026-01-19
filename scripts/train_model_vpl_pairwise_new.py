import os
import json
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.model.vpl_new.vae_new import VAEModel
from src.model.vpl_new.trainer import VPLTrainer, Annealer
from src.model.vpl_new.utils import PreferenceDataset, create_vpl_dataset, compute_step_rewards, convert_to_pairwise
from src.utils.logger import ExperimentLogger
from src.utils.utils import _load_dataset_sequences

# --- Configuration ---
FLAGS = {
    # Data
    "features": ["IMU_VerAccelVal", "Bounce_rate_6D", "Pitch_rate_6D", "IMU_LongAccelVal"],
    "test_driver_name": "강신길",
    "train_driver_names": ["박재일", "이지환", "조현석", "한규택"],
    "time_range": (5, 7),
    "downsample": 5,
    "tie_ratio": 0.0,
    "context_size": 32,
    "val_size": 0.2,
    
    # Inference (New)
    "query_ratio": 0.7, # Test Driver Data 중 z 추론에 사용할 비율 (나머지는 Eval용)

    # Model
    "hidden_dim": 128,
    "batch_size": 32,
    "latent_dim": 32,
    "kl_weight": 1.0,
    "flow_prior": False,
    "use_annealing": True,
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
    return {
        'observations': torch.stack([torch.from_numpy(item['observations']).float() for item in batch]),
        'observations_2': torch.stack([torch.from_numpy(item['observations_2']).float() for item in batch]),
        'labels': torch.stack([torch.from_numpy(item['labels']).float() for item in batch]),
        'driver_name': [item.get('driver_name') for item in batch] # 문자열은 텐서 변환 제외
    }

def estimate_latent(model, dataset, device):
    """데이터셋의 모든 쿼리에 대해 z를 추론하고 평균과 분포를 반환"""
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
    Test Driver 평가 프로세스:
    1. Split: Raw Data -> Query Set (z 추론용) / Eval Set (보상 검증용)
    2. Inference: Query Set -> Pairwise 변환 -> z 추론
    3. Evaluation: Eval Set -> Reward 계산 -> AUROC
    """
    print(f"Evaluating rewards for {driver_name}...")
    
    # 1. Split Data
    # Stratified Split을 사용하여 Class 비율 유지
    try:
        X_query, X_eval, y_query, y_eval = train_test_split(
            X_raw, y_raw, train_size=query_ratio, random_state=42, stratify=y_raw
        )
    except ValueError:
        # 데이터가 너무 적거나 클래스 하나만 있는 경우
        print("  Warning: Stratified split failed (not enough data?). Using random split.")
        X_query, X_eval, y_query, y_eval = train_test_split(
            X_raw, y_raw, train_size=query_ratio, random_state=42
        )

    print(f"  Split: Query({len(X_query)}) / Eval({len(X_eval)})")

    # 2. Inference (Get z using Query Set)
    print("  Converting Query Set to Pairwise...")
    query_dict = convert_to_pairwise(
        X_query, y_query, driver_name, 
        context_size=FLAGS["context_size"], 
        tie_ratio=FLAGS["tie_ratio"]
    )
    
    if query_dict is None:
        print("  Error: Failed to generate pairs from Query Set. Cannot infer z.")
        return None, None

    query_dataset = PreferenceDataset(query_dict)
    z_mean, _, _ = estimate_latent(model, query_dataset, device)
    print(f"  Inferred z: {z_mean.shape}")

    # 3. Compute Rewards (using Eval Set)
    # step_rewards: (N_episodes, T)
    step_rewards = compute_step_rewards(model, X_eval, z_mean, device)
    
    # Mean reward per episode (scaled)
    mean_rewards = step_rewards.mean(axis=1)
    
    # 4. AUROC & Statistics
    if len(np.unique(y_eval)) > 1:
        auroc = roc_auc_score(y_eval, mean_rewards)
        print(f"  AUROC (Eval Set): {auroc:.4f}")
        
        # Plot ROC
        from src.utils.visualization import plot_roc_curve
        plot_roc_curve(
            y_eval, mean_rewards, 
            save_path=logger.log_dir / f"roc_{driver_name}.png",
            title=f"ROC Curve - {driver_name} (Eval Set)"
        )
    else:
        print("  Warning: Single class data in Eval Set, skipping AUROC.")
        
    # 5. Plot Step Rewards (Trajectory)
    from src.utils.visualization import plot_test_step_rewards
    plot_test_step_rewards(
        step_rewards, y_eval, driver_name, n_samples=5,
        save_path=logger.log_dir / f"rewards_{driver_name}.png"
    )

    # 시각화를 위해 Query Dataset도 반환
    return mean_rewards, query_dataset


def visualize_latent_space(model, train_dataset, test_driver_results, device, save_path):
    """모든 드라이버의 Latent Space(t-SNE) 시각화"""
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
            
    # 2. Collect Test Drivers Data (from Inference Result)
    # test_driver_results: {name: (z_global, query_dataset)} 형태가 아니라
    # 여기서는 evaluate_driver 내부에서 생성한 query_dataset을 사용해야 함.
    # 하지만 구조상 main에서 전달받기 복잡하므로, 
    # 간단히 test_driver_results에 {name: query_dataset}을 담아 넘긴다고 가정.
    
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
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latents)-1))
    z_embedded = tsne.fit_transform(latents)
    
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        mask = (labels == label)
        plt.scatter(z_embedded[mask, 0], z_embedded[mask, 1], label=label, alpha=0.6)
        
    plt.title("Latent Space Visualization (t-SNE)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Latent plot saved to {save_path}")

def plot_history(metrics, save_dir):
    """학습 History 그래프 (모든 Metrics 동적 시각화)"""
    print("Plotting training history...")
    
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
            epochs = range(1, len(values) + 1)
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
            plt.plot(epochs, values, label=label, linestyle=style, color=color, alpha=0.8)
            
        plt.title(f'{name.replace("_", " ").capitalize()}')
        plt.xlabel('Epochs')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(save_dir / 'history_all.png')
    plt.close()

def plot_preference_predictions(model, dataset, z_mean, driver_name, device, save_path):
    """Query Set에 대한 예측 확률 Scatter Plot"""
    print(f"Plotting preference predictions for {driver_name}...")
    model.eval()
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    all_probs = []
    all_labels = []
    z_fixed = torch.from_numpy(z_mean).float().to(device)
    
    with torch.no_grad():
        for batch in loader:
            obs1 = batch['observations'].to(device)
            obs2 = batch['observations_2'].to(device)
            labels = batch['labels'].to(device)
            B, N, T, F = obs1.shape
            z_batch = z_fixed.view(1, 1, 1, -1).expand(B, N, T, -1)
            
            r1 = model.decode(obs1, z_batch)
            r2 = model.decode(obs2, z_batch)
            
            r_hat1 = r1.sum(dim=2) / model.scaling
            r_hat2 = r2.sum(dim=2) / model.scaling
            
            probs = torch.sigmoid(r_hat1 - r_hat2)
            all_probs.append(probs.cpu().numpy().flatten())
            all_labels.append(labels.cpu().numpy().flatten())
            
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    
    plt.figure(figsize=(10, 6))
    idx = np.arange(len(all_probs))
    mask_0 = (all_labels < 0.5)
    mask_1 = (all_labels >= 0.5)
    
    plt.scatter(idx[mask_0], all_probs[mask_0], c='blue', alpha=0.5, label='Label 0 (Prefer Obs2)', s=15)
    plt.scatter(idx[mask_1], all_probs[mask_1], c='red', alpha=0.5, label='Label 1 (Prefer Obs1)', s=15)
    
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.title(f"Preference Prediction Scatter - {driver_name}\n(Using global z from Query Set)")
    plt.xlabel("Query Index")
    plt.ylabel("Predicted Probability (P(Obs1 > Obs2))")
    plt.legend()
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Scatter plot saved to {save_path}")

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path("artifacts/vpl_pairwise_new") / timestamp
    
    logger = ExperimentLogger(
        log_dir=log_dir,
        experiment_name="vpl_experiment",
        add_timestamp=False 
    )
    
    print(f"Experiment Directory: {logger.log_dir}")
    
    with open(logger.log_dir / "flags.json", "w", encoding='utf-8') as f:
        json.dump(FLAGS, f, indent=4, ensure_ascii=False)
    
    # 1. Dataset Creation
    print("Creating datasets...")
    # create_vpl_dataset이 이제 (mean, std)도 반환하고, test_driver_data는 Raw Data (X,y)임
    train_dict_all, test_driver_raw_data, (data_mean, data_std) = create_vpl_dataset(
        train_driver_names=FLAGS["train_driver_names"],
        test_driver_names=[FLAGS["test_driver_name"]],
        features=FLAGS["features"],
        time_range=FLAGS["time_range"],
        downsample=FLAGS["downsample"],
        tie_ratio=FLAGS["tie_ratio"],
        context_size=FLAGS["context_size"]
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
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=FLAGS["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
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
        annealer=annealer,
        reward_scaling=T
    ).to(FLAGS["device"])

    # 4. Training
    trainer = VPLTrainer(model, logger, FLAGS)
    print("Starting training...")
    
    metrics, _ = trainer.train(train_loader, val_loader) 
    plot_history(metrics, logger.log_dir)
    
    model_path = logger.log_dir / "best_model.pt"
    if os.path.exists(model_path):
        print(f"Loading best model from {model_path}")
        model.load_state_dict(torch.load(model_path))
    else:
        print("Warning: No best model found. Using last model state.")

    # 5. Evaluation & Visualization
    print("\n" + "="*50 + "\nStarting Evaluation\n" + "="*50)
    
    test_results_for_viz = {}

    # (A) Test Driver Analysis
    test_name = FLAGS["test_driver_name"]
    if test_name in test_driver_raw_data:
        X_raw, y_raw = test_driver_raw_data[test_name]
        
        # evaluate_driver에서 Split -> Infer -> Eval 다 수행
        mean_rewards, query_dataset = evaluate_driver(
            model, test_name, X_raw, y_raw, 
            FLAGS["query_ratio"], FLAGS["device"], logger
        )

        if query_dataset is not None:
            # Scatter Plot은 Query Set에 대해 수행 (모델이 얼마나 z를 잘 맞췄는지 + 선호도 예측 일관성)
            # Eval Set은 Pair 형태가 아니므로(Reward Scalar), Scatter를 그릴 수 없음 (Step Reward Plot으로 대체됨)
            z_mean, _, _ = estimate_latent(model, query_dataset, FLAGS["device"])
            plot_preference_predictions(
                model, query_dataset, z_mean, test_name, 
                FLAGS["device"], 
                save_path=logger.log_dir / f"scatter_pred_{test_name}.png"
            )
            
            test_results_for_viz[test_name] = query_dataset
    
    # (B) All Drivers Latent Visualization
    visualize_latent_space(
        model, train_dataset, test_results_for_viz, 
        FLAGS["device"], 
        save_path=logger.log_dir / "latent_space_tsne.png"
    )

    print("\nAll Done.")

if __name__ == "__main__":
    main()