import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.titlesize'] = 16  # 14
plt.rcParams['axes.labelsize'] = 14  # 12
plt.rcParams['legend.fontsize'] = 13 # 11
plt.rcParams['xtick.labelsize'] = 12 # 10
plt.rcParams['ytick.labelsize'] = 12 # 10

def plot_training_curves(history, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].plot(history['train_loss'])
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)

    axes[1].plot(history['val_loss'])
    axes[1].set_title('Validation Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True)

    axes[2].plot(history['val_auroc'])
    axes[2].set_title('Validation AUROC')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('AUROC')
    axes[2].grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_roc_curve(y_true, y_probs, save_path=None, title='ROC Curve'):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_prediction_scatter(y_true, y_probs, threshold=0.5, save_path=None, title='Prediction Scatter Plot'):
    indices = np.arange(len(y_true))

    good_mask = y_true == 1
    bad_mask = y_true == 0

    plt.figure(figsize=(12, 6))
    plt.scatter(indices[bad_mask], y_probs[bad_mask], c='blue', alpha=0.6, s=20, label='Bad (False)')
    plt.scatter(indices[good_mask], y_probs[good_mask], c='red', alpha=0.6, s=20, label='Good (True)')
    plt.axhline(y=threshold, color='black', linestyle='--', linewidth=1, label=f'Threshold ({threshold:.3f})')
    plt.xlabel('Sample Index')
    plt.ylabel('Predicted Probability')
    plt.title(title)
    plt.legend()
    plt.ylim([-0.05, 1.05])
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_step_rewards(model, X_val, y_val, n_samples=5, save_path=None):
    X_sub = X_val[:n_samples]
    y_sub = y_val[:n_samples]

    step_rewards = model.step_rewards(X_sub, detach=True)
    y_np = y_sub.reshape(-1)

    n_cols = 2
    n_rows = math.ceil(n_samples / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    axes = axes.flatten()

    for i in range(n_samples):
        label = "Good" if y_np[i] == 1 else "Bad"
        color = 'red' if y_np[i] == 1 else 'blue'

        axes[i].plot(step_rewards[i], color=color, linewidth=2)
        axes[i].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[i].set_title(f'Sample {i} - Label: {label} (Total: {step_rewards[i].sum():.2f})')
        axes[i].set_xlabel('Step')
        axes[i].set_ylabel('Reward')
        axes[i].grid(True, alpha=0.3)

    for j in range(n_samples, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def save_regression_params(model, feature_cols, save_path):
    beta = model.model.coef_.reshape(-1)
    intercept = model.model.intercept_[0]
    mu = model.scaler.mean_
    sigma = model.scaler.scale_

    is_online = hasattr(model, 'reduce')

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    ax.axis('off')

    y_pos = 0.95
    line_height = 0.03

    def add_text(text, fontsize=10, weight='normal', color='black'):
        nonlocal y_pos
        ax.text(0.05, y_pos, text, fontsize=fontsize, weight=weight,
                family='monospace', verticalalignment='top', color=color)
        y_pos -= line_height

    model_type = "Online (Step-wise)" if is_online else "Offline (Episode-wise)"
    add_text(f"=== Regression Model Parameters ({model_type}) ===", fontsize=12, weight='bold')
    add_text("")

    add_text("Step 1: Feature Mapping / Aggregation", fontsize=11, weight='bold', color='darkblue')
    phi_idx = 0
    feature_phi_map = []

    for feat_name, ops_list in model.basis.items():
        if is_online:
            for op in ops_list:
                if op == 'abs':
                    add_text(f"  φ_{{{phi_idx}}} = |{feat_name}_t|")
                elif op == 'quad':
                    add_text(f"  φ_{{{phi_idx}}} = ({feat_name}_t)²")
                elif op == 'cube':
                    add_text(f"  φ_{{{phi_idx}}} = ({feat_name}_t)³")
                elif op == 'shifted_exp':
                    w3_val = model.w3[feat_name]
                    w4_val = model.w4[feat_name]
                    add_text(f"  φ_{{{phi_idx}}} = exp({w3_val:.3f} × max(|{feat_name}_t| - {w4_val:.3f}, 0))")
                feature_phi_map.append((feat_name, op, phi_idx))
                phi_idx += 1
        else:
            for op in ops_list:
                if op == 'mean':
                    add_text(f"  φ_{{{phi_idx}}} = mean({feat_name})")
                elif op == 'std':
                    add_text(f"  φ_{{{phi_idx}}} = std({feat_name})")
                elif op == 'max':
                    add_text(f"  φ_{{{phi_idx}}} = max({feat_name})")
                elif op == 'min':
                    add_text(f"  φ_{{{phi_idx}}} = min({feat_name})")
                elif op == 'abs_mean':
                    add_text(f"  φ_{{{phi_idx}}} = mean(|{feat_name}|)")
                elif op == 'sqrt_mean':
                    add_text(f"  φ_{{{phi_idx}}} = mean({feat_name}²)")
                elif op == 'rmse':
                    add_text(f"  φ_{{{phi_idx}}} = sqrt(mean({feat_name}²))")
                feature_phi_map.append((feat_name, op, phi_idx))
                phi_idx += 1

    add_text("")
    add_text("Step 2: Standardization", fontsize=11, weight='bold', color='darkblue')
    for i in range(min(len(mu), 10)):
        add_text(f"  φ̃_{{{i}}} = (φ_{{{i}}} - {mu[i]:.3f}) / {sigma[i]:.3f}")
    if len(mu) > 10:
        add_text(f"  ... ({len(mu) - 10} more features)")

    add_text("")
    add_text("Step 3: Logistic Regression", fontsize=11, weight='bold', color='darkblue')
    if is_online:
        reduce_str = f" (reduce={model.reduce})"
        add_text(f"  r_t = Σ β_i × φ̃_i(x_t) + b_adj{reduce_str}")
    else:
        add_text(f"  logit = Σ β_i × φ̃_i + {intercept:.3f}")

    add_text("")
    add_text("Coefficients (β):", fontsize=10, weight='bold', color='darkgreen')
    for i in range(min(len(beta), 15)):
        feat_info = feature_phi_map[i] if i < len(feature_phi_map) else ("?", "?", i)
        add_text(f"  β_{{{i}}} = {beta[i]:+.4f}  [{feat_info[0]}.{feat_info[1]}]")
    if len(beta) > 15:
        add_text(f"  ... ({len(beta) - 15} more coefficients)")

    add_text("")
    add_text(f"Intercept: {intercept:.4f}", fontsize=10, weight='bold', color='darkgreen')

    if is_online:
        add_text("")
        add_text(f"Reduce mode: {model.reduce}", fontsize=10, weight='bold', color='purple')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_kfold_roc_curves(fold_results, save_path=None, title='K-Fold Cross-Validation ROC Curves'):
    plt.figure(figsize=(10, 8))
    plt.rcParams.update({
        'font.size': 16,          # 기본 글자 크기
        'axes.titlesize': 18,     # 제목
        'axes.labelsize': 16,     # x, y축 라벨
        'xtick.labelsize': 16,    # x축 눈금
        'ytick.labelsize': 16,    # y축 눈금
        'legend.fontsize': 13    # 범례
    })

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for fold_idx, (y_true, y_probs) in enumerate(fold_results):
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'Fold {fold_idx} (AUC = {roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='navy', label='Random', alpha=0.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    plt.plot(mean_fpr, mean_tpr, color='darkorange', lw=2, label = f"Mean ROC (AUC = {mean_auc:.4f}")

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2,
                     label='± 1 std. dev.')

    y_true_all = np.concatenate([y_true for y_true, _ in fold_results])
    y_probs_all = np.concatenate([y_probs for _, y_probs in fold_results])
    fpr_oof, tpr_oof, _ = roc_curve(y_true_all, y_probs_all)
    oof_auc = auc(fpr_oof, tpr_oof)

    plt.plot(fpr_oof, tpr_oof, color='red', lw=2.5, linestyle='-',
             label=f'OOF ROC (AUC = {oof_auc:.4f})', alpha=0.9)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# VPL-specific visualization functions

def plot_vpl_training_curves(history, save_path=None):
    """Plot VPL training curves: loss and accuracy"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curves
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy curve
    axes[1].plot(history['val_accuracy'], label='Val Accuracy', color='green', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_driver_latent_space(driver_latents, save_path=None):
    """Visualize driver latents using PCA or t-SNE"""
    from sklearn.decomposition import PCA

    driver_names = list(driver_latents.keys())
    latents = np.array([driver_latents[name]['z_mean_point'] for name in driver_names])

    n_drivers = len(driver_names)
    latent_dim = latents.shape[1]

    # Handle single driver case
    if n_drivers == 1:
        plt.figure(figsize=(8, 6))
        # Show first 2 dimensions of latent
        if latent_dim >= 2:
            plt.scatter([latents[0, 0]], [latents[0, 1]], s=200, alpha=0.7, color='steelblue')
            plt.annotate(driver_names[0], (latents[0, 0], latents[0, 1]),
                        xytext=(5, 5), textcoords='offset points', fontsize=14, fontweight='bold')
            plt.xlabel('Latent Dimension 1')
            plt.ylabel('Latent Dimension 2')
        else:
            plt.scatter([latents[0, 0]], [0], s=200, alpha=0.7, color='steelblue')
            plt.annotate(driver_names[0], (latents[0, 0], 0),
                        xytext=(5, 5), textcoords='offset points', fontsize=14, fontweight='bold')
            plt.xlabel('Latent Dimension 1')
            plt.ylabel('(Single Driver)')
        plt.title(f'Driver Latent Space (1 driver, latent_dim={latent_dim})')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        return

    # Multiple drivers: use PCA if needed
    if latent_dim > 2:
        pca = PCA(n_components=2)
        latents_2d = pca.fit_transform(latents)
        explained_var = pca.explained_variance_ratio_
    else:
        latents_2d = latents
        explained_var = None

    plt.figure(figsize=(8, 6))
    plt.scatter(latents_2d[:, 0], latents_2d[:, 1], s=100, alpha=0.7, color='steelblue')

    for i, name in enumerate(driver_names):
        plt.annotate(name, (latents_2d[i, 0], latents_2d[i, 1]),
                    xytext=(5, 5), textcoords='offset points', fontsize=12)

    if explained_var is not None:
        plt.xlabel(f'PC1 ({explained_var[0]*100:.1f}%)')
        plt.ylabel(f'PC2 ({explained_var[1]*100:.1f}%)')
        plt.title('Driver Latent Space (PCA)')
    else:
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.title('Driver Latent Space')

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_driver_reward_comparison(episode, reward_dict, save_path=None):
    """Plot reward curves for same episode with different driver latents"""
    plt.figure(figsize=(10, 6))

    for driver_name, rewards in reward_dict.items():
        plt.plot(rewards, label=driver_name, linewidth=2, alpha=0.8)

    plt.xlabel('Time Step')
    plt.ylabel('Reward')
    plt.title('Reward Comparison Across Drivers for Same Episode')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_test_step_rewards(step_rewards, y_true, driver_name, n_samples=10, save_path=None):
    true_rewards = step_rewards[y_true == 1]
    false_rewards = step_rewards[y_true == 0]

    n_true = min(n_samples, len(true_rewards))
    n_false = min(n_samples, len(false_rewards))
    total_plots = n_true + n_false

    if total_plots == 0:
        return

    n_cols = 2
    n_rows = math.ceil(total_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3.5 * n_rows), constrained_layout=True)
    axes_flat = axes.flatten()

    for i in range(n_true):
        ax = axes_flat[i]
        ax.plot(true_rewards[i], color='red', alpha=0.8)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        ax.set_title(f'True Sample #{i+1}', fontsize=10, color='darkred', fontweight='bold')
        ax.grid(True, alpha=0.3)

    for i in range(n_false):
        ax = axes_flat[n_true + i]
        ax.plot(false_rewards[i], color='blue', alpha=0.8)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        ax.set_title(f'False Sample #{i+1}', fontsize=10, color='darkblue', fontweight='bold')
        ax.grid(True, alpha=0.3)

    for j in range(total_plots, len(axes_flat)):
        axes_flat[j].axis('off')

    fig.suptitle(f'{driver_name} - Step Rewards per Episode', fontsize=16)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_test_step_rewards(step_rewards, y_true, driver_name, n_samples=10, save_path=None):
    true_mask = (y_true == 1)
    false_mask = (y_true == 0)

    true_rewards = step_rewards[true_mask]
    false_rewards = step_rewards[false_mask]

    n_true = min(n_samples, len(true_rewards))
    n_false = min(n_samples, len(false_rewards))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    for i in range(n_true):
        axes[0].plot(true_rewards[i], alpha=0.6, color='red')
    axes[0].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[0].set_title(f'{driver_name} - True Episodes (n={len(true_rewards)})')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Reward')
    axes[0].grid(True, alpha=0.3)

    for i in range(n_false):
        axes[1].plot(false_rewards[i], alpha=0.6, color='blue')
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[1].set_title(f'{driver_name} - False Episodes (n={len(false_rewards)})')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Reward')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_vpl_test_evaluation(test_results, save_dir):
    from pathlib import Path
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for driver_name, results in test_results.items():
        y_true = results['y_true']
        y_probs = results['y_probs']

        plot_roc_curve(
            y_true, y_probs,
            save_path=save_dir / f'test_{driver_name}_roc_curve.png',
            title=f'Test Driver ROC Curve - {driver_name}'
        )

        plot_prediction_scatter(
            y_true, y_probs,
            threshold=0.5,
            save_path=save_dir / f'test_{driver_name}_scatter.png',
            title=f'Test Driver Prediction Scatter - {driver_name}'
        )

    if len(test_results) > 1:
        plt.figure(figsize=(10, 8))

        for driver_name, results in test_results.items():
            y_true = results['y_true']
            y_probs = results['y_probs']
            fpr, tpr, _ = roc_curve(y_true, y_probs)
            auroc = results['auroc']
            plt.plot(fpr, tpr, lw=2, label=f'{driver_name} (AUC = {auroc:.4f})')

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Test Drivers ROC Curves')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'test_combined_roc_curves.png', dpi=150, bbox_inches='tight')
        plt.close()


def visualize_all_driver_latents(model, all_driver_data, device, save_path, context_size=128):
    """
    모든 드라이버의 latent z를 TSNE/PCA로 시각화

    Args:
        model: VAEModel
        all_driver_data: {driver_name: {'observations': ..., 'observations_2': ..., 'labels': ...}}
        device: cuda/cpu
        save_path: 저장 경로
        context_size: query당 페어 개수 (annotation size)
    """
    import torch
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    all_z = []
    driver_labels = []
    driver_names = []

    print("\nExtracting latent vectors for all drivers...")

    for driver_name, data in all_driver_data.items():
        n_pairs = len(data['observations'])
        n_queries = max(1, n_pairs // context_size)

        driver_z_list = []

        for i in range(n_queries):
            start_idx = i * context_size
            end_idx = min((i + 1) * context_size, n_pairs)

            obs1_chunk = data['observations'][start_idx:end_idx]
            obs2_chunk = data['observations_2'][start_idx:end_idx]
            labels_chunk = data['labels'][start_idx:end_idx]

            obs1 = torch.from_numpy(obs1_chunk).float().to(device).unsqueeze(0)
            obs2 = torch.from_numpy(obs2_chunk).float().to(device).unsqueeze(0)
            labels = torch.from_numpy(labels_chunk).float().to(device).unsqueeze(0)

            with torch.no_grad():
                mean, _ = model.encode(obs1, obs2, labels)

            z = mean.squeeze(0).cpu().numpy()
            driver_z_list.append(z)

        driver_z = np.stack(driver_z_list, axis=0)
        all_z.append(driver_z)
        driver_labels.extend([driver_name] * len(driver_z))
        driver_names.append(driver_name)
        print(f"  {driver_name}: {n_queries} latent samples (from {n_pairs} pairs)")

    all_z = np.concatenate(all_z, axis=0)
    print(f"\nTotal latent samples: {len(all_z)}")
    print(f"Latent dimension: {all_z.shape[1]}")

    # 각 드라이버의 평균 latent 계산
    driver_mean_latents = []
    for driver_name in driver_names:
        mask = np.array([label == driver_name for label in driver_labels])
        driver_mean_z = all_z[mask].mean(axis=0)
        driver_mean_latents.append(driver_mean_z)

    driver_mean_latents = np.array(driver_mean_latents)
    print(f"Driver mean latents shape: {driver_mean_latents.shape}")

    fig, axes = plt.subplots(1, 3, figsize=(24, 6))

    # TSNE - 모든 샘플
    print("\nComputing TSNE for all samples...")
    z_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(all_z)

    from matplotlib.patches import Ellipse, Circle
    import matplotlib.transforms as transforms

    colors = plt.cm.tab10(np.linspace(0, 1, len(driver_names)))

    # 각 드라이버의 TSNE 공간 분산 저장
    driver_tsne_stats = {}

    for i, driver_name in enumerate(driver_names):
        mask = np.array([label == driver_name for label in driver_labels])
        points = z_tsne[mask]

        # 샘플 점들
        axes[0].scatter(points[:, 0], points[:, 1],
                       label=driver_name, alpha=0.6, s=20, color=colors[i])

        # 분산 계산
        std_x = points[:, 0].std()
        std_y = points[:, 1].std()
        overall_std = np.sqrt(std_x**2 + std_y**2)

        # 저장 (두 번째 subplot에서 사용)
        driver_tsne_stats[driver_name] = {
            'std_x': std_x,
            'std_y': std_y,
            'overall_std': overall_std,
            'cov': np.cov(points[:, 0], points[:, 1])
        }

    axes[0].set_title("TSNE: All Latent Samples", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("TSNE Component 1")
    axes[0].set_ylabel("TSNE Component 2")
    axes[0].legend(loc='best', fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # TSNE - 드라이버 평균만
    print("Computing TSNE for driver means...")
    z_mean_tsne = TSNE(n_components=2, perplexity=min(5, len(driver_names)-1), random_state=42).fit_transform(driver_mean_latents)

    def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', edgecolor='red', **kwargs):
        """2D 신뢰 타원 그리기"""
        if len(x) < 2:
            return None

        cov = np.cov(x, y)
        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]) if cov[0, 0] * cov[1, 1] > 0 else 0

        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                         facecolor=facecolor, edgecolor=edgecolor, **kwargs)

        scale_x = np.sqrt(cov[0, 0]) * n_std
        scale_y = np.sqrt(cov[1, 1]) * n_std

        mean_x = np.mean(x)
        mean_y = np.mean(y)

        transf = transforms.Affine2D() \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)

        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)

    for i, driver_name in enumerate(driver_names):
        # 첫 번째 subplot의 분산 정보 가져오기
        stats = driver_tsne_stats[driver_name]
        overall_std = stats['overall_std']

        # 분산 원 그리기 (별 뒤에)
        circle = Circle((z_mean_tsne[i, 0], z_mean_tsne[i, 1]),
                       radius=overall_std,
                       facecolor=colors[i], alpha=0.15,
                       edgecolor=colors[i], linewidth=2, linestyle='--')
        axes[1].add_patch(circle)

        # 평균 latent (별)
        axes[1].scatter(z_mean_tsne[i, 0], z_mean_tsne[i, 1],
                       label=driver_name, alpha=1.0, s=200, color=colors[i],
                       marker='*', edgecolors='black', linewidths=2, zorder=10)

        # 드라이버 이름
        axes[1].annotate(driver_name,
                        (z_mean_tsne[i, 0], z_mean_tsne[i, 1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold')

        # 분산 텍스트
        axes[1].annotate(f'σ={overall_std:.2f}',
                        (z_mean_tsne[i, 0], z_mean_tsne[i, 1]),
                        xytext=(8, -12), textcoords='offset points',
                        fontsize=8, color=colors[i], fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                 edgecolor=colors[i], alpha=0.8))

    axes[1].set_title("TSNE: Driver Means + Variance", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("TSNE Component 1")
    axes[1].set_ylabel("TSNE Component 2")
    axes[1].legend(loc='best', fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # PCA - 모든 샘플
    print("Computing PCA for all samples...")
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(all_z)
    explained_var = pca.explained_variance_ratio_

    for i, driver_name in enumerate(driver_names):
        mask = np.array([label == driver_name for label in driver_labels])
        axes[2].scatter(z_pca[mask, 0], z_pca[mask, 1],
                       label=driver_name, alpha=0.6, s=20, color=colors[i])

    axes[2].set_title("PCA: All Latent Samples", fontsize=14, fontweight='bold')
    axes[2].set_xlabel(f"PC1 ({explained_var[0]*100:.1f}%)")
    axes[2].set_ylabel(f"PC2 ({explained_var[1]*100:.1f}%)")
    axes[2].legend(loc='best', fontsize=10)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Latent space visualization saved to: {save_path}")


def visualize_episode_probabilities(test_results, save_path):
    """
    각 드라이버의 episode별 예측 확률 scatter plot

    Args:
        test_results: {driver_name: {'mean_rewards': array, 'y_true': array, 'auroc': float}}
        save_path: 저장 경로
    """
    n_drivers = len(test_results)

    if n_drivers == 0:
        return

    # subplot 레이아웃 계산
    n_cols = min(2, n_drivers)
    n_rows = math.ceil(n_drivers / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12 * n_cols, 6 * n_rows), squeeze=False)
    axes = axes.flatten()

    for idx, (driver_name, results) in enumerate(test_results.items()):
        ax = axes[idx]

        mean_rewards = results['mean_rewards']
        y_true = results['y_true']
        auroc = results.get('auroc', 0.0)

        # Rewards를 표준화 후 확률로 변환
        # 모델은 상대적 비교로 학습되므로 절대값이 calibration 안 됨
        mean_rewards_normalized = (mean_rewards - mean_rewards.mean()) / (mean_rewards.std() + 1e-8)
        y_probs = 1 / (1 + np.exp(-mean_rewards_normalized))

        # True/False 분리
        true_mask = (y_true == 1)
        false_mask = (y_true == 0)

        indices = np.arange(len(y_true))

        # Scatter plot
        ax.scatter(indices[false_mask], y_probs[false_mask],
                  c='blue', alpha=0.6, s=30, label='Bad (False)', edgecolors='darkblue', linewidths=0.5)
        ax.scatter(indices[true_mask], y_probs[true_mask],
                  c='red', alpha=0.6, s=30, label='Good (True)', edgecolors='darkred', linewidths=0.5)

        # Threshold line
        ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1.5, label='Threshold (0.5)', alpha=0.7)

        # 통계 정보
        true_probs = y_probs[true_mask]
        false_probs = y_probs[false_mask]

        if len(true_probs) > 0 and len(false_probs) > 0:
            true_mean = true_probs.mean()
            false_mean = false_probs.mean()
            separation = true_mean - false_mean

            # 텍스트 박스에 통계 표시
            stats_text = f'AUROC: {auroc:.3f}\n'
            stats_text += f'Good mean: {true_mean:.3f}\n'
            stats_text += f'Bad mean: {false_mean:.3f}\n'
            stats_text += f'Separation: {separation:.3f}'

            ax.text(0.02, 0.98, stats_text,
                   transform=ax.transAxes,
                   fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_xlabel('Episode Index', fontsize=12)
        ax.set_ylabel('Predicted Probability (Good)', fontsize=12)
        ax.set_title(f'{driver_name} - Episode Probabilities', fontsize=14, fontweight='bold')
        ax.set_ylim([-0.05, 1.05])
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)

    # 빈 subplot 제거
    for j in range(len(test_results), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Episode probability scatter saved to: {save_path}")
