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