import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_training_curves(history, save_path=None):
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))

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

    axes[2].plot(history['val_acc'])
    axes[2].set_title('Validation Accuracy')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy')
    axes[2].grid(True)

    axes[3].plot(history['val_auroc'])
    axes[3].set_title('Validation AUROC')
    axes[3].set_xlabel('Epoch')
    axes[3].set_ylabel('AUROC')
    axes[3].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_roc_curve(y_true, y_probs, save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_prediction_scatter(y_true, y_probs, threshold=0.5, save_path=None):
    indices = np.arange(len(y_true))

    good_mask = y_true == 1
    bad_mask = y_true == 0

    plt.figure(figsize=(12, 6))
    plt.scatter(indices[bad_mask], y_probs[bad_mask], c='blue', alpha=0.6, s=20, label='Bad (False)')
    plt.scatter(indices[good_mask], y_probs[good_mask], c='red', alpha=0.6, s=20, label='Good (True)')
    plt.axhline(y=threshold, color='black', linestyle='--', linewidth=1, label=f'Threshold ({threshold:.3f})')
    plt.xlabel('Sample Index')
    plt.ylabel('Predicted Probability')
    plt.title('Prediction Scatter Plot')
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

def save_combination_params(model, feature_cols, save_path):
    w_raw, b_raw = model.raw_reward_params()
    beta = model.model.coef_.reshape(-1)
    intercept = model.model.intercept_[0]
    mu = model.scaler.mean_
    sigma = model.scaler.scale_

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111)
    ax.axis('off')

    y_pos = 0.95
    line_height = 0.035

    def add_text(text, fontsize=10, weight='normal', color='black'):
        nonlocal y_pos
        ax.text(0.05, y_pos, text, fontsize=fontsize, weight=weight,
                family='monospace', verticalalignment='top', color=color)
        y_pos -= line_height

    add_text("=== Reward Function (with explicit scaling) ===", fontsize=12, weight='bold')
    add_text("")

    add_text("Step 1: Feature Mapping", fontsize=11, weight='bold', color='darkblue')
    if model.form == "quad_only":
        for i, feat in enumerate(feature_cols):
            add_text(f"  $\\phi_{{{i}}}$ = ({feat})²")
    elif model.form == "linear_quad":
        for i, feat in enumerate(feature_cols):
            add_text(f"  $\\phi_{{{2*i}}}$ = |{feat}|,  $\\phi_{{{2*i+1}}}$ = ({feat})²")
    elif model.form == "quad_exp":
        for i, feat in enumerate(feature_cols):
            add_text(f"  $\\phi_{{{2*i}}}$ = ({feat})²,  $\\phi_{{{2*i+1}}}$ = exp({model.w3[i]:.3f} × max(|{feat}| - {model.w4[i]:.3f}, 0))")

    add_text("")
    add_text("Step 2: Standardization", fontsize=11, weight='bold', color='darkblue')
    for i in range(len(mu)):
        add_text(f"  $\\tilde{{\\phi}}_{{{i}}}$ = ($\\phi_{{{i}}}$ - {mu[i]:.3f}) / {sigma[i]:.3f}")

    add_text("")
    add_text("Step 3: Linear Combination", fontsize=11, weight='bold', color='darkblue')
    add_text(f"  $r(s_t) = \\sum_i \\beta_i \\times \\tilde{{\\phi}}_i$ + {intercept:.3f}")
    for i in range(len(beta)):
        add_text(f"    $\\beta_{{{i}}}$ = {beta[i]:.3f}")

    add_text("")
    add_text("=== Final Equation (raw form) ===", fontsize=11, weight='bold', color='darkgreen')

    terms = []
    if model.form == "quad_only":
        for i, feat in enumerate(feature_cols):
            terms.append(f"{w_raw[i]:.3f} × ({feat})²")
    elif model.form == "linear_quad":
        for i, feat in enumerate(feature_cols):
            terms.append(f"{w_raw[2*i]:.3f} × |{feat}|")
            terms.append(f"{w_raw[2*i+1]:.3f} × ({feat})²")
    elif model.form == "quad_exp":
        for i, feat in enumerate(feature_cols):
            terms.append(f"{w_raw[2*i]:.3f} × ({feat})²")
            terms.append(f"{w_raw[2*i+1]:.3f} × exp({model.w3[i]:.3f} × max(|{feat}| - {model.w4[i]:.3f}, 0))")

    equation_parts = ["$r(s_t)$ = "]
    max_line_length = 80
    current_line = equation_parts[0]

    for i, term in enumerate(terms):
        term_str = term if i == 0 else f" + {term}"
        if len(current_line) + len(term_str) > max_line_length:
            add_text(current_line)
            current_line = "         " + term_str.strip()
        else:
            current_line += term_str

    current_line += f" + {b_raw:.3f}"
    add_text(current_line)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_kfold_roc_curves(fold_results, save_path=None):
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
    plt.title('K-Fold Cross-Validation ROC Curves')
    plt.legend(loc="lower right")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def save_all_plots(model, history, val_loader, paths, is_online=False, feature_cols=None, verbose=0):
    X_val, y_val = val_loader.dataset.tensors
    y_val_np = y_val.numpy()

    y_probs = model.predict_probability(X_val)

    if 'train_loss' in history:
        plot_training_curves(history, save_path=paths.get('training_curves.png', create=True))
    plot_roc_curve(y_val_np, y_probs, save_path=paths.get('roc_curve.png', create=True))
    plot_prediction_scatter(y_val_np, y_probs, threshold=model.best_threshold, save_path=paths.get('prediction_scatter.png', create=True))

    if is_online:
        plot_step_rewards(model, X_val, y_val_np, n_samples=10, save_path=paths.get('step_rewards.png', create=True))

    if hasattr(model, 'raw_reward_params') and feature_cols:
        save_combination_params(model, feature_cols, save_path=paths.get('combination_params.png', create=True))

    if verbose >= 2:
        print(f"  Plots saved to: {paths.run_dir}")
    return y_val_np, y_probs