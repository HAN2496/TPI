import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
import torch

def plot_training_curves(history, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history['train_loss'])
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)

    axes[1].plot(history['val_acc'])
    axes[1].set_title('Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
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

def plot_prediction_scatter(y_true, y_probs, save_path=None):
    indices = np.arange(len(y_true))

    good_mask = y_true == 1
    bad_mask = y_true == 0

    plt.figure(figsize=(12, 6))
    plt.scatter(indices[bad_mask], y_probs[bad_mask], c='blue', alpha=0.6, s=20, label='Bad (False)')
    plt.scatter(indices[good_mask], y_probs[good_mask], c='red', alpha=0.6, s=20, label='Good (True)')
    plt.axhline(y=0.5, color='black', linestyle='--', linewidth=1, label='Threshold (0.5)')
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
    model.eval()
    with torch.no_grad():
        step_rewards = model.get_step_rewards(X_val[:n_samples])

    step_rewards = step_rewards.cpu().numpy()
    y_val_np = y_val[:n_samples].cpu().numpy()

    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 3 * n_samples))
    if n_samples == 1:
        axes = [axes]

    for i in range(n_samples):
        color = 'red' if y_val_np[i] == 1 else 'blue'
        label = 'Good' if y_val_np[i] == 1 else 'Bad'

        axes[i].plot(step_rewards[i], color=color, linewidth=2)
        axes[i].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[i].set_title(f'Sample {i} - Label: {label} (Total: {step_rewards[i].sum():.2f})')
        axes[i].set_xlabel('Step')
        axes[i].set_ylabel('Reward')
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def save_all_plots(model, history, X_val, y_val, paths, is_online=False, is_combination=False):
    if is_combination:
        y_probs = model.model.predict_proba(X_val)[:, 1]
        y_val_np = y_val
    else:
        model.eval()
        with torch.no_grad():
            logits = model(X_val)
            probs = torch.sigmoid(logits).cpu().numpy()
        y_probs = probs
        y_val_np = y_val.cpu().numpy()

    if not is_combination:
        plot_training_curves(history, save_path=paths.get('training_curves.png', create=True))
    plot_roc_curve(y_val_np, y_probs, save_path=paths.get('roc_curve.png', create=True))
    plot_prediction_scatter(y_val_np, y_probs, save_path=paths.get('prediction_scatter.png', create=True))

    if is_online and not is_combination:
        plot_step_rewards(model, X_val, y_val, n_samples=10, save_path=paths.get('step_rewards.png', create=True))
