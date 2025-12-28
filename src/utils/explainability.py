import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from captum.attr import IntegratedGradients

class CaptumExplainer:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()
        self.ig = IntegratedGradients(self.model.forward)

    def explain_sample(self, x_sample, baseline=None):
        if not isinstance(x_sample, torch.Tensor):
            x_sample = torch.tensor(x_sample, dtype=torch.float32)

        if x_sample.ndim == 2:
            x_sample = x_sample.unsqueeze(0)

        x_sample = x_sample.to(self.device)

        if baseline is None:
            baseline = torch.zeros_like(x_sample)
        elif not isinstance(baseline, torch.Tensor):
            baseline = torch.tensor(baseline, dtype=torch.float32)
            if baseline.ndim == 2:
                baseline = baseline.unsqueeze(0)
            baseline = baseline.to(self.device)

        with torch.set_grad_enabled(True):
            attributions = self.ig.attribute(
                x_sample,
                baselines=baseline,
                n_steps=50,
                internal_batch_size=1
            )

        attributions = attributions.detach().cpu().numpy()
        return attributions

def create_explainer(model, device='cpu'):
    return CaptumExplainer(model, device)

def compute_feature_importance(attributions):
    abs_attr = np.abs(attributions)
    feature_importance = abs_attr.mean(axis=(0, 1))
    return feature_importance

def compute_temporal_importance(attributions):
    abs_attr = np.abs(attributions)
    temporal_importance = abs_attr.mean(axis=(0, 2))
    return temporal_importance

def compute_feature_time_matrix(attributions):
    abs_attr = np.abs(attributions)
    feature_time_matrix = abs_attr.mean(axis=0)
    return feature_time_matrix

def plot_feature_importance(feature_importance, feature_names, save_path=None):
    plt.figure(figsize=(10, 6))
    colors = ['#d62728' if imp > 0 else '#1f77b4' for imp in feature_importance]
    bars = plt.barh(feature_names, feature_importance, color=colors)
    plt.xlabel('Attribution Score (Feature Importance)', fontsize=12)
    plt.title('Global Feature Importance', fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

def plot_temporal_importance(temporal_importance, save_path=None):
    timesteps = list(range(len(temporal_importance)))

    plt.figure(figsize=(12, 5))
    plt.plot(timesteps, temporal_importance, marker='o', linewidth=2, markersize=6, color='#2ca02c')
    plt.fill_between(timesteps, 0, temporal_importance, alpha=0.3, color='#2ca02c')
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Attribution Score', fontsize=12)
    plt.title('Temporal Importance over Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

def plot_feature_time_heatmap(feature_time_matrix, feature_names, save_path=None):
    plt.figure(figsize=(14, 6))

    vmax = feature_time_matrix.max()
    sns.heatmap(feature_time_matrix.T,
                xticklabels=[f't={t}' for t in range(feature_time_matrix.shape[0])],
                yticklabels=feature_names,
                cmap='YlOrRd',
                vmin=0, vmax=vmax,
                cbar_kws={'label': 'Attribution Score'},
                linewidths=0.5, linecolor='gray')
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title('Feature × Time Attribution Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

def plot_top_contributions(feature_time_matrix, feature_names, top_k=5, save_path=None):
    all_contributions = []

    for t in range(feature_time_matrix.shape[0]):
        for f in range(feature_time_matrix.shape[1]):
            all_contributions.append({
                'time': t,
                'feature': feature_names[f],
                'attribution': feature_time_matrix[t, f]
            })

    all_contributions.sort(key=lambda x: x['attribution'], reverse=True)
    top_contributions = all_contributions[:top_k]

    plt.figure(figsize=(10, 6))
    labels = [f"{c['feature']} (t={c['time']})" for c in top_contributions]
    values = [c['attribution'] for c in top_contributions]
    colors = ['#d62728' if v > 0 else '#1f77b4' for v in values]

    plt.barh(labels, values, color=colors)
    plt.xlabel('Attribution Score', fontsize=12)
    plt.title(f'Top {top_k} Feature×Time Contributions', fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
