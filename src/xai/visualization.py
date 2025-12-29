import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importance(feature_importance, feature_names, save_path=None):
    """
    Plot global feature importance as horizontal bar chart.

    Args:
        feature_importance: Feature importance scores (F,)
        feature_names: List of feature names
        save_path: Optional path to save the figure
    """
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
    """
    Plot temporal importance over time as line plot.

    Args:
        temporal_importance: Temporal importance scores (T,)
        save_path: Optional path to save the figure
    """
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


def plot_feature_time_heatmap(feature_time_matrix, feature_names, x_sample=None, save_path=None):
    """
    Plot feature × time attribution heatmap.

    If x_sample is provided, overlays sensor data on top of attribution background.
    Each feature gets a subplot with sensor values (line) and attribution (background color).

    Args:
        feature_time_matrix: Feature × time attribution matrix (T, F)
        feature_names: List of feature names
        x_sample: Optional sensor data to overlay (T, F)
        save_path: Optional path to save the figure
    """
    if x_sample is not None:
        n_features = len(feature_names)
        fig, axes = plt.subplots(n_features, 1, figsize=(16, 3*n_features), sharex=True)

        if n_features == 1:
            axes = [axes]

        timesteps = np.arange(feature_time_matrix.shape[0])
        vmax = feature_time_matrix.max()

        cmap = plt.get_cmap('YlOrRd')

        for f_idx, (ax, feature_name) in enumerate(zip(axes, feature_names)):
            # Background: Attribution heatmap (colored regions)
            attribution_values = feature_time_matrix[:, f_idx]
            colors = cmap(attribution_values / vmax)

            for t in range(len(timesteps)):
                ax.axvspan(t - 0.5, t + 0.5, facecolor=colors[t], alpha=0.6, zorder=0)

            # Foreground: Sensor data line plot
            ax.plot(timesteps, x_sample[:, f_idx],
                   color='black', linewidth=2.5, marker='o', markersize=4,
                   label=f'{feature_name} (sensor value)', zorder=2)

            ax.set_ylabel(f'{feature_name}\nSensor Value', fontsize=11, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3, zorder=1)

            # Right Y-axis: Attribution value range
            ax2 = ax.twinx()
            ax2.set_ylabel('Attribution', fontsize=9, color='red')
            ax2.tick_params(axis='y', labelcolor='red', labelsize=8)
            ax2.set_ylim(0, vmax)
            ax2.set_yticks([0, vmax/2, vmax])

        axes[-1].set_xlabel('Time Step', fontsize=12)
        fig.suptitle('Sensor Data with Attribution Heatmap Overlay',
                    fontsize=14, fontweight='bold', y=0.995)

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, orientation='vertical',
                           pad=0.02, aspect=30, label='Attribution Score')

        plt.tight_layout()

    else:
        # Original heatmap only (no sensor data overlay)
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
    """
    Plot top-K (feature, time) contributions.

    Args:
        feature_time_matrix: Feature × time attribution matrix (T, F)
        feature_names: List of feature names
        top_k: Number of top contributions to display
        save_path: Optional path to save the figure
    """
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
