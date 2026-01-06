"""
Visualization utilities for TimeSHAP explanations (project-local).

- NO dependency on timeshap.plot (Altair).
- All plots are matplotlib-based and saveable via plt.savefig.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

def plot_explanation_summary(
    explanation: Dict,
    instance: np.ndarray,
    feature_names: Optional[List[str]] = None,
    figsize: tuple = (16, 12),
    save_path: Optional[str] = None
):
    """
    Create a comprehensive summary plot with all explanation levels.

    Args:
        explanation: Output from TimeSHAPExplainer.explain_instance()
        instance: Original input instance of shape (T, F)
        feature_names: Feature names
        figsize: Figure size
        save_path: Path to save the figure (optional)
    """
    if feature_names is None:
        feature_names = explanation.get('feature_names', None)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(
        f"TimeSHAP Explanation Summary (Prediction: {explanation['prediction']:.4f})",
        fontsize=14,
        fontweight='bold'
    )

    # 1. Event importance (top-left)
    ax = axes[0, 0]
    event_scores = explanation['event_scores']
    timesteps = np.arange(len(event_scores))
    ax.bar(timesteps, event_scores, color='steelblue', alpha=0.7)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Event Importance')
    ax.set_title('Temporal Importance (Which timesteps matter?)')
    ax.grid(axis='y', alpha=0.3)

    # 2. Feature importance (top-right)
    ax = axes[0, 1]
    feature_scores = explanation['feature_scores']
    n_features = len(feature_scores)
    if feature_names is None:
        feature_names = [f'F{i}' for i in range(n_features)]

    # Sort by importance
    sorted_indices = np.argsort(feature_scores)
    sorted_scores = feature_scores[sorted_indices]
    sorted_names = [feature_names[i] for i in sorted_indices]

    ax.barh(sorted_names, sorted_scores, color='coral', alpha=0.7)
    ax.set_xlabel('Feature Importance')
    ax.set_title('Feature Importance (Which features matter?)')
    ax.grid(axis='x', alpha=0.3)

    # 3. Cell-level heatmap (bottom-left)
    ax = axes[1, 0]
    cell_scores = explanation['cell_scores']  # (T, F)
    im = ax.imshow(
        cell_scores.T,  # (F, T) for better visualization
        aspect='auto',
        cmap='RdBu_r',
        interpolation='nearest'
    )
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Feature')
    ax.set_title('Cell-Level Importance (Feature x Time)')
    if feature_names:
        ax.set_yticks(range(n_features))
        ax.set_yticklabels(feature_names)
    plt.colorbar(im, ax=ax, label='Importance')

    # 4. Input data overlay (bottom-right)
    ax = axes[1, 1]
    # Show normalized input data
    instance_normalized = (instance - instance.min(axis=0)) / (instance.max(axis=0) - instance.min(axis=0) + 1e-8)
    im = ax.imshow(
        instance_normalized.T,  # (F, T)
        aspect='auto',
        cmap='viridis',
        interpolation='nearest'
    )
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Feature')
    ax.set_title('Input Data (Normalized)')
    if feature_names:
        ax.set_yticks(range(n_features))
        ax.set_yticklabels(feature_names)
    plt.colorbar(im, ax=ax, label='Value')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    return fig


def plot_top_k_cells(
    cell_scores: np.ndarray,
    instance: np.ndarray,
    feature_names: Optional[List[str]] = None,
    k: int = 10,
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None
):
    """
    Plot top-k most important cells with their values.

    Args:
        cell_scores: Cell importance scores of shape (T, F)
        instance: Original input instance of shape (T, F)
        feature_names: Feature names
        k: Number of top cells to show
        figsize: Figure size
        save_path: Path to save the figure (optional)
    """
    T, F = cell_scores.shape

    if feature_names is None:
        feature_names = [f'F{i}' for i in range(F)]

    # Flatten and get top-k
    flat_scores = cell_scores.flatten()
    flat_indices = np.argsort(np.abs(flat_scores))[::-1][:k]

    # Convert back to (t, f) indices
    top_timesteps = flat_indices // F
    top_features = flat_indices % F

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    labels = [f'{feature_names[f]} @ t={t}' for t, f in zip(top_timesteps, top_features)]
    scores = [cell_scores[t, f] for t, f in zip(top_timesteps, top_features)]
    values = [instance[t, f] for t, f in zip(top_timesteps, top_features)]

    x = np.arange(k)
    width = 0.35

    # Plot scores and values side by side
    ax.barh(x - width/2, scores, width, label='Importance', color='steelblue', alpha=0.7)
    ax.barh(x + width/2, values, width, label='Value', color='coral', alpha=0.7)

    ax.set_yticks(x)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Magnitude')
    ax.set_title(f'Top {k} Most Important Cells')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    return fig

def _as_2d_instance(instance: np.ndarray) -> np.ndarray:
    x = np.asarray(instance)
    if x.ndim == 3:
        x = x.squeeze(0)
    if x.ndim != 2:
        raise ValueError(f"instance must be (T,F) or (1,T,F). got {x.shape}")
    return x


def plot_event_importance(
    event_scores: Union[np.ndarray, List[float]],
    figsize: tuple = (12, 4),
    max_events: Optional[int] = 200,
    save_path: Optional[str] = None,
):
    scores = np.asarray(event_scores).reshape(-1)

    if max_events is not None and scores.shape[0] > max_events:
        scores = scores[-max_events:]
        x = np.arange(scores.shape[0]) - scores.shape[0]
        xlabel = "Timestep (relative, last window)"
    else:
        x = np.arange(scores.shape[0])
        xlabel = "Timestep"

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x, scores, alpha=0.75)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_title("Event / Temporal Importance")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Shapley Value")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig


def plot_feature_importance(
    feature_scores: Union[np.ndarray, List[float]],
    feature_names: Optional[List[str]] = None,
    figsize: tuple = (10, 6),
    top_k: Optional[int] = None,
    save_path: Optional[str] = None,
):
    scores = np.asarray(feature_scores).reshape(-1)
    F = scores.shape[0]
    if feature_names is None:
        feature_names = [f"F{i}" for i in range(F)]

    order = np.argsort(np.abs(scores))[::-1]
    if top_k is not None:
        order = order[: min(top_k, F)]

    names = [feature_names[i] for i in order][::-1]
    vals = scores[order][::-1]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(names, vals, alpha=0.75)
    ax.axvline(0.0, linewidth=1.0)
    ax.set_title("Feature Importance")
    ax.set_xlabel("Shapley Value")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig


def plot_cell_importance(
    cell_scores: np.ndarray,
    feature_names: Optional[List[str]] = None,
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None,
):
    cs = np.asarray(cell_scores)
    if cs.ndim == 3:
        cs = cs.squeeze(0)
    if cs.ndim != 2:
        raise ValueError(f"cell_scores must be (T,F) (or (1,T,F)). got {cs.shape}")

    T, F = cs.shape
    if feature_names is None:
        feature_names = [f"F{i}" for i in range(F)]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cs.T, aspect="auto", cmap="RdBu_r", interpolation="nearest")
    ax.set_title("Cell-Level Importance (Feature x Time)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Feature")
    ax.set_yticks(range(F))
    ax.set_yticklabels(feature_names)
    plt.colorbar(im, ax=ax, label="Shapley Value")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig


def plot_temporal_pruning_hint(
    event_scores: Union[np.ndarray, List[float]],
    pruned_idx: Optional[int],
    figsize: tuple = (12, 4),
    save_path: Optional[str] = None,
):
    """
    'Pruning plot'을 timeshap.plot 없이 대체:
    - event_scores를 그리고
    - pruned_idx(음수면 T+pruned_idx)를 세로선으로 표시
    """
    scores = np.asarray(event_scores).reshape(-1)
    T = scores.shape[0]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(np.arange(T), scores, linewidth=1.2)
    ax.axhline(0.0, linewidth=1.0)

    if pruned_idx is not None:
        cut = pruned_idx
        if cut < 0:
            cut = T + cut
        if 0 <= cut < T:
            ax.axvline(cut, linestyle="--", linewidth=1.2)
            ax.text(cut, ax.get_ylim()[1], f" pruned_idx={pruned_idx}", va="top")

    ax.set_title("Temporal Importance + Pruning Boundary (hint)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Shapley Value")
    ax.grid(alpha=0.25)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig


# ---- 이미 너가 만들어둔 summary / top-k 셀 플롯은 그대로 두고 사용하면 됨 ----
# (pasted.txt에 있는 plot_explanation_summary / plot_top_k_cells 그대로 유지)


def plot_all_timeshap(
    explanation: Dict,
    instance: np.ndarray,
    feature_names: Optional[List[str]] = None,
    save_dir: Optional[str] = None,
):
    """
    기존 코드에서 explanation['event_data'] 등 DataFrame을 넘기던 부분을
    explanation['event_scores'/'feature_scores'/'cell_scores'] 기반으로 통일.
    :contentReference[oaicite:3]{index=3}
    """
    x = _as_2d_instance(instance)

    if feature_names is None:
        feature_names = explanation.get("feature_names", None)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    figures = {}

    # 1) Event
    figures["event"] = plot_event_importance(
        explanation["event_scores"],
        save_path=os.path.join(save_dir, "event_importance.png") if save_dir else None,
    )

    # 2) Feature
    figures["feature"] = plot_feature_importance(
        explanation["feature_scores"],
        feature_names=feature_names,
        save_path=os.path.join(save_dir, "feature_importance.png") if save_dir else None,
    )

    # 3) Cell
    figures["cell"] = plot_cell_importance(
        explanation["cell_scores"],
        feature_names=feature_names,
        save_path=os.path.join(save_dir, "cell_importance.png") if save_dir else None,
    )

    # 4) Pruning hint (대체 플롯)
    figures["temporal_pruning"] = plot_temporal_pruning_hint(
        explanation["event_scores"],
        pruned_idx=explanation.get("pruned_idx", None),
        save_path=os.path.join(save_dir, "temporal_pruning.png") if save_dir else None,
    )

    # 5) Summary (너가 이미 구현해둔 걸 호출)
    # 아래 두 함수(plot_explanation_summary, plot_top_k_cells)는
    # pasted.txt에 있는 구현을 그대로 두고 import/호출만 맞춰주면 됩니다.
    figures["summary"] = plot_explanation_summary(
        explanation,
        x,
        feature_names=feature_names,
        save_path=os.path.join(save_dir, "summary.png") if save_dir else None,
    )

    figures["top_cells"] = plot_top_k_cells(
        explanation["cell_scores"],
        x,
        feature_names=feature_names,
        k=10,
        save_path=os.path.join(save_dir, "top_cells.png") if save_dir else None,
    )

    return figures
