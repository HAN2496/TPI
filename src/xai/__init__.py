"""
XAI (Explainable AI) module for time series model interpretation.

This module provides:
- Explainers: IntegratedGradients, KernelShap
- Visualizations: Feature importance, temporal importance, heatmaps
- Utilities: Baseline computation, attribution metrics
"""

from .models import (
    IntegratedGradientsExplainer,
    KernelShapExplainer,
    create_explainer
)

from .utils import (
    compute_baselines,
    compute_feature_importance,
    compute_temporal_importance,
    compute_feature_time_matrix
)

from .visualization import (
    plot_feature_importance,
    plot_temporal_importance,
    plot_feature_time_heatmap,
    plot_top_contributions
)

__all__ = [
    # Explainers
    'IntegratedGradientsExplainer',
    'KernelShapExplainer',
    'create_explainer',

    # Utils
    'compute_baselines',
    'compute_feature_importance',
    'compute_temporal_importance',
    'compute_feature_time_matrix',

    # Visualization
    'plot_feature_importance',
    'plot_temporal_importance',
    'plot_feature_time_heatmap',
    'plot_top_contributions',
]
