"""
XAI (Explainable AI) module for TPI.

Provides TimeSHAP-based explanations for TPI models to understand:
- Which features are important for predictions
- Which timesteps are important
- Which (feature, timestep) combinations are important
"""

from .model_adapters import (
    OnlineModelAdapter,
    OfflineModelAdapter,
    create_model_adapter,
    RewardEstimator,
)

from .timeshap_explainer import (
    TimeSHAPExplainer,
    create_background_dataset,
)

from .visualization import (
    plot_explanation_summary,
    plot_top_k_cells,
    plot_all_timeshap,
)

__all__ = [
    # Model adapters
    'OnlineModelAdapter',
    'OfflineModelAdapter',
    'create_model_adapter',
    'RewardEstimator',
    # Explainer
    'TimeSHAPExplainer',
    'create_background_dataset',
    # Visualization
    'plot_explanation_summary',
    'plot_top_k_cells',
    'plot_all_timeshap',
]
