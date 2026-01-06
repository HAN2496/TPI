"""
TimeSHAP explainer for TPI models.

Provides interfaces to explain:
1. Feature importance: Which features are important for the prediction?
2. Temporal importance: Which timesteps are important?
3. Feature-Temporal importance: Which features at which timesteps are important?

Based on TimeSHAP: https://github.com/feedzai/timeshap
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple, Union

try:
    import timeshap.explainer as tsx
except ImportError:
    raise ImportError(
        "timeshap is not installed. Please install it with: pip install timeshap"
    )

from .model_adapters import create_model_adapter, OnlineModelAdapter, OfflineModelAdapter


class TimeSHAPExplainer:
    """
    TimeSHAP-based explainer for TPI models.

    Computes Shapley values to explain:
    - Event-level: Which time windows contribute to the prediction?
    - Feature-level: Which features are most important?
    - Cell-level: Which (feature, timestep) pairs are most important?
    """

    def __init__(
        self,
        model,
        background_data: np.ndarray,
        model_type: str = 'online',
        device: str = 'cpu',
        feature_names: Optional[List[str]] = None
    ):
        """
        Args:
            model: TPI model (OnlineLSTM, OfflineLSTM, etc.)
            background_data: Background dataset of shape (N, T, F)
                Used as reference for computing Shapley values
            model_type: 'online' or 'offline'
            device: Device to run the model on
            feature_names: Names of features (for visualization)
        """
        self.model = model
        self.background_data = background_data
        self.model_type = model_type
        self.device = device

        # Create model adapter for TimeSHAP
        self.model_adapter = create_model_adapter(
            model, model_type=model_type, device=device, return_rewards=False
        )

        # Feature names
        n_features = background_data.shape[2]
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(n_features)]
        self.feature_names = feature_names

        # Create baseline (mean of background data)
        self.baseline = background_data.mean(axis=0)  # (T, F)

    def _create_model_function(self):
        """
        Create model function compatible with TimeSHAP API.

        TimeSHAP expects:
        - Input: (N, T, F) numpy array
        - Output: (N, 1) numpy array
        """
        def f(x: np.ndarray) -> np.ndarray:
            """
            Args:
                x: Input of shape (N, T, F)
            Returns:
                predictions of shape (N, 1)
            """
            preds = self.model_adapter(x)  # (N,) or scalar

            # Ensure output is (N, 1)
            if isinstance(preds, (int, float)):
                preds = np.array([[preds]])
            elif preds.ndim == 1:
                preds = preds.reshape(-1, 1)

            return preds

        return f

    def explain_instance(
        self,
        instance: np.ndarray,
        pruning_dict: Optional[Dict] = None,
        event_dict: Optional[Dict] = None,
        feature_dict: Optional[Dict] = None,
        cell_dict: Optional[Dict] = None,
        verbose: bool = False
    ) -> Dict:
        """
        Explain a single instance using TimeSHAP.

        Args:
            instance: Input instance of shape (T, F)
            pruning_dict: Pruning configuration (default: {'tol': 0.025})
            event_dict: Event explanation config (default: {})
            feature_dict: Feature explanation config (default: {})
            cell_dict: Cell explanation config (default: {})
            verbose: Print progress

        Returns:
            Dictionary with explanations at all levels
        """
        # Ensure instance is (T, F)
        if instance.ndim == 3:
            instance = instance.squeeze(0)

        # Set default configs
        if pruning_dict is None:
            pruning_dict = {'tol': 0.025}
        if event_dict is None:
            event_dict   = {"rs": 42, "nsamples": 2048}
        if feature_dict is None:
            feature_dict = {"rs": 42, "nsamples": 2048}
        if cell_dict is None:
            cell_dict    = {"rs": 42, "nsamples": 2048, "top_x": 5}

        # Create model function
        f = self._create_model_function()

        # Get model prediction
        prediction = f(instance[np.newaxis, ...])[0, 0]

        if verbose:
            print(f"Prediction: {prediction:.4f}")
            print("Computing pruning...")

        instance = np.asarray(instance)
        if instance.ndim == 2:          # (T, F)
            instance = instance[None, :, :]    # -> (1, T, F)
        elif instance.ndim != 3:
            raise ValueError(f"instance shape must be (T,F) or (1,T,F). got {instance.shape}")

        # print("instance:", instance.shape)      # (T_inst, F)
        # print("baseline:", self.baseline.shape) # (T_base, F) 또는 (1, F)

        # Step 1: Pruning
        pruning_data, pruned_idx = tsx.local_pruning(
            f=f,
            data=instance,
            pruning_dict=pruning_dict,
            baseline=self.baseline,
            entity_uuid=0,
            entity_col='entity',
            verbose=verbose
        )

        if verbose:
            print(f"Pruned index: {pruned_idx}")
            print("Computing event-level explanations...")

        # Step 2: Event-level
        event_data = tsx.local_event(
            f=f,
            data=instance,
            event_dict=event_dict,
            entity_uuid=0,
            entity_col='entity',
            baseline=self.baseline,
            pruned_idx=pruned_idx
        )

        if verbose:
            print("Computing feature-level explanations...")

        # Step 3: Feature-level
        feature_data = tsx.local_feat(
            f=f,
            data=instance,
            feature_dict=feature_dict,
            entity_uuid=0,
            entity_col='entity',
            baseline=self.baseline,
            pruned_idx=pruned_idx
        )


        if verbose:
            print("Computing cell-level explanations...")


        # Step 4: Cell-level
        cell_data = tsx.local_cell_level(
            f=f,
            data=instance,
            cell_dict=cell_dict,
            event_data=event_data,
            feat_data=feature_data,
            entity_uuid=0,
            entity_col='entity',
            baseline=self.baseline,
            pruned_idx=pruned_idx
        )



        # Extract scores from DataFrames
        event_scores = self._extract_event_scores(event_data)
        feature_scores = self._extract_feature_scores(feature_data)
        cell_scores = self._extract_cell_scores(cell_data, instance.squeeze(0).shape)

        return {
            'prediction': prediction,
            'pruned_idx': pruned_idx,
            'pruning_data': pruning_data,
            'event_data': event_data,
            'feature_data': feature_data,
            'cell_data': cell_data,
            'event_scores': event_scores,
            'feature_scores': feature_scores,
            'cell_scores': cell_scores,
            'feature_names': self.feature_names,
        }

    def _extract_event_scores(self, event_data: pd.DataFrame) -> np.ndarray:
        """Extract event importance scores from TimeSHAP output."""
        # TimeSHAP returns DataFrame with coalition information
        # Extract the Shapley values for each event
        if 'Shapley Value' in event_data.columns:
            return event_data['Shapley Value'].values
        elif 'Coalition' in event_data.columns:
            # Group by coalition and sum
            grouped = event_data.groupby('Coalition')['Shapley Value'].sum()
            return grouped.values
        else:
            # Fallback: return raw values
            return event_data.values.flatten()

    def _extract_feature_scores(self, feature_data: pd.DataFrame) -> np.ndarray:
        """Extract feature importance scores from TimeSHAP output."""
        if 'Shapley Value' in feature_data.columns:
            # Group by feature and sum across time
            if 'Feature' in feature_data.columns:
                grouped = feature_data.groupby('Feature')['Shapley Value'].sum()
                return grouped.values
            else:
                return feature_data['Shapley Value'].values
        else:
            return feature_data.values.flatten()

    def _extract_cell_scores(self, cell_data: pd.DataFrame, shape: Tuple[int, int]) -> np.ndarray:
        """Extract cell-level scores and reshape to (T, F)."""
        T, F = shape

        if 'Shapley Value' in cell_data.columns:
            values = cell_data['Shapley Value'].values
            # Reshape to (T, F)
            if len(values) == T * F:
                return values.reshape(T, F)
            else:
                # Pad or truncate
                cell_scores = np.zeros((T, F))
                min_len = min(len(values), T * F)
                cell_scores.flat[:min_len] = values[:min_len]
                return cell_scores
        else:
            return np.zeros((T, F))


def create_background_dataset(
    full_data: np.ndarray,
    n_samples: int = 100,
    strategy: str = 'random'
) -> np.ndarray:
    """
    Create background dataset for TimeSHAP from full dataset.

    Args:
        full_data: Full dataset of shape (N, T, F)
        n_samples: Number of background samples
        strategy: Sampling strategy
            - 'random': Random sampling
            - 'kmeans': K-means clustering (representative samples)

    Returns:
        Background dataset of shape (n_samples, T, F)
    """
    n_total = full_data.shape[0]

    if strategy == 'random':
        indices = np.random.choice(n_total, size=min(n_samples, n_total), replace=False)
        return full_data[indices]

    elif strategy == 'kmeans':
        from sklearn.cluster import KMeans

        # Flatten to (N, T*F) for clustering
        n, t, f = full_data.shape
        flat_data = full_data.reshape(n, -1)

        # Cluster
        kmeans = KMeans(n_clusters=min(n_samples, n_total), random_state=42)
        kmeans.fit(flat_data)

        # Get closest samples to centroids
        distances = kmeans.transform(flat_data)
        indices = distances.argmin(axis=0)

        return full_data[indices]

    else:
        raise ValueError(f"Unknown strategy: {strategy}")
