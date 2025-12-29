import numpy as np

def compute_baselines(X, y, n_conditional_samples=50):
    """
    Compute different baseline strategies for SHAP-based explanations.

    Args:
        X: Input data (n_samples, T, F)
        y: Labels (n_samples,)
        n_conditional_samples: Number of samples for conditional imputation

    Returns:
        dict: Baseline strategies
            - 'zero': All zeros
            - 'mean_all': Mean of all samples
            - 'mean_negative': Mean of negative samples (y==0)
            - 'conditional': Multiple negative samples for conditional expectation
    """
    baselines = {}

    # Zero baseline
    baselines['zero'] = np.zeros_like(X[0])

    # Mean of all samples
    baselines['mean_all'] = X.mean(axis=0)

    # Mean of negative samples only
    X_negative = X[y == 0]
    if len(X_negative) > 0:
        baselines['mean_negative'] = X_negative.mean(axis=0)
    else:
        baselines['mean_negative'] = None

    # Conditional imputation: multiple samples from negative distribution
    if len(X_negative) >= n_conditional_samples:
        np.random.seed(42)
        indices = np.random.choice(len(X_negative), size=n_conditional_samples, replace=False)
        baselines['conditional'] = X_negative[indices]
    elif len(X_negative) > 0:
        baselines['conditional'] = X_negative
    else:
        baselines['conditional'] = None

    return baselines


def compute_feature_importance(attributions):
    """
    Compute global feature importance by averaging over batch and time.

    Args:
        attributions: Attribution array (B, T, F) or (1, T, F)

    Returns:
        np.ndarray: Feature importance (F,)
    """
    abs_attr = np.abs(attributions)
    feature_importance = abs_attr.mean(axis=(0, 1))
    return feature_importance


def compute_temporal_importance(attributions):
    """
    Compute temporal importance by averaging over batch and features.

    Args:
        attributions: Attribution array (B, T, F) or (1, T, F)

    Returns:
        np.ndarray: Temporal importance (T,)
    """
    abs_attr = np.abs(attributions)
    temporal_importance = abs_attr.mean(axis=(0, 2))
    return temporal_importance


def compute_feature_time_matrix(attributions):
    """
    Compute feature × time interaction matrix by averaging over batch.

    Args:
        attributions: Attribution array (B, T, F) or (1, T, F)

    Returns:
        np.ndarray: Feature × time matrix (T, F)
    """
    abs_attr = np.abs(attributions)
    feature_time_matrix = abs_attr.mean(axis=0)
    return feature_time_matrix
