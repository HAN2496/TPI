import numpy as np
import pandas as pd

def to_tsfresh_df(X, feature_names=None):
    """
    Convert (N, T, D) numpy array to tsfresh-compatible long DataFrame.
    
    Args:
        X: (N, T, D) numpy array
        feature_names: list of strings, length D. If None, uses ["0", "1", ...].
        
    Returns:
        df: pandas DataFrame with columns ["id", "time"] + feature_names
    """
    N, T, D = X.shape
    if feature_names is None:
        feature_names = [str(i) for i in range(D)]
    
    # Create ids and time indices
    ids = np.repeat(np.arange(N), T)
    times = np.tile(np.arange(T), N)
    
    # Reshape data: stack time steps for each episode
    # X -> (N*T, D)
    data_flat = X.reshape(N * T, D)
    
    df = pd.DataFrame(data_flat, columns=feature_names)
    df["id"] = ids
    df["time"] = times
    
    return df
