import numpy as np
import pandas as pd


def to_tsfresh_df(X, feature_names=None):
    N, T, D = X.shape
    if feature_names is None:
        feature_names = [str(i) for i in range(D)]
    ids = np.repeat(np.arange(N), T)
    times = np.tile(np.arange(T), N)
    df = pd.DataFrame(X.reshape(N * T, D), columns=feature_names)
    df["id"] = ids
    df["time"] = times
    return df
