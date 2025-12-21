from src.utils.data_loader import DatasetManager
import matplotlib.pyplot as plt

n_samples = 1
downsamples = [1, 5, 10]

fig, axes = plt.subplots(4, 1, figsize=(10, 12))

import numpy as np
for j, ds in enumerate(downsamples):
    manager = DatasetManager("datasets", downsample=ds)
    dataset = manager.get("강신길")
    t, X, y = dataset.to_sequences(time_range=(5, 7), pad=False)
    t = np.array(t)
    print(t.shape, X[0].shape, y.shape)

    if n_samples is not None:
        t = t[:n_samples]
        X = X[:n_samples]

    for i, col in enumerate(len(X[0].shape)):
        for t_sample, x_sample in zip(t, X):
            axes[i].plot(t_sample, x_sample[:, i], label=f"ds={ds}" if t_sample is t[0] else None)
        axes[i].set_title(col)
        axes[i].set_xlabel("Time (s)")
        axes[i].set_ylabel("Value")
        axes[i].legend()

plt.tight_layout()
plt.show()
