import numpy as np
import matplotlib.pyplot as plt
from src.utils.data_loader import DatasetManager

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

DRIVER = "강신길"
TIME_RANGE = (5, 7)
DEFAULT_FEATURES = ['Pitch_rate_6D', 'IMU_VerAccelVal', 'Bounce_rate_6D', 'IMU_LongAccelVal',
                    'Roll_rate_6D', 'IMU_YawRtVal', 'IMU_LatAccelVal', 'IMU_RollRtVal']


def get_feature_names(driver):
    dataset = DatasetManager("datasets").get(driver)
    common = None
    for item in dataset:
        if item['label'] is not None:
            cols = set(c for c in item['states'].columns if c != 'Time')
            common = cols if common is None else common & cols
    return sorted(common)


def show_downsample(driver=DRIVER, time_range=TIME_RANGE, downsamples=[1, 5, 10], features=DEFAULT_FEATURES):
    feature_names = features
    n_features = len(feature_names)
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 3 * n_features))

    for ds in downsamples:
        dataset = DatasetManager("datasets", downsample=ds).get(driver)
        t, X, _ = dataset.to_sequences(feature_names, time_range, pad=False)
        for i, ax in enumerate(axes):
            ax.plot(t[0], X[0][:, i], label=f"ds={ds}")

    for ax, name in zip(axes, feature_names):
        ax.set_title(name)
        ax.set_xlabel("Time (s)")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle(f"Downsample 비교 ({driver})", fontsize=14)
    plt.tight_layout()
    plt.show()


def show_smooth(driver=DRIVER, time_range=TIME_RANGE, cutoffs=[5.0, 12.0, 15.0], features=DEFAULT_FEATURES):
    feature_names = features
    fig, axes = plt.subplots(len(feature_names), 1, figsize=(12, 3 * len(feature_names)))

    dataset_raw = DatasetManager("datasets").get(driver)
    t_raw, X_raw, _ = dataset_raw.to_sequences(feature_names, time_range, pad=False)
    for i, ax in enumerate(axes):
        ax.plot(t_raw[0], X_raw[0][:, i], color='black', linewidth=1.5, label='Raw', zorder=10)

    for cutoff in cutoffs:
        dataset_sm = DatasetManager("datasets", smooth=True, smooth_cutoff=cutoff).get(driver)
        t_sm, X_sm, _ = dataset_sm.to_sequences(feature_names, time_range, pad=False)
        for i, ax in enumerate(axes):
            ax.plot(t_sm[0], X_sm[0][:, i], linewidth=1, label=f"cutoff={cutoff}Hz")

    for ax, name in zip(axes, feature_names):
        ax.set_title(name)
        ax.set_xlabel("Time (s)")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle(f"Smoothing 비교 ({driver})", fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # show_downsample()
    driver = "강신길"
    time_range = (5, 7)
    show_smooth(driver=driver, time_range=time_range)
