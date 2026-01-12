import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.utils.vpl_dataset import create_vpl_dataset

driver_names = ['강신길', '박재일']
time_range = [5, 7]
downsample = 5
features = ['IMU_VerAccelVal', 'Bounce_rate_6D', 'Pitch_rate_6D', 'IMU_LongAccelVal']  # v1 features
set_len = 10

print("Creating VPL dataset...")
print(f"Drivers: {driver_names}")
print(f"Time range: {time_range}")
print(f"Downsample: {downsample}")
print(f"Features: {features}")
print(f"Set length: {set_len}\n")

dataset, driver_map = create_vpl_dataset(
    driver_names=driver_names,
    time_range=time_range,
    downsample=downsample,
    features=features,
    set_len=set_len
)

print(f"\nDriver map: {driver_map}")
print(f"\nDataset keys: {dataset.keys()}")
print(f"Observations shape: {dataset['observations'].shape}")
print(f"Labels shape: {dataset['labels'].shape}")
print(f"Driver IDs shape: {dataset['driver_ids'].shape}")

_, set_len_actual, T, d = dataset['observations'].shape
encoder_input_dim = set_len_actual * (T * d + 1)

print(f"\nDimensions:")
print(f"  set_len: {set_len_actual}")
print(f"  T (timesteps): {T}")
print(f"  d (features): {d}")
print(f"  Encoder input dim: {encoder_input_dim}")

unique_driver_ids, counts = np.unique(dataset['driver_ids'], return_counts=True)
print(f"\nDriver distribution:")
for driver_id, count in zip(unique_driver_ids, counts):
    print(f"  Driver {driver_map[driver_id]}: {count} queries")

print(f"\nLabel distribution:")
all_labels = dataset['labels'].flatten()
n_true = np.sum(all_labels == 1)
n_false = np.sum(all_labels == 0)
print(f"  True: {n_true} ({n_true/len(all_labels)*100:.1f}%)")
print(f"  False: {n_false} ({n_false/len(all_labels)*100:.1f}%)")

print("\nDataset creation test passed!")
