import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
from src.data.splits import _load_dataset_sequences

FEATURES = ["IMU_VerAccelVal", "Bounce_rate_6D", "Pitch_rate_6D", "IMU_LongAccelVal"]
LABELS = ["VerAccel", "BounceRate", "PitchRate", "LongAccel"]
DRIVER = "강신길"
TIME_RANGE = (5, 7)
DOWNSAMPLE = 1
N = 12  # 랜덤 trajectory 수

config = {'features': FEATURES}
rng = np.random.default_rng(42)

X, y = _load_dataset_sequences(DRIVER, TIME_RANGE, DOWNSAMPLE, config)
i = rng.integers(len(X))

fig, axes = plt.subplots(len(FEATURES), 1, figsize=(4 * 1.2, 2 * 1 * len(FEATURES)))
# fig.suptitle(f"{DRIVER} - {'True' if y[i] else 'False'}", fontsize=13)

for d, (ax, label) in enumerate(zip(axes, LABELS)):
    ax.plot(X[i, :, d], lw=1.5, color='gray')
    ax.set_ylabel(label, fontsize=22)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_xticks([])

plt.tight_layout()
out = Path(__file__).parent / "trajectories.png"
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
