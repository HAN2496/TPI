from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
from loader import Dataset, View

ROOT = Path(__file__).resolve().parents[2]

FEATURES = ["IMU_VerAccelVal", "Bounce_rate_6D", "Pitch_rate_6D", "IMU_LongAccelVal"]
LABELS = ["VerAccel", "BounceRate", "PitchRate", "LongAccel"]
DRIVER = "강신길"
AROUND = (0, 2)          # event_time(=5s) 기준. 구 TIME_RANGE=(5, 7)과 동일
DOWNSAMPLE = 1
N = 12  # 랜덤 trajectory 수

rng = np.random.default_rng(42)

view = View(features=FEATURES, around=AROUND, downsample=DOWNSAMPLE)
X, y = view(Dataset(ROOT / "datasets")[DRIVER])
i = rng.integers(len(X))

fig, axes = plt.subplots(len(FEATURES), 1, figsize=(4 * 1.2, 2 * 1 * len(FEATURES)))
# fig.suptitle(f"{DRIVER} - {'True' if y[i] else 'False'}", fontsize=13)

for d, (ax, label) in enumerate(zip(axes, LABELS)):
    ax.plot(X[i, :, d], lw=1.5, color='gray')
    ax.set_ylabel(label, fontsize=22)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_xticks([])

plt.tight_layout()
out = Path(__file__).resolve().parents[2] / "outputs/lab/dataset/trajectories.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=120)
print(f"Saved: {out}")
