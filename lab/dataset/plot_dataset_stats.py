from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[2]
rows = []
for state in (root / "datasets").rglob("*_state_*.csv"):
    if state.name.endswith("_smooth.csv"):
        continue
    # if "old" in state.relative_to(root / "datasets").parts:
    #     continue
    prefix, label = state.stem.split("_state_")
    info = state.with_name(f"{prefix}_info_{label}.txt")
    driver = re.search(r'"Driver"\s*:\s*"([^"]+)"', info.read_text(encoding="utf-8")).group(1)
    rows.append({"driver": driver, "label": label})

df = pd.DataFrame(rows)
counts = pd.crosstab(df.driver, df.label).reindex(columns=["True", "False", "None"], fill_value=0).sort_index()

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
ax = counts.plot.bar(figsize=(18, 8), width=0.75, color=["#4CAF50", "#E15759", "#999999"])
ax.set_title("Dataset Distribution by Driver and Label", fontsize=14, weight="bold")
ax.set_xlabel("Driver (Labeler)", fontsize=11, weight="bold")
ax.set_ylabel("Number of Datasets", fontsize=11, weight="bold")
ax.grid(axis="y", alpha=0.3)
ax.legend()
ax.set_ylim(0, max(10, counts.to_numpy().max() * 1.1))

for bars in ax.containers:
    ax.bar_label(bars, labels=[int(b.get_height()) or "" for b in bars], padding=2, fontsize=9)
for i, total in enumerate(counts.sum(axis=1)):
    ax.text(i, ax.get_ylim()[1] * 0.96, f"Total: {total}", ha="center", weight="bold", fontsize=10)

plt.tight_layout()
out = root / "outputs/dataset_stats.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150)
print(out)
