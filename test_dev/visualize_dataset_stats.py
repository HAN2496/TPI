import os
import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---- Korean font setup (Windows) ----
mpl.rcParams['font.family'] = 'Malgun Gothic'   # 맑은 고딕
mpl.rcParams['axes.unicode_minus'] = False      # 마이너스(-) 깨짐 방지

def scan_datasets(base_folder='datasets'):
    """
    Scan all datasets and collect statistics by driver and label.

    Returns:
        dict: {driver: {label: count}}
    """
    stats = defaultdict(lambda: {'True': 0, 'False': 0, 'None': 0})

    for dirpath, dirnames, filenames in os.walk(base_folder):
        info_files = [f for f in filenames if f.endswith('.txt') and '_info_' in f]

        for info_file in info_files:
            info_path = os.path.join(dirpath, info_file)

            # Extract label from filename: <timestamp>_info_<label>.txt
            label = info_file.split('_info_')[1].replace('.txt', '')

            # Read driver info
            with open(info_path, 'r', encoding='utf-8') as f:
                info = json.load(f)

            driver = info.get('Driver')
            if driver:
                stats[driver][label] += 1

    return dict(stats)

def plot_dataset_stats(stats, save_path='artifacts/dataset_stats.png'):
    """
    Plot dataset statistics as grouped bar chart.

    Args:
        stats: {driver: {label: count}}
        save_path: Path to save the figure
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    drivers = sorted(stats.keys())
    labels = ['True', 'False', 'None']

    # Prepare data
    data = {label: [stats[driver].get(label, 0) for driver in drivers] for label in labels}

    # Calculate totals for each driver
    totals = [sum(stats[driver].values()) for driver in drivers]

    # Bar positions
    x = np.arange(len(drivers))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot bars
    bars_true = ax.bar(x - width, data['True'], width, label='True', color='#2ca02c', alpha=0.8)
    bars_false = ax.bar(x, data['False'], width, label='False', color='#d62728', alpha=0.8)
    bars_none = ax.bar(x + width, data['None'], width, label='None', color='#7f7f7f', alpha=0.8)

    # Add value labels on bars
    for bars in [bars_true, bars_false, bars_none]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=9)

    # Add total count above each driver
    max_bar_height = max(
        max(data['True']),
        max(data['False']),
        max(data['None'])
    )

    # y축 상단 여유 확보 (20% 여백)
    ax.set_ylim(0, max_bar_height * 1.10)

    for i, total in enumerate(totals):
        ax.text(
            i,
            max_bar_height * 1.05,   # 막대와 확실히 분리
            f'Total: {total}',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )

    # Labels and title
    ax.set_xlabel('Driver (Labeler)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Datasets', fontsize=12, fontweight='bold')
    ax.set_title('Dataset Distribution by Driver and Label', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(drivers, fontsize=11)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Figure saved to: {save_path}")
    plt.close()

    # Print summary
    print("\n=== Dataset Summary ===")
    for driver in drivers:
        driver_stats = stats[driver]
        total = sum(driver_stats.values())
        print(f"\n{driver}:")
        print(f"  Total: {total}")
        print(f"  True:  {driver_stats['True']:3d} ({driver_stats['True']/total*100:5.1f}%)")
        print(f"  False: {driver_stats['False']:3d} ({driver_stats['False']/total*100:5.1f}%)")
        print(f"  None:  {driver_stats['None']:3d} ({driver_stats['None']/total*100:5.1f}%)")

if __name__ == '__main__':
    stats = scan_datasets()
    plot_dataset_stats(stats)
