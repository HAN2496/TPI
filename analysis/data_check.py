import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.signal import butter, filtfilt

from src.utils.data_loader import DatasetManager
from src.utils import ExperimentPaths, ExperimentLogger

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

def design_lpf(fs, cutoff, order=2):
    """
    Butterworth LPF 설계
    fs     : 샘플링 주파수 [Hz]
    cutoff : 차단 주파수 [Hz]
    order  : 필터 차수
    """
    nyq = 0.5 * fs  # 나이퀴스트 주파수
    normal_cutoff = cutoff / nyq  # 정규화 차단 주파수 (0~1)
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a



def apply_lpf_to_states(states, cutoff=2.0, order=2):
    """
    states: Time 포함된 DataFrame
    cutoff: 차단 주파수 [Hz]
    order : 필터 차수
    """
    # 샘플링 주파수 추정
    t = states['Time'].values
    dt = np.median(np.diff(t))
    fs = 1.0 / dt

    b, a = design_lpf(fs, cutoff=cutoff, order=order)

    filtered = states.copy()
    for col in states.columns:
        if col == 'Time':
            continue
        x = states[col].values
        # 양방향 필터링으로 위상 지연 제거
        filtered[col] = filtfilt(b, a, x)

    return filtered

def analyze_dataset_structure():
    """Analyze and print dataset structure and label distribution"""
    manager = DatasetManager("datasets")

    # --- 1) 데이터셋 구조 확인용 출력 ---
    print("총 Driver 수:", len(manager))
    print("Driver 목록:", manager.keys())

    # --- 2) 사람별 True/False/None 개수 집계 ---
    per_person = defaultdict(lambda: {"True": 0, "False": 0, "None": 0})

    for driver_name, dataset in manager:
        print(f"\n{driver_name}: {len(dataset)} 주행 데이터")

        for idx, run_data in enumerate(dataset):
            label = run_data['label']
            timestamp = run_data['timestamp']

            if label is True:
                label_str = "True"
            elif label is False:
                label_str = "False"
            else:
                label_str = "None"
            per_person[driver_name][label_str] += 1

    return per_person

# --- 3) 스택 막대 시각화 ---
def print_label_counts(per_person):
    people = list(per_person.keys())
    true_vals  = [per_person[p]["True"]  for p in people]
    false_vals = [per_person[p]["False"] for p in people]
    none_vals  = [per_person[p]["None"]  for p in people]

    x = np.arange(len(people))

    plt.figure(figsize=(max(8, len(people) * 0.8), 4))
    plt.bar(x, true_vals, label="True")
    for i, val in enumerate(true_vals):
        plt.text(i, val/2, str(val), ha='center', va='center', fontsize=10, color='white', fontweight='bold')

    plt.bar(x, false_vals, bottom=true_vals, label="False")
    for i, val in enumerate(false_vals):
        plt.text(i, true_vals[i] + val / 2, str(val), ha='center', va='center', fontsize=10, color='white', fontweight='bold')

    bottom_tf = [t + f for t, f in zip(true_vals, false_vals)]
    plt.bar(x, none_vals, bottom=bottom_tf, label="None")
    for i, val in enumerate(none_vals):
        plt.text(i, bottom_tf[i] + val / 2, str(val), ha='center', va='center', fontsize=10, color='white', fontweight='bold')

    plt.xticks(x, people, rotation=45, ha="right")
    plt.ylabel("Count")
    plt.title("Label distribution per person (Driver 기준)")
    plt.legend()
    plt.tight_layout()

    # Save to shared directory (cross-driver analysis)
    shared_dir = ExperimentPaths.get_shared_dir(create=True)
    plt.savefig(shared_dir / "label_distribution.png")
    print(f"Label distribution saved to {shared_dir / 'label_distribution.png'}")
    plt.close()


# --- 4) 한 사람의 하나의 데이터에 대한 모든 feature plot ---
def plot_all_features_of_sample(driver_name="강신길"):
    driver_name = driver_name
    manager = DatasetManager("datasets")
    sample_dataset = manager.get(driver_name)
    sample_run = sample_dataset[0]  # 첫 번째 주행 데이터

    states = sample_run['states']
    label = sample_run['label']
    timestamp = sample_run['timestamp']

    # Time 컬럼 제외한 모든 컬럼들 가져오기
    feature_cols = [col for col in states.columns if col != 'Time']
    n_features = len(feature_cols)

    # subplot 레이아웃 계산 (행 x 열)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3))
    axes = axes.flatten() if n_features > 1 else [axes]

    for i, col in enumerate(feature_cols):
        ax = axes[i]
        ax.plot(states['Time'], states[col], linewidth=1)
        # ax.set_xlabel('Time (s)')
        ax.set_ylabel(col)
        ax.set_title(col)
        ax.grid(alpha=0.3)

    # 빈 subplot 숨기기
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f'All Features - Driver: {driver_name}, Label: {label}, Timestamp: {timestamp}',
                fontsize=14, y=1.00)
    plt.tight_layout()

    # Use new path structure
    paths = ExperimentPaths(driver_name)
    features_dir = paths.get_analysis_dir()
    plt.savefig(features_dir / "all_features_sample.png")
    plt.close()

def plot_all_features_with_lpf(driver_name="강신길", cutoff=2.0, order=2):
    manager = DatasetManager("datasets")
    sample_dataset = manager.get(driver_name)
    sample_run = sample_dataset[0]  # 첫 번째 주행 데이터

    states = sample_run['states']
    label = sample_run['label']
    timestamp = sample_run['timestamp']

    # LPF 적용
    states_lpf = apply_lpf_to_states(states, cutoff=cutoff, order=order)

    feature_cols = [col for col in states.columns if col != 'Time']
    n_features = len(feature_cols)

    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3))
    axes = axes.flatten() if n_features > 1 else [axes]

    for i, col in enumerate(feature_cols):
        ax = axes[i]
        t = states['Time']

        # 원본
        ax.plot(t, states[col], linewidth=1.2,  linestyle='--', color='gray', label='Raw')

        # LPF
        ax.plot(t, states_lpf[col], linewidth=0.8, color='red', label=f'LPF ({cutoff} Hz)')

        ax.set_ylabel(col)
        ax.set_title(col)
        ax.grid(alpha=0.3)

        # 각 subplot마다 legend 넣으면 지저분하니, 첫 번째에만
        if i == 0:
            ax.legend()

    # 남는 subplot 숨기기
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(
        f'All Features with LPF - Driver: {driver_name}, Label: {label}, Timestamp: {timestamp}',
        fontsize=14,
        y=1.00
    )
    plt.tight_layout()

    paths = ExperimentPaths(driver_name)
    features_dir = paths.get_analysis_dir()
    save_path = features_dir / f"all_features_sample_lpf_{cutoff}Hz.png"
    plt.savefig(save_path, dpi=300)
    print(f"LPF feature plot saved to {save_path}")
    plt.close()

def print_all_features(states):
    feature_cols = [col for col in states.columns if col != 'Time']
    print("Features in the dataset:")
    for col in feature_cols:
        print(f"        \"{col}\", ")

if __name__ == "__main__":
    # Setup logging in shared directory (cross-driver analysis)
    shared_dir = ExperimentPaths.get_shared_dir(create=True)
    logger = ExperimentLogger(str(shared_dir), "data_check_results", add_timestamp=False)
    logger.start()

    # per_person = analyze_dataset_structure()
    # print_label_counts(per_person)
    # plot_all_features_of_sample(driver_name="강신길")
    # plot_all_features_with_lpf(driver_name="강신길", cutoff=12.0, order=2)
    # plot_all_features_of_sample(driver_name="박재일")
    # plot_all_features_of_sample(driver_name="한규택")

    # Stop logging
    logger.stop()
    print(f"\nData check results saved to {shared_dir / 'data_check_results.txt'}")

    print_all_features(DatasetManager("datasets").get("강신길")[0]['states'])