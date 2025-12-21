import os
import scipy.io as sio
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt

def load_state_mat(mat_path):
    mat = sio.loadmat(mat_path)
    keys = [k for k in mat.keys() if not k.startswith("__")]

    data_dict = {}

    for key in keys:
        value = mat[key]
        if isinstance(value, np.ndarray):
            if value.ndim == 1:
                data_dict[key] = value
            elif value.ndim == 2 and (value.shape[0] == 1 or value.shape[1] == 1):
                data_dict[key] = value.flatten()
            else:
                raise ValueError(f"⚠ 경고: '{os.path.basename(mat_path)}'의 '{key}' 항목이 1차원 또는 벡터 형태가 아닙니다.")

    lengths = [len(v) for v in data_dict.values()]
    if len(set(lengths)) != 1:
        print(f"⚠ 경고: '{os.path.basename(mat_path)}'에서 컬럼 길이가 서로 다른 항목이 있습니다.")

    df = pd.DataFrame(data_dict)
    return df


def convert_mat_to_csv(mat_path, csv_path):
    df = load_state_mat(mat_path)
    df.to_csv(csv_path, index=False)
    return df


def convert_all_mat_in_root(root_dir, overwrite=False):
    root_dir = os.path.abspath(root_dir)
    print(f"🔍 루트 폴더 순회 시작: {root_dir}")

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(".mat"):
                mat_path = os.path.join(dirpath, filename)
                base, _ = os.path.splitext(filename)
                csv_filename = base + ".csv"
                csv_path = os.path.join(dirpath, csv_filename)

                if os.path.exists(csv_path) and not overwrite:
                    print(f"⏩ 스킵 (이미 존재): {csv_path}")
                    continue

                try:
                    convert_mat_to_csv(mat_path, csv_path)
                except Exception as e:
                    print(f"❌ 변환 실패: {mat_path}")
                    print(f"   이유: {e}")

    print("✅ 전체 변환 완료.")

def design_lpf(fs, cutoff=10.0, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def apply_lpf_to_df(df, cutoff=10.0, order=2):
    if "Time" not in df.columns:
        raise ValueError("DataFrame에 'Time' 컬럼이 없습니다.")

    t = df["Time"].values
    dt = np.median(np.diff(t))
    fs = 1.0 / dt

    b, a = design_lpf(fs, cutoff=cutoff, order=order)

    df_filt = df.copy()
    for col in df.columns:
        if col == "Time":
            continue
        x = df[col].values
        try:
            df_filt[col] = filtfilt(b, a, x)
        except Exception:
            df_filt[col] = x

    return df_filt

def smooth_all_csv_in_root(root_dir, overwrite=False, cutoff=10.0, order=2):
    """
    root_dir 내부의 모든 CSV 파일을 순회하며
    LPF 스무딩을 적용한 *_smooth.csv 파일을 생성한다.
    """
    root_dir = os.path.abspath(root_dir)
    print(f"🔍 CSV 스무딩 순회 시작: {root_dir}")

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(".csv") and not filename.lower().endswith("_smooth.csv"):
                csv_path = os.path.join(dirpath, filename)
                
                smooth_path = csv_path.replace(".csv", "_smooth.csv")
                
                if os.path.exists(smooth_path) and not overwrite:
                    print(f"⏩ 스킵 (이미 존재): {smooth_path}")
                    continue

                try:
                    df = pd.read_csv(csv_path)

                    # LPF 적용
                    df_smooth = apply_lpf_to_df(df, cutoff=cutoff, order=order)

                    df_smooth.to_csv(smooth_path, index=False)
                    print(f"✅ 스무딩 완료: {smooth_path}")

                except Exception as e:
                    print(f"❌ 스무딩 실패: {csv_path}")
                    print(f"   이유: {e}")

    print("🏁 CSV 스무딩 전체 완료.")

if __name__ == "__main__":
    convert_all_mat_in_root("datasets", overwrite=False)
    smooth_all_csv_in_root("datasets", overwrite=False, cutoff=12.0, order=2)
