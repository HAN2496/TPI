import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from pathlib import Path
from sklearn.cluster import DBSCAN
from collections import defaultdict

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def print_location_summary(df, topn_files=3):
    g = df[df['location_id'] != -1].groupby('location_id')

    summary = g.agg(
        n=('location_id', 'size'),

        start_lat_mean=('start_lat', 'mean'),
        start_lon_mean=('start_lon', 'mean'),
        end_lat_mean=('end_lat', 'mean'),
        end_lon_mean=('end_lon', 'mean'),

        start_lat_std=('start_lat', 'std'),
        start_lon_std=('start_lon', 'std'),
        end_lat_std=('end_lat', 'std'),
        end_lon_std=('end_lon', 'std'),

        start_lat_min=('start_lat', 'min'),
        start_lat_max=('start_lat', 'max'),
        start_lon_min=('start_lon', 'min'),
        start_lon_max=('start_lon', 'max'),

        end_lat_min=('end_lat', 'min'),
        end_lat_max=('end_lat', 'max'),
        end_lon_min=('end_lon', 'min'),
        end_lon_max=('end_lon', 'max'),
    ).sort_values('n', ascending=False)

    print("\n=== Location Summary (excluding noise) ===")
    for loc_id, row in summary.iterrows():
        print(f"\n[location_id={loc_id}] n={int(row['n'])}")
        print(f"  start_mean: ({row['start_lat_mean']:.6f}, {row['start_lon_mean']:.6f})")
        print(f"  end_mean  : ({row['end_lat_mean']:.6f}, {row['end_lon_mean']:.6f})")

        # std (NaN 방지: n=1이면 std가 NaN)
        s_lat_std = row['start_lat_std']
        s_lon_std = row['start_lon_std']
        e_lat_std = row['end_lat_std']
        e_lon_std = row['end_lon_std']
        print(f"  start_std : ({(0.0 if pd.isna(s_lat_std) else s_lat_std):.6f}, {(0.0 if pd.isna(s_lon_std) else s_lon_std):.6f})")
        print(f"  end_std   : ({(0.0 if pd.isna(e_lat_std) else e_lat_std):.6f}, {(0.0 if pd.isna(e_lon_std) else e_lon_std):.6f})")

        print(f"  start_box : lat[{row['start_lat_min']:.6f}, {row['start_lat_max']:.6f}] "
              f"lon[{row['start_lon_min']:.6f}, {row['start_lon_max']:.6f}]")
        print(f"  end_box   : lat[{row['end_lat_min']:.6f}, {row['end_lat_max']:.6f}] "
              f"lon[{row['end_lon_min']:.6f}, {row['end_lon_max']:.6f}]")

        files = df[df['location_id'] == loc_id]['filename'].head(topn_files).tolist()
        print(f"  sample_files: {files}")

    noise_n = (df['location_id'] == -1).sum()
    if noise_n:
        print(f"\n[noise] n={noise_n}")
        print("  sample_files:", df[df['location_id'] == -1]['filename'].head(topn_files).tolist())



def parse_kml_coordinates(kml_path):
    """
    KML 파일에서 좌표 리스트를 파싱하여 반환합니다.
    반환값: [(lon, lat, alt), ...] 형태의 numpy array
    """
    try:
        tree = ET.parse(kml_path)
        root = tree.getroot()
        
        # Namespace 처리 (KML은 보통 xmlns가 있음)
        ns = {'kml': 'http://www.opengis.net/kml/2.2'}
        
        # <coordinates> 태그 찾기 (LineString 내부)
        # 여러 경로가 있을 수 있으나 첫 번째 Placemark의 경로를 사용한다고 가정
        coords_text = root.find('.//kml:coordinates', ns)
        
        if coords_text is not None and coords_text.text:
            coords_str = coords_text.text.strip().split()
            coords = []
            for c in coords_str:
                if not c.strip():
                    continue
                parts = c.split(',')
                # Ensure we have at least lon, lat and they are not empty
                if len(parts) >= 2 and parts[0].strip() and parts[1].strip():
                    try:
                        lon = float(parts[0])
                        lat = float(parts[1])
                        alt = float(parts[2]) if len(parts) > 2 and parts[2].strip() else 0.0
                        coords.append([lon, lat, alt])
                    except ValueError:
                        continue # Skip malformed coordinates
            return np.array(coords)
    except Exception as e:
        print(f"Error parsing {kml_path}: {e}")
        return None
    return None

def load_gps_data(datasets_dir="datasets"):
    """
    datasets 폴더를 순회하며 모든 KML 파일을 찾아 GPS 정보를 추출합니다.
    """
    data_list = []
    
    # datasets 폴더 하위의 모든 폴더 탐색
    search_path = os.path.join(datasets_dir, "**", "*.kml")
    kml_files = glob.glob(search_path, recursive=True)
    
    print(f"Found {len(kml_files)} KML files.")
    
    for kml_file in kml_files:
        coords = parse_kml_coordinates(kml_file)
        if coords is None or len(coords) == 0:
            continue
            
        # 대표 좌표 추출 (경로의 중간점 사용)
        # 시작점, 끝점, 중간점 등을 피처로 쓸 수 있음. 여기서는 중간점(Mean) 사용.
        mean_coord = np.mean(coords, axis=0)
        
        # 파일명에서 정보 추출 (예: 20251124_064637_gps_True.kml)
        filename = os.path.basename(kml_file)
        parent_dir = os.path.basename(os.path.dirname(kml_file))
        
        # Driver 정보 등을 info 파일에서 읽을 수도 있지만, 일단 파일 구조상 추론 어렵다면 패스
        # 여기서는 부모 폴더명 등을 활용하거나 나중에 병합
        
        data_list.append({
            'file_path': kml_file,
            'filename': filename,
            'dataset_folder': parent_dir,
            'mean_lon': mean_coord[0],
            'mean_lat': mean_coord[1],
            'start_lon': coords[0][0],
            'start_lat': coords[0][1],
            'end_lon': coords[-1][0],
            'end_lat': coords[-1][1]
        })
        
    return pd.DataFrame(data_list)

def cluster_locations(df, eps=0.005, min_samples=3):
    """
    GPS 좌표를 기반으로 DBSCAN 클러스터링을 수행하여 장소 ID를 부여합니다.
    eps: 클러스터링 거리 임계값 (도 단위, 0.001도 approx 100m)
    """
    # 클러스터링에 사용할 좌표 (위도, 경도)
    # 시작점과 끝점이 비슷하면 같은 경로로 볼 수 있음 -> [start_lat, start_lon, end_lat, end_lon] 사용
    X = df[['start_lat', 'start_lon']].values  # ✅ start-only

    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = db.fit_predict(X)

    df['location_id'] = labels
    return df

def visualize_clusters(df):
    """
    클러스터링 결과를 산점도로 시각화합니다.
    """
    plt.figure(figsize=(12, 8))
    
    # 노이즈(Outlier, -1)와 클러스터를 분리
    unique_labels = sorted(df['location_id'].unique())
    
    # 색상맵 생성
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        if label == -1:
            color = 'k' # Noise는 검은색
            label_name = "Noise"
            alpha = 0.3
            marker = 'x'
        else:
            label_name = f"Location {label}"
            alpha = 0.8
            marker = 'o'
            
        subset = df[df['location_id'] == label]
        
        # 시작점만 찍어서 위치 확인
        plt.scatter(subset['start_lon'], subset['start_lat'], 
                    c=[color], label=label_name, alpha=alpha, marker=marker, s=50, edgecolors='w')
        
        # 텍스트 라벨 (클러스터 중심에 하나만)
        if label != -1:
            center_x = subset['start_lon'].mean()
            center_y = subset['start_lat'].mean()
            plt.text(center_x, center_y, str(label), fontsize=12, fontweight='bold')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('GPS Location Clustering (Start Points)')
    plt.legend(bbox_to_anchor=(0.95, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = "artifacts/location_clustering.png"
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")
    plt.close()

def main():
    print("Loading GPS data...")
    df = load_gps_data()
    
    if df.empty:
        print("No GPS data found.")
        return

    print(f"Loaded {len(df)} GPS tracks.")
    
    # 클러스터링 수행
    # eps=0.002는 대략 200m 정도의 오차 허용 (위도 기준)
    print("Clustering locations...")
    df = cluster_locations(df, eps=0.002, min_samples=2)
    # print_location_summary(df, topn_files=3)

    # 결과 요약
    n_clusters = df['location_id'].nunique() - (1 if -1 in df['location_id'].values else 0)
    print(f"Found {n_clusters} unique locations (excluding noise).")
    print(df['location_id'].value_counts())
    
    # 결과 저장
    save_path = "artifacts/gps_locations.csv"
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"Clustering results saved to {save_path}")
    
    # 시각화
    visualize_clusters(df)

    loc_id = 64

    subset = df[df['location_id'] == loc_id]

    print(f"location_id={loc_id}, n={len(subset)}")
    print(subset[['filename', 'start_lat', 'start_lon']])

    print(
        subset[['filename', 'start_lat', 'start_lon']]
        .sort_values(['start_lat', 'start_lon'])
        .to_string(index=False)
    )


if __name__ == "__main__":
    os.makedirs("artifacts", exist_ok=True)
    main()
