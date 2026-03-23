"""
Forward Sequential Feature Selection
한계 정보량(Marginal AUROC Gain) 기반으로 핵심 변수를 하나씩 선택합니다.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from pathlib import Path
from src.utils import DatasetManager, convert_driver_name, ExperimentLogger


class PIModel(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1).mean(dim=1)


def load_data(driver_name, time_range, downsample, smooth=False, smooth_cutoff=12.0):
    manager = DatasetManager("datasets", downsample=downsample, smooth=smooth, smooth_cutoff=smooth_cutoff)
    dataset = manager.get(driver_name)

    ignore_cols = {'IEB_StrkDpthPcVal', 'VCU_AccPedDepVal', 'SAS_AnglVal', 'SAS_SpdVal', 'VCU_MotTqCmdFrntVal', 'VCU_MotTqCmdRearVal',
                   'ABS_ActvSta', 'IEB_BrkActvSta', 'SAS_AlvCnt1Val', 'SAS_Crc1Val', 'SAS_IntSta', 'VCU_GearPosSta', 'Time',
                   'IEB_EstTtlBrkFrcNmV', 'MCU_Mg1EstTqPcVal', 'MCU_Mg1EstTqVal', 'MCU_Mg2EstTqPcVal', 'MCU_Mg2EstTqVal',
                   'WHL_SpdFLVal', 'WHL_SpdFRVal', 'WHL_SpdRLVal', 'WHL_SpdRRVal'}

    common_cols = None
    for item in dataset:
        if item['label'] is not None:
            cols = set(c for c in item['states'].columns if c not in ignore_cols)
            common_cols = cols if common_cols is None else common_cols & cols
    feature_names = sorted(common_cols)

    _, X_np, y_np = dataset.to_sequences(feature_names, time_range, fill_value=0.0, pad=True)

    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32).view(-1) 

    mean = X.mean(dim=(0, 1), keepdim=True)
    std = X.std(dim=(0, 1), keepdim=True)
    X = (X - mean) / (std + 1e-8)

    return X, y, feature_names


def evaluate_feature_subset(X, y, selected_indices, epochs=50, batch_size=256, n_splits=5):
    X_subset = X[:, :, selected_indices]
    
    # 클래스 불균형을 고려하여 분할 비율을 유지하는 StratifiedKFold 적용
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    auc_scores = []
    
    # PyTorch Tensor를 numpy 배열로 변환하여 split 인덱스 추출
    y_np = y.numpy()
    
    for train_idx, val_idx in skf.split(np.zeros(len(y_np)), y_np):
        X_train, X_val = X_subset[train_idx], X_subset[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = PIModel(len(selected_indices))
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        criterion = nn.BCEWithLogitsLoss()
        
        loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        
        # 1. 학습 데이터(Train)로만 파라미터 업데이트
        for _ in range(epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                loss = criterion(model(batch_X), batch_y)
                loss.backward()
                optimizer.step()
                
        # 2. 한 번도 보지 못한 검증 데이터(Validation)로 AUROC 평가
        with torch.no_grad():
            val_preds = torch.sigmoid(model(X_val)).numpy()
            auc = roc_auc_score(y_val.numpy(), val_preds)
            auc_scores.append(auc)
            
    # 5번의 Fold 교차 검증 결과 평균치 반환
    return np.mean(auc_scores)


def main(driver_name, time_range, downsample, target_features, epochs=30, batch_size=256, n_splits=5, smooth=False, smooth_cutoff=12.0):
    driver_name = convert_driver_name(driver_name)
    X, y, feature_names = load_data(driver_name, time_range, downsample, smooth=smooth, smooth_cutoff=smooth_cutoff)
    print(f"Loaded: {X.shape[0]} episodes, {X.shape[1]} timesteps, {len(feature_names)} features")

    selected_indices = []
    selected_names = []
    current_best_auc = 0.5

    num_rounds = min(target_features, len(feature_names))
    print(f"\n--- Forward Feature Selection (Target: Top {num_rounds}/{len(feature_names)}) ---")

    for round_idx in range(1, num_rounds + 1):
        print(f"\n[Round {round_idx}]")
        best_gain = -1.0
        best_feature_idx = -1
        best_round_auc = 0.0

        for i, name in enumerate(feature_names):
            if i in selected_indices:
                continue
            auc = evaluate_feature_subset(X, y, selected_indices + [i], epochs=epochs, batch_size=batch_size, n_splits=n_splits)
            gain = auc - current_best_auc
            # print(f"  {name:>30} | AUROC: {auc:.4f} (Gain: {gain:+.4f})")

            if gain > best_gain:
                best_gain = gain
                best_feature_idx = i
                best_round_auc = auc

        selected_indices.append(best_feature_idx)
        selected_names.append(feature_names[best_feature_idx])
        current_best_auc = best_round_auc

        print(f"  Winner: '{feature_names[best_feature_idx]}'")
        print(f"  Selected: {selected_names}")
        print(f"  AUROC: {current_best_auc:.4f}")

    ranks = {name: (selected_names.index(name) + 1 if name in selected_names else num_rounds + 1)
             for name in feature_names}
    return ranks


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--driver", default=None, help="특정 드라이버 지정 (미입력 시 전체)")
    parser.add_argument("-t", "--time-range", type=float, nargs=2, default=[5, 8])
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("-n", "--target-features", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--smooth", action="store_true")
    parser.add_argument("--smooth-cutoff", type=float, default=12.0)
    args = parser.parse_args()

    drivers = [args.driver] if args.driver else DatasetManager("datasets").keys()

    save_dir = Path("artifacts/analysis")
    save_dir.mkdir(parents=True, exist_ok=True)

    all_ranks = {}  # driver -> {feature -> rank}

    with ExperimentLogger(str(save_dir), "sequential_feature_selection", add_timestamp=False):
        for idx, driver in enumerate(drivers):
            print(f"\n{'='*60}")
            print(f"[{idx+1}/{len(drivers)}] Driver: {driver}")
            print(f"{'='*60}")
            all_ranks[driver] = main(driver, args.time_range, args.downsample, args.target_features,
                                     epochs=args.epochs, batch_size=args.batch_size, n_splits=args.n_splits,
                                     smooth=args.smooth, smooth_cutoff=args.smooth_cutoff)

        all_features = sorted({f for ranks in all_ranks.values() for f in ranks})
        driver_list = list(all_ranks.keys())

        print(f"\n\n{'='*60}")
        print("Feature Ranking Summary")
        print(f"drivers: {driver_list}")
        print(f"{'='*60}")

        summary = []
        for feature in all_features:
            ranks = [all_ranks[d].get(feature, args.target_features + 1) for d in driver_list]
            summary.append((feature, ranks, np.mean(ranks)))

        summary.sort(key=lambda x: x[2])
        for feature, ranks, avg in summary:
            print(f"  {feature:<35} {ranks}  avg: {avg:.2f}")
