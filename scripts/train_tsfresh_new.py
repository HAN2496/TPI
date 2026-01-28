"""
DRM 개념 포함한 tsfreshpipeline
(Rethinking Diverse Human Preference Learning
through Principal Component Analysis)
"""

import os
import numpy as np
import json
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.manifold import TSNE

# --- Custom Modules ---
from src.utils.utils import _load_dataset_sequences
from src.model.tsfresh.pipeline_new import TSFreshPipeline
# 시각화 모듈에 새로 추가한 함수들이 포함되어 있다고 가정합니다.
from src.model.tsfresh.visualization import (
    plot_latent_space, 
    plot_roc_curve, 
    plot_reward_distribution,
    plot_reward_scatter,
    plot_feature_contribution,   # [New] Global Basis 해석
    plot_user_preference_radar   # [New] 사용자 성향 분석
)

VEHICLE_TSFRESH_SETTINGS = {
    "maximum": None,
    "minimum": None,
    "mean": None,
    "median": None,
    "standard_deviation": None,
    "variance": None,
    "skewness": None,
    "kurtosis": None,
    "abs_energy": None,
    "absolute_maximum": None,
    "mean_abs_change": None,
    
    # 피크 관련 (급거동)
    "number_peaks": [{"n": 1}, {"n": 3}],
    
    # FFT (주파수) - 여기가 핵심! coeff를 0~9까지만 제한
    "fft_coefficient": [
        {"coeff": k, "attr": "real"} for k in range(20)
    ] + [
        {"coeff": k, "attr": "abs"} for k in range(20)
    ],
    
    # 자기상관
    "autocorrelation": [{"lag": k} for k in range(1, 20)],
}

# --- Configuration ---
FLAGS = {
    "timestamp": None,
    "save_root": "artifacts/tsfresh_new",
    "random_state": 42,

    # Data Settings
    "features": ["IMU_VerAccelVal", "Bounce_rate_6D", "Pitch_rate_6D", "IMU_LongAccelVal"],
    "train_driver_names": ["김진명", "김태근", "조현석", "한규택", "박재일", "이지환"],
    "test_driver_names": ["강신길"],
    "time_range": (5, 7.5),
    "downsample": 3,

    # Pipeline Settings
    "pca_components": 15,
    "tsfresh_settings": VEHICLE_TSFRESH_SETTINGS, # 'minimal', 'efficient', 'comprehensive', or custom dict

    # Evaluation Settings
    "k_fold": 5,
}

def main():
    # --- 0. Setup Directories ---
    if FLAGS["timestamp"] is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(FLAGS["save_root"]) / timestamp
        is_training = True
    else:
        timestamp = FLAGS["timestamp"]
        log_dir = Path(FLAGS["save_root"]) / timestamp
        is_training = False
    
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Fold별 결과를 저장할 폴더 생성
    folds_dir = log_dir / "folds"
    folds_dir.mkdir(exist_ok=True)

    pipeline_path = log_dir / "global_basis_pipeline.pkl"
    config = {'features': FLAGS["features"]}

    # --- 1. Global Preference Basis (Common for all folds) ---
    if is_training:
        print("\n[Step 1] Building Global Preference Basis (DRM Approach)...")
        all_X = []
        all_y = []
        
        for driver in FLAGS["train_driver_names"]:
            print(f"  Loading pool data: {driver}...")
            X, y = _load_dataset_sequences(driver, FLAGS["time_range"], FLAGS["downsample"], config)
            all_X.append(X)
            all_y.extend(y)
            
        X_train_total = np.concatenate(all_X, axis=0)
        y_train_total = np.array(all_y).astype(int)
        
        print(f"  Total Pool Size: {X_train_total.shape}, Labels: {y_train_total.shape}")

        pipeline = TSFreshPipeline(
            feature_names=FLAGS["features"], 
            n_components=FLAGS["pca_components"], 
            extraction_settings=FLAGS["tsfresh_settings"],
            verbose=1
        )

        print("\n  Fitting Pipeline on Preference Differences (Global)...")
        # Difference PCA 수행
        pipeline.fit(X_train_total, y_train_total, use_preference_diff=True)
        pipeline.save(pipeline_path)
        print("  Global Basis Pipeline Saved.")

    else:
        print(f"\n[Inference] Loading Global Basis from {pipeline_path}...")
        pipeline = TSFreshPipeline.load(pipeline_path)

    flags_path = log_dir / "flags.json"
    if is_training:
        with open(flags_path, "w", encoding='utf-8') as f:
            json.dump(FLAGS, f, indent=4, ensure_ascii=False)
    else:
        if flags_path.exists():
            with open(flags_path, "r", encoding='utf-8') as f:
                loaded_flags = json.load(f)
                print("Loaded flags from training experiment.")

    # --- [Viz 1] Global Basis Interpretation (Loop 밖에서 1회 수행) ---
    print("\n  Saving Global Basis Feature Contributions...")
    plot_feature_contribution(
        pipeline, 
        feature_names=FLAGS["features"], 
        n_top=10, 
        save_path=log_dir / "global_basis_contribution.png"
    )

    # --- 2. Test-Time Adaptation with K-Fold ---
    test_driver = FLAGS["test_driver_names"][0]
    print(f"\n[Step 2] Adapting to Test Driver: {test_driver} (K-Fold={FLAGS['k_fold']})...")

    # Load Test Data
    X_test_raw, y_test_raw = _load_dataset_sequences(test_driver, FLAGS["time_range"], FLAGS["downsample"], config)
    y_test_raw = y_test_raw.astype(int)

    # Project Test Data onto Global Basis (Z = phi(x) * W)
    print("  Projecting test data onto Global Basis...")
    Z_test_basis = pipeline.transform(X_test_raw)
    
    # K-Fold Setup
    skf = StratifiedKFold(n_splits=FLAGS["k_fold"], shuffle=True, random_state=FLAGS["random_state"])
    
    # Result Containers
    fold_auroc = []
    fold_acc = []
    oof_preds = np.zeros(len(y_test_raw))      # Out-Of-Fold Predictions
    oof_targets = np.zeros(len(y_test_raw))    # 실제 타겟 (Shuffle 섞임 방지 위해 저장)
    user_weights_list = []
    
    print("\n  Starting Cross-Validation...")
    
    for fold_idx, (adapt_idx, eval_idx) in enumerate(skf.split(Z_test_basis, y_test_raw)):
        print(f"\n  --- Fold {fold_idx+1}/{FLAGS['k_fold']} ---")
        
        # Split Data
        Z_adapt, Z_eval = Z_test_basis[adapt_idx], Z_test_basis[eval_idx]
        y_adapt, y_eval = y_test_raw[adapt_idx], y_test_raw[eval_idx]
        
        # Train User-Specific Weights (Adaptation)
        clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=FLAGS["random_state"])
        clf.fit(Z_adapt, y_adapt)
        
        # Inference
        y_pred_prob = clf.predict_proba(Z_eval)[:, 1]
        y_pred = clf.predict(Z_eval)
        
        # OOF 저장
        oof_preds[eval_idx] = y_pred_prob
        oof_targets[eval_idx] = y_eval
        
        # Metrics
        auroc = roc_auc_score(y_eval, y_pred_prob)
        acc = accuracy_score(y_eval, y_pred)
        
        fold_auroc.append(auroc)
        fold_acc.append(acc)
        user_weights_list.append(clf.coef_.flatten())
        
        print(f"    AUROC: {auroc:.4f}, Accuracy: {acc:.4f}")

        # --- Save Per-Fold Artifacts ---
        fold_prefix = folds_dir / f"fold_{fold_idx+1}"
        
        # 1. Save Model
        joblib.dump(clf, f"{fold_prefix}_model.pkl")
        
        # 2. Save ROC Curve
        plot_roc_curve(
            y_eval, y_pred_prob, 
            title=f"ROC Curve (Fold {fold_idx+1}) - {test_driver}",
            save_path=f"{fold_prefix}_roc.png"
        )
        
        # 3. Save Reward Scatter (Good/Bad Separation)
        plot_reward_scatter(
            y_eval, y_pred_prob,
            title=f"Reward Scatter (Fold {fold_idx+1}) - {test_driver}",
            save_path=f"{fold_prefix}_scatter.png"
        )
        
        # 4. Save User Preference Radar (New Viz)
        plot_user_preference_radar(
            clf.coef_.flatten(),
            save_path=f"{fold_prefix}_radar.png"
        )

    # --- 3. Aggregate Results ---
    mean_auroc = np.mean(fold_auroc)
    std_auroc = np.std(fold_auroc)
    mean_acc = np.mean(fold_acc)
    
    # OOF Score (전체 데이터를 하나로 합쳐서 계산)
    oof_auroc = roc_auc_score(oof_targets, oof_preds)
    
    mean_weights = np.mean(user_weights_list, axis=0)

    print(f"\n[Final Results] Driver {test_driver}")
    print(f"  Mean AUROC: {mean_auroc:.4f} (+/- {std_auroc:.4f})")
    print(f"  OOF AUROC : {oof_auroc:.4f}")
    print(f"  Mean Acc  : {mean_acc:.4f}")
    
    # Visualize Average Preference Radar
    print("  Saving Average User Preference Radar...")
    plot_user_preference_radar(
        mean_weights,
        save_path=log_dir / f"average_preference_radar_{test_driver}.png"
    )

    # Save Metrics JSON
    with open(log_dir / "results.json", "w") as f:
        json.dump({
            "test_driver": test_driver,
            "k_fold": FLAGS["k_fold"],
            "metrics": {
                "mean_auroc": float(mean_auroc),
                "std_auroc": float(std_auroc),
                "oof_auroc": float(oof_auroc),
                "mean_accuracy": float(mean_acc)
            },
            "fold_details": {
                "auroc": fold_auroc,
                "accuracy": fold_acc
            },
            "mean_weights": mean_weights.tolist()
        }, f, indent=4)

    print(f"\nDone. All artifacts saved in: {log_dir}")

if __name__ == "__main__":
    main()