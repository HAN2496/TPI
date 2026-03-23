import os
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.manifold import TSNE

from src.utils.utils import _load_dataset_sequences
from src.model.tsfresh.pipeline import TSFreshPipeline
from src.model.tsfresh.visualization import (
    plot_latent_space, 
    plot_roc_curve, 
    plot_reward_distribution, 
    plot_reward_scatter,
    plot_kfold_roc_curves
)

# --- Configuration ---
FLAGS = {
    # System
    "timestamp": "best/20260126_174219",  # None (Train new), "test" (Train in 'test' dir), or "YYYYMMDD_HHMMSS" (Inference only)
    "save_root": "artifacts/tsfresh",
    "random_state": 42,

    # Data
    "features": ["IMU_LongAccelVal", "Bounce_rate_6D", "Pitch_rate_6D"],
    "train_driver_names": ["김진명", "김태근", "조현석", "한규택", "박재일", "이지환"],
    "test_driver_names": ["강신길"],
    "time_range": (5, 7.5),
    "downsample": 5,

    # Pipeline Settings
    "pca_components": 2,
    "tsfresh_settings": "efficient", 
    
    # Visualization (t-SNE)
    "tsne_perplexity": 30,
    
    # Evaluation
    "k_folds": 5,          # if > 1, use StratifiedKFold
    "test_split_size": 0.2 # Only used if k_folds <= 1
}

def main():
    # --- 1. Experiment Setup ---
    if FLAGS["timestamp"] == "test":
        timestamp = "test"
        log_dir = Path(FLAGS["save_root"]) / timestamp
        is_training = True
    elif FLAGS["timestamp"] is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(FLAGS["save_root"]) / timestamp
        is_training = True
    else:
        # Inference Mode
        timestamp = FLAGS["timestamp"]
        log_dir = Path(FLAGS["save_root"]) / timestamp
        is_training = False
        print(f"Inference Mode: Loading experiment from {log_dir}")
        
        if not log_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {log_dir}")

    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Experiment Dir: {log_dir}")
    
    pipeline_path = log_dir / "pipeline.pkl"
    
    # Save/Load Flags
    flags_path = log_dir / "flags.json"
    if is_training:
        with open(flags_path, "w", encoding='utf-8') as f:
            json.dump(FLAGS, f, indent=4, ensure_ascii=False)
    else:
        if flags_path.exists():
            with open(flags_path, "r", encoding='utf-8') as f:
                loaded_flags = json.load(f)
                print("Loaded flags from training experiment.")
    
    config = {'features': FLAGS["features"]}

    # --- 2. Train / Load Pipeline ---
    if is_training:
        print("\n[Training Mode] Loading Train Data...")
        all_X = []
        all_driver_labels = [] 
        
        for driver in FLAGS["train_driver_names"]:
            print(f"  Loading {driver}...")
            X, y = _load_dataset_sequences(driver, FLAGS["time_range"], FLAGS["downsample"], config)
            all_X.append(X)
            all_driver_labels.extend([driver] * len(X))
            
        X_train_all = np.concatenate(all_X, axis=0)
        print(f"Total Train Data Shape: {X_train_all.shape}")
        
        # Fit Pipeline
        print("\nFitting TSFresh Pipeline (This may take a while)...")
        pipeline = TSFreshPipeline(
            feature_names=FLAGS["features"], 
            n_components=FLAGS["pca_components"], 
            extraction_settings=FLAGS["tsfresh_settings"],
            verbose=1
        )
        
        pipeline.fit(X_train_all)
        pipeline.save(pipeline_path)
        
        n_kept = pipeline.pca.n_components_
        print(f"Pipeline saved. PCA kept {n_kept} components.")
        
        # Visualize Train Space
        print("\nVisualizing Train Latent Space (t-SNE)...")
        Z_train_pca = pipeline.transform(X_train_all)
        
        tsne = TSNE(n_components=2, perplexity=FLAGS["tsne_perplexity"], random_state=FLAGS["random_state"])
        Z_train_tsne = tsne.fit_transform(Z_train_pca)
        
        plot_latent_space(
            Z_train_tsne, 
            all_driver_labels, 
            title=f"Train Drivers (t-SNE on {n_kept}D PCA space)",
            save_path=log_dir / "latent_space_train_tsne.png"
        )
        
    else:
        print(f"\n[Inference Mode] Loading pipeline from {pipeline_path}...")
        if not pipeline_path.exists():
            raise FileNotFoundError(f"Pipeline file not found: {pipeline_path}")
            
        pipeline = TSFreshPipeline.load(pipeline_path)
        n_kept = pipeline.pca.n_components_
        print(f"Pipeline loaded. PCA components: {n_kept}")


    # --- 3. Evaluate on Test Driver (Adaptation) ---
    print("\nEvaluating on Test Driver...")
    test_driver = FLAGS["test_driver_names"][0]
    
    print(f"  Loading {test_driver}...")
    X_test_raw, y_test_raw = _load_dataset_sequences(test_driver, FLAGS["time_range"], FLAGS["downsample"], config)
    y_test_raw = y_test_raw.astype(int)
    
    print("  Transforming Test Data (PCA)...")
    Z_test_pca = pipeline.transform(X_test_raw)
    
    # --- Visualization Suite (Shared) ---
    print("  Running t-SNE for Test Driver visualization...")
    tsne_test = TSNE(n_components=2, perplexity=min(FLAGS["tsne_perplexity"], len(Z_test_pca)-1), random_state=FLAGS["random_state"])
    Z_test_tsne = tsne_test.fit_transform(Z_test_pca)
    
    plot_latent_space(
        Z_test_tsne,
        y_test_raw, 
        title=f"Test Driver {test_driver} (t-SNE)",
        save_path=log_dir / "latent_space_test_tsne.png"
    )

    # --- K-Fold vs Single Split Logic ---
    n_splits = FLAGS.get("k_folds", 1)
    
    if n_splits > 1:
        print(f"\nStarting {n_splits}-Fold Cross Validation...")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=FLAGS["random_state"])
        
        fold_results = {'auroc': [], 'acc': []}
        fold_predictions = [] # List of (y_true, y_prob)
        
        print(f"{ 'Fold':<6} { 'Train':<7} { 'Test':<6} { 'AUROC':<10} { 'Acc':<10}")
        print("-" * 50)
        
        for i, (train_idx, test_idx) in enumerate(skf.split(Z_test_pca, y_test_raw)):
            Z_train, Z_test = Z_test_pca[train_idx], Z_test_pca[test_idx]
            y_train, y_test = y_test_raw[train_idx], y_test_raw[test_idx]
            
            clf = LogisticRegression(class_weight='balanced', random_state=FLAGS["random_state"], max_iter=1000)
            clf.fit(Z_train, y_train)
            
            y_prob = clf.predict_proba(Z_test)[:, 1]
            y_pred = clf.predict(Z_test)
            
            auroc = roc_auc_score(y_test, y_prob)
            acc = accuracy_score(y_test, y_pred)
            
            fold_results['auroc'].append(auroc)
            fold_results['acc'].append(acc)
            fold_predictions.append((y_test, y_prob))
            
            print(f"{i:<6} {len(train_idx):<7} {len(test_idx):<6} {auroc:<10.4f} {acc:<10.4f}")
            
        mean_auroc = np.mean(fold_results['auroc'])
        mean_acc = np.mean(fold_results['acc'])
        
        # OOF AUC
        y_true_all = np.concatenate([y for y, _ in fold_predictions])
        y_prob_all = np.concatenate([p for _, p in fold_predictions])
        oof_auc = roc_auc_score(y_true_all, y_prob_all)
        
        print("-" * 50)
        print(f"{ 'Mean':<6} { '':<7} { '':<6} {mean_auroc:<10.4f} {mean_acc:<10.4f}")
        print(f"{ 'OOF':<6} { '':<7} { '':<6} {oof_auc:<10.4f}")
        
        # Plot K-Fold ROC
        plot_kfold_roc_curves(
            fold_predictions, 
            title=f"K-Fold ROC Curves - {test_driver}",
            save_path=log_dir / "kfold_roc_curves.png"
        )
        
        # Save Summary
        with open(log_dir / "kfold_results.json", "w") as f:
            json.dump({
                "test_driver": test_driver,
                "n_splits": n_splits,
                "fold_results": fold_results,
                "mean_auroc": mean_auroc,
                "mean_acc": mean_acc,
                "oof_auc": oof_auc,
                "pca_components_kept": int(n_kept)
            }, f, indent=4)

    else:
        # Single Split (Legacy Mode)
        print(f"\nSingle Split Evaluation (Test Ratio: {FLAGS['test_split_size']})...")
        Z_adapt, Z_eval, y_adapt, y_eval = train_test_split(
            Z_test_pca, y_test_raw, 
            test_size=FLAGS["test_split_size"], 
            random_state=FLAGS["random_state"],
            stratify=y_test_raw
        )
        
        clf = LogisticRegression(class_weight='balanced', random_state=FLAGS["random_state"], max_iter=1000)
        clf.fit(Z_adapt, y_adapt)
        
        y_pred_prob = clf.predict_proba(Z_eval)[:, 1]
        y_pred_reward = clf.decision_function(Z_eval)
        y_pred = clf.predict(Z_eval)
        
        auroc = roc_auc_score(y_eval, y_pred_prob)
        acc = accuracy_score(y_eval, y_pred)
        
        print(f"\n[Result] Test Driver {test_driver}")
        print(f"  AUROC: {auroc:.4f}")
        print(f"  Accuracy: {acc:.4f}")
        
        plot_roc_curve(
            y_eval, y_pred_prob,
            title=f"ROC Curve - {test_driver}",
            save_path=log_dir / "roc_curve.png"
        )
        
        plot_reward_distribution(
            y_pred_reward, y_eval,
            title=f"Estimated Reward Distribution (Logits) - {test_driver}",
            save_path=log_dir / "reward_dist.png"
        )
        
        plot_reward_scatter(
            y_eval, y_pred_prob,
            title=f"Prediction Scatter (P(Good)) - {test_driver}",
            save_path=log_dir / "scatter_pred.png"
        )
        
        with open(log_dir / "results.json", "w") as f:
            json.dump({
                "test_driver": test_driver,
                "auroc": auroc,
                "accuracy": acc,
                "weights": clf.coef_.flatten().tolist(),
                "bias": float(clf.intercept_[0])
            }, f, indent=4)
            
    print("\nDone. Check artifacts at:", log_dir)

if __name__ == "__main__":
    main()