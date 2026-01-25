import torch
import numpy as np
import os
import argparse
from tqdm import tqdm
from captum.attr import IntegratedGradients

from src.configs.utils import create_model, load_config
from src.configs.registries import MODELS
from src.utils import ExperimentPaths, convert_driver_name, prepare_training_data_kfold
from src.xai.visualization import visualize_captum, plot_feature_importance, plot_temporal_importance

def explain_fold(driver_name, model_type, model_name, time_range, downsample, n_splits, test_ratio, device, fold_idx, test_loader, config, output_dir):
    print(f"\nExplaining Fold {fold_idx}...")
    
    # 1. Load Model
    paths = ExperimentPaths(driver_name, model_type, model_name, time_range, tag=f"kfold/fold_{fold_idx}", tag_as_subdir=True)
    
    # Instantiate model manually to use updated config (with input_dim)
    model_cls = MODELS[config['model_type']]
    model = model_cls(**config['args'])
    model = model.to(device)
    
    model_path = paths.best_model

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 2. Prepare Data (Test Set of this fold)
    X_test, y_test = test_loader.dataset.tensors
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    # 3. Integrated Gradients
    ig = IntegratedGradients(model)
    
    # Compute attributions for all test samples
    # We process in batches to avoid OOM
    batch_size = 32
    attributions_list = []
    
    print("  Computing attributions...")
    for i in tqdm(range(0, len(X_test), batch_size)):
        batch_x = X_test[i:i+batch_size]
        # Target: For binary classification (sigmoid), output is scalar (prob of class 1).
        # We don't need target index if output is 1D.
        # If output is (B, 1), we might need target=None or squeeze.
        
        # Check model output shape
        with torch.no_grad():
            out = model(batch_x)
        
        # If output is (B, 1), target=None is fine for simple scalar regression/binary.
        # Captum handles scalar output automatically.
        
        baseline = torch.zeros_like(batch_x)
        
        attr = ig.attribute(batch_x, baselines=baseline, n_steps=50)
        attributions_list.append(attr.detach().cpu().numpy())
        
    attributions = np.concatenate(attributions_list, axis=0) # (N, T, F)
    
    # --- Debugging ---
    print(f"  [Debug] Attributions Stats:")
    print(f"    Shape: {attributions.shape}")
    print(f"    Range: [{attributions.min():.6f}, {attributions.max():.6f}]")
    print(f"    Mean: {attributions.mean():.6f}, Std: {attributions.std():.6f}")
    print(f"    Non-zero: {np.count_nonzero(attributions)} / {attributions.size} ({100*np.count_nonzero(attributions)/attributions.size:.2f}%)")
    
    X_test_np = X_test.cpu().numpy()
    print(f"  [Debug] Input Data Stats:")
    print(f"    Range: [{X_test_np.min():.6f}, {X_test_np.max():.6f}]")
    print(f"    Mean: {X_test_np.mean():.6f}")
    # -----------------
    
    # 4. Global Importance (Mean Absolute Attribution)
    # Feature Importance
    global_feat_imp = np.mean(np.abs(attributions), axis=(0, 1)) # (F,)
    plot_feature_importance(
        global_feat_imp, 
        config['features'], 
        save_path=os.path.join(output_dir, f"fold_{fold_idx}_global_feature_importance.png")
    )
    
    # Temporal Importance
    global_time_imp = np.mean(np.abs(attributions), axis=(0, 2)) # (T,)
    plot_temporal_importance(
        global_time_imp, 
        save_path=os.path.join(output_dir, f"fold_{fold_idx}_global_temporal_importance.png")
    )
    
    # 5. Local Explanations (Samples)
    # Select a few True Positive, True Negative, False Positive, False Negative
    with torch.no_grad():
        preds = torch.sigmoid(model(X_test)).cpu().numpy().flatten()
        
    y_true = y_test.cpu().numpy().flatten()
    preds_binary = (preds > 0.5).astype(int)
    
    indices = {
        'TP': np.where((y_true == 1) & (preds_binary == 1))[0],
        'TN': np.where((y_true == 0) & (preds_binary == 0))[0],
        'FP': np.where((y_true == 0) & (preds_binary == 1))[0],
        'FN': np.where((y_true == 1) & (preds_binary == 0))[0]
    }
    
    for case, idxs in indices.items():
        if len(idxs) == 0: continue
        
        # Pick top confidence sample for TP/TN, or worst error for FP/FN
        if case in ['TP', 'TN']:
            # Most confident
            selected_idx = idxs[np.argmax(np.abs(preds[idxs] - (1 if case=='TP' else 0)))] # Should be min distance to target... wait
            # TP: want pred close to 1. TN: want pred close to 0.
            # actually argmax(preds) for TP, argmin(preds) for TN is "most confident"
            if case == 'TP': selected_idx = idxs[np.argmax(preds[idxs])]
            else: selected_idx = idxs[np.argmin(preds[idxs])]
        else:
            # Most wrong (High confidence error)
            if case == 'FP': selected_idx = idxs[np.argmax(preds[idxs])] # pred close to 1, but true is 0
            else: selected_idx = idxs[np.argmin(preds[idxs])] # pred close to 0, but true is 1
            
        # Visualize
        attr = attributions[selected_idx]
        inp = X_test[selected_idx].cpu().numpy()
        
        visualize_captum(
            attr, inp, 
            feature_names=config['features'],
            title=f"Fold {fold_idx} - {case} (idx={selected_idx}, Prob={preds[selected_idx]:.4f})",
            save_path=os.path.join(output_dir, f"fold_{fold_idx}_local_{case}_{selected_idx}.png")
        )

def main():
    parser = argparse.ArgumentParser(description="Explain K-Fold Models using Captum")
    parser.add_argument('-d', '--driver-name', required=True, help='Driver name')
    parser.add_argument('-mt', '--model-type', required=True, help='Model type')
    parser.add_argument('-mn', '--model-name', default='base', help='Model name')
    parser.add_argument('-t', '--time-range', type=str, default="[5, 7]", help='Time range as list [start, end]')
    parser.add_argument('--downsample', type=int, default=5, help='Downsample rate')
    parser.add_argument('--n-splits', type=int, default=5, help='Number of K-Fold splits')
    parser.add_argument('--test-ratio', type=float, default=0.1, help='Test set ratio')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    
    args = parser.parse_args()
    
    # Parse time range safely
    import ast
    time_range = ast.literal_eval(args.time_range)
    
    driver_name = convert_driver_name(args.driver_name)
    config = load_config(driver_name, args.model_type, args.model_name)
    
    # Re-create data splits to match training
    # Note: random_state=42 MUST match training to get same folds
    print("Preparing data...")
    test_loader, folds = prepare_training_data_kfold(
        driver_name, config, time_range, args.downsample, args.n_splits, args.test_ratio, 
        is_neural=True, random_state=42
    )
    
    output_dir = f"artifacts/analysis/explain_kfold/{driver_name}_{args.model_type}_{args.model_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # We can explain the 'Test Set' using the best model from a specific fold,
    # OR we can explain the 'Validation Set' of each fold.
    # Usually, we want to see how the model behaves on unseen data.
    # The 'test_loader' returned by prepare_training_data_kfold is the hold-out test set (if test_ratio > 0).
    # Let's explain the hold-out TEST SET using the Best Fold Model (or all fold models).
    
    # Let's do it for all folds to see consistency.
    for fold_idx, _, _ in folds:
        explain_fold(
            driver_name, args.model_type, args.model_name, time_range, 
            args.downsample, args.n_splits, args.test_ratio, args.device,
            fold_idx, test_loader, config, output_dir
        )
        
    print(f"\nAll explanations saved to {output_dir}")

if __name__ == "__main__":
    main()
