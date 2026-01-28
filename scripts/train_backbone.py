import os
import json
import random
import joblib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 기존 모듈 import 가정
from src.utils.utils import _load_dataset_sequences

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# =========================
# Config
# =========================
@dataclass
class CFG:
    # Data Settings
    features: tuple = ("IMU_VerAccelVal", "Bounce_rate_6D", "Pitch_rate_6D")
    train_driver_names: tuple = ("김진명", "김태근", "조현석", "한규택", "박재일", "이지환")
    test_driver_name: str = "강신길"
    time_range: tuple = (4.5, 7)
    downsample: int = 3
    
    # Model Architecture
    backbone: str = "Autoencoder"  # 'ResNet1D' or 'InceptionTime' or 'Autoencoder'
    feature_dim: int = 64            # Backbone output dimension (Hidden dim)
    
    # Training (Pre-training Ranking Model)
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # DRM / PCA Settings
    pca_components: int = 15         # Number of Global Basis vectors
    whiten_pca: bool = False
    
    # Adaptation Settings
    adapt_k_fold: int = 5
    
    # Logging
    save_root: str = "artifacts/neural_drm"
    seed: int = 42

# =========================
# 1. Models (Backbones)
# =========================

class ResNetBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet1D(nn.Module):
    """
    Simple ResNet-1D for Sensor Data
    Output: (B, feature_dim, T) - Sequence of features
    """
    def __init__(self, in_channels, feature_dim=64):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.layer1 = ResNetBlock1D(32, 32)
        self.layer2 = ResNetBlock1D(32, 64, stride=1) # Keep temporal length
        self.layer3 = ResNetBlock1D(64, feature_dim, stride=1)
        
    def forward(self, x):
        # x: (B, C, T)
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x # (B, F, T)

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[9, 19, 39], bottleneck_channels=32):
        super().__init__()
        self.use_bottleneck = in_channels > 1
        original_in_channels = in_channels
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, 1, bias=False)
            in_channels = bottleneck_channels

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, k, padding=k//2, bias=False)
            for k in kernel_sizes
        ])
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(original_in_channels, out_channels, 1, bias=False)
        )
        self.bn = nn.BatchNorm1d(out_channels * 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        input_mp = x
        if self.use_bottleneck:
            x = self.bottleneck(x)

        outputs = [conv(x) for conv in self.convs]
        outputs.append(self.maxpool_conv(input_mp))

        return self.relu(self.bn(torch.cat(outputs, dim=1)))

class InceptionTime(nn.Module):
    """
    InceptionTime for Sensor Data
    Output: (B, feature_dim, T)
    """
    def __init__(self, in_channels, feature_dim=64, depth=3):
        super().__init__()
        self.blocks = nn.ModuleList()
        current_dim = in_channels
        # InceptionTime usually has fixed filters per layer, e.g. 32
        base_filters = feature_dim // 4 # Since we concat 4 branches
        
        for i in range(depth):
            self.blocks.append(InceptionModule(current_dim, base_filters))
            current_dim = base_filters * 4
            if i % 3 == 2: # Residual connection every 3 blocks usually, but here simple skip
                pass 
        
        self.final_conv = nn.Conv1d(current_dim, feature_dim, 1)
        
    def forward(self, x):
        # x: (B, C, T)
        for block in self.blocks:
            x = block(x)
        x = self.final_conv(x)
        return x # (B, F, T)

class RankingModel(nn.Module):
    """
    End-to-End Ranking Model for Pre-training.
    Assumes Score = Sum( r(s_t) ) = Sum( w * h_t )
    """
    def __init__(self, backbone, feature_dim):
        super().__init__()
        self.backbone = backbone
        # Bias=False is crucial to maintain strict linearity assumption: sum(r) = w * sum(h)
        self.head = nn.Linear(feature_dim, 1, bias=False) 
        
    def forward(self, x):
        # x: (B, C, T)
        # 1. Extract per-step features
        features_seq = self.backbone(x) # (B, F, T)
        
        # 2. Sum Pooling (Trajectory Feature)
        # This enforces the additivity assumption: Phi(tau) = sum(phi(s_t))
        features_traj = features_seq.sum(dim=2) # (B, F)
        
        # 3. Score
        logits = self.head(features_traj) # (B, 1)
        return logits, features_traj

class Autoencoder1D(nn.Module):
    """
    Autoencoder for Unsupervised Feature Learning.
    Encoder extracts features, Decoder reconstructs the input.
    """
    def __init__(self, in_channels, feature_dim=64, seq_len=50): 
        # seq_len: Time dimension length (input T). 
        # Note: If T varies, AdaptiveAvgPool or fixed upsampling is needed. 
        # Here we assume fixed T or handle via ConvTranspose output padding dynamically.
        super().__init__()
        
        # Encoder (Compression)
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2, stride=2),  # T -> T/2
            
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2, stride=2),  # T/2 -> T/4
            
            nn.Conv1d(64, feature_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU()                   # Output: (B, feature_dim, T/4)
        )
        
        # Decoder (Reconstruction)
        # Note: Hardcoded upsampling for simplicity. 
        # For strict symmetry, use ConvTranspose1d with exact params.
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(feature_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2), # T/4 -> T/2
            
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2), # T/2 -> T
            
            nn.Conv1d(32, in_channels, kernel_size=7, stride=1, padding=3),
            # No activation at the end (Linear reconstruction for sensor values)
        )

    def forward(self, x):
        # x: (B, C, T)
        encoded = self.encoder(x)      # Latent features
        decoded = self.decoder(encoded) # Reconstruction
        
        # Handle shape mismatch due to pooling/upsampling (Trim or Pad)
        if decoded.shape[2] != x.shape[2]:
            decoded = F.interpolate(decoded, size=x.shape[2], mode='linear', align_corners=False)
            
        return decoded, encoded

def get_model(cfg: CFG):
    """
    Returns the appropriate model and mode (Supervised/Unsupervised).
    """
    if cfg.backbone == "Autoencoder":
        # Autoencoder is its own model, trained with MSE
        model = Autoencoder1D(len(cfg.features), cfg.feature_dim)
        mode = "unsupervised"
    elif cfg.backbone == "ResNet1D":
        backbone = ResNet1D(len(cfg.features), cfg.feature_dim)
        model = RankingModel(backbone, cfg.feature_dim)
        mode = "supervised"
    elif cfg.backbone == "InceptionTime":
        backbone = InceptionTime(len(cfg.features), cfg.feature_dim)
        model = RankingModel(backbone, cfg.feature_dim)
        mode = "supervised"
    else:
        raise ValueError(f"Unknown backbone: {cfg.backbone}")
    
    return model, mode

# =========================
# Utilities
# =========================
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class SensorDataset(Dataset):
    def __init__(self, X, y):
        # X shape: (N, T, D) -> Permute to (N, D, T) for PyTorch Conv1d
        self.X = torch.FloatTensor(X).permute(0, 2, 1) 
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_backbone(cfg: CFG):
    if cfg.backbone == "ResNet1D":
        return ResNet1D(len(cfg.features), cfg.feature_dim)
    elif cfg.backbone == "InceptionTime":
        return InceptionTime(len(cfg.features), cfg.feature_dim)
    else:
        raise ValueError(f"Unknown backbone: {cfg.backbone}")

# =========================
# Main Pipeline
# =========================
def main():
    cfg = CFG()
    seed_all(cfg.seed)
    
    # Setup Directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(cfg.save_root) / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    
    with open(log_dir / "cfg.json", "w") as f:
        json.dump(cfg.__dict__, f, indent=4, ensure_ascii=False)
    
    print(f"[{timestamp}] Start Neural DRM Pipeline")
    print(f"Backbone: {cfg.backbone}, PCA Dim: {cfg.pca_components}")

    # ---------------------------------------------------------
    # [Step 1] Pre-training Feature Extractor (Ranking Model)
    # ---------------------------------------------------------
    print("\n[Step 1] Loading Train Data & Training Ranking Model...")
    
    all_X_train = []
    all_y_train = []
    
    config = {'features': list(cfg.features)}
    
    # Load all train drivers
    for driver in cfg.train_driver_names:
        X, y = _load_dataset_sequences(driver, cfg.time_range, cfg.downsample, config)
        all_X_train.append(X)
        all_y_train.extend(y)
        
    X_train_np = np.concatenate(all_X_train, axis=0)
    y_train_np = np.array(all_y_train)
    
    # Scale Data (StandardScaler per channel)
    # Flatten -> Scale -> Reshape
    N, T, D = X_train_np.shape
    scaler = StandardScaler()
    X_train_flat = X_train_np.reshape(-1, D)
    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(N, T, D)
    
    # Dataset & Loader
    ds_train = SensorDataset(X_train_scaled, y_train_np)
    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True)



    # Initialize Model & Loss
    model, mode = get_model(cfg)
    model = model.to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    if mode == "unsupervised":
        criterion = nn.MSELoss()  # Reconstruction Loss
        print(" -> Using MSELoss for Reconstruction")
    else:
        criterion = nn.BCEWithLogitsLoss() # Ranking Loss
        print(" -> Using BCEWithLogitsLoss for Ranking")
    
    # Train Loop
    model.train()
    loss_history = []
    
    for epoch in range(cfg.epochs):
        epoch_loss = 0
        for bx, by in dl_train:
            bx = bx.to(cfg.device)
            by = by.to(cfg.device).unsqueeze(1)
            
            optimizer.zero_grad()
            
            if mode == "unsupervised":
                # Autoencoder: Input -> Input (Reconstruction)
                # Output: (decoded, encoded)
                recons, _ = model(bx)
                loss = criterion(recons, bx) # Target is Input (bx)
            else:
                # Ranking: Input -> Label (Classification)
                logits, _ = model(bx)
                loss = criterion(logits, by) # Target is Label (by)
                
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dl_train)
        loss_history.append(avg_loss)
        if (epoch + 1) % 5 == 0:
            print(f" Epoch {epoch+1}/{cfg.epochs} | Loss: {avg_loss:.4f}")

    # Save Pre-trained Encoder
    if mode == "unsupervised":
        # 오토인코더의 경우 encoder 부분만 저장 (나중에 feature extraction 용도)
        torch.save(model.encoder.state_dict(), log_dir / "backbone.pth")
        # 전체 모델 저장 (복원 확인용, 선택사항)
        torch.save(model.state_dict(), log_dir / "autoencoder_full.pth")
    else:
        # Ranking Model의 경우 backbone 저장
        torch.save(model.backbone.state_dict(), log_dir / "backbone.pth")

    joblib.dump(scaler, log_dir / "scaler.pkl")
    joblib.dump(scaler, log_dir / "scaler.pkl")
    
    plt.plot(loss_history)
    plt.title("Ranking Model Training Loss")
    plt.savefig(log_dir / "train_loss.png")
    plt.close()

    # Autoencoder Reconstruction Visualization
    if mode == "unsupervised":
        print("\n[Step 1.5] Visualizing Autoencoder Reconstruction...")
        model.eval()

        idx_good = np.where(y_train_np == 1)[0][:3]
        idx_bad = np.where(y_train_np == 0)[0][:3]

        with torch.no_grad():
            for label, indices in [("Good", idx_good), ("Bad", idx_bad)]:
                for i, idx in enumerate(indices):
                    x = ds_train.X[idx:idx+1].to(cfg.device)
                    recon, _ = model(x)

                    n_channels = len(cfg.features)
                    fig, axes = plt.subplots(n_channels, 1, figsize=(10, 3 * n_channels))
                    if n_channels == 1:
                        axes = [axes]

                    for ch, (ax, feat_name) in enumerate(zip(axes, cfg.features)):
                        ax.plot(x[0, ch].cpu().numpy(), label='Original', alpha=0.7)
                        ax.plot(recon[0, ch].cpu().numpy(), label='Reconstructed', alpha=0.7, linestyle='--')
                        ax.set_title(feat_name)
                        ax.legend()
                        ax.grid(True, alpha=0.3)

                    mse = F.mse_loss(recon, x).item()
                    fig.suptitle(f"{label} Sample {i+1} (MSE: {mse:.4f})")
                    plt.tight_layout()
                    plt.savefig(log_dir / f"recon_{label}_{i}.png")
                    plt.close()

        print(f" Saved 6 reconstruction plots to {log_dir}")

    # ---------------------------------------------------------
    # [Step 2] Basis Extraction via PCA on Difference Vectors
    # ---------------------------------------------------------
    print("\n[Step 2] Extracting Global Preference Basis (Neural DRM)...")
    model.eval()
    phi_list = []
    dl_eval = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=False)
    with torch.no_grad():
        for bx, _ in dl_eval:
            bx = bx.to(cfg.device)
            
            if mode == "unsupervised":
                # Autoencoder: returns (decoded, features_seq)
                # features_seq shape: (Batch, Feature, Time)
                _, features_seq = model(bx)
                
                # Ranking Model과 조건을 맞추기 위해 강제로 Sum Pooling 수행
                features_traj = features_seq.sum(dim=2) 
                
            else:
                # RankingModel: returns (logits, features_traj)
                # features_traj is already sum-pooled inside the model
                _, features_traj = model(bx)
            phi_list.append(features_traj.cpu().numpy())
            
    Phi_train = np.concatenate(phi_list, axis=0) # (N, Feature_Dim)
    
    # 2. Generate Difference Vectors (Good - Bad)
    # Randomly pair Good and Bad samples to create difference directions
    idx_good = np.where(y_train_np == 1)[0]
    idx_bad = np.where(y_train_np == 0)[0]
    
    n_pairs = max(len(idx_good), len(idx_bad)) * 2 # Generate enough pairs
    
    # Sample with replacement
    pairs_good = np.random.choice(idx_good, n_pairs)
    pairs_bad = np.random.choice(idx_bad, n_pairs)
    
    Z_diff = Phi_train[pairs_good] - Phi_train[pairs_bad] # (n_pairs, F)
    
    print(f" Constructed {Z_diff.shape[0]} difference vectors.")
    
    # 3. Fit PCA
    pca = PCA(n_components=cfg.pca_components, random_state=cfg.seed, whiten=cfg.whiten_pca)
    pca.fit(Z_diff)
    
    # Basis Matrix V (Components)
    # pca.components_ shape is (n_components, feature_dim)
    V_basis = pca.components_ 
    
    print(f" PCA Explained Variance: {np.sum(pca.explained_variance_ratio_):.4f}")
    joblib.dump(pca, log_dir / "pca_basis.pkl")

    # ---------------------------------------------------------
    # [Step 3] Adaptation to Test Driver
    # ---------------------------------------------------------
    print(f"\n[Step 3] Adapting to Test Driver: {cfg.test_driver_name}")
    
    # Load Test Data
    X_test, y_test = _load_dataset_sequences(cfg.test_driver_name, cfg.time_range, cfg.downsample, config)
    y_test = y_test.astype(int)
    
    # Scale Test Data
    N_test, T_test, D_test = X_test.shape
    X_test_scaled = scaler.transform(X_test.reshape(-1, D)).reshape(N_test, T_test, D)
    
    # Extract Features using Frozen Backbone
    ds_test = SensorDataset(X_test_scaled, y_test)
    dl_test = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False)
    
    phi_test_list = []
    with torch.no_grad():
        for bx, _ in dl_test:
            bx = bx.to(cfg.device)
            
            # [수정된 부분] Step 2와 동일한 로직 적용
            if mode == "unsupervised":
                # Autoencoder: returns (decoded, features_seq)
                _, features_seq = model(bx)
                features_traj = features_seq.sum(dim=2) # (Batch, Feature)로 변환
            else:
                # RankingModel: returns (logits, features_traj)
                _, features_traj = model(bx)
            phi_test_list.append(features_traj.cpu().numpy())
    Phi_test = np.concatenate(phi_test_list, axis=0) # (N_test, Feature_Dim)
    
    # Project onto PCA Basis (Dimensionality Reduction & Basis Alignment)
    # Projected Feature = Phi_test * V.T
    X_final = Phi_test @ V_basis.T # (N_test, K)
    
    print(f" Test Features Shape: {X_final.shape}")
    
    # K-Fold Adaptation & Evaluation
    skf = StratifiedKFold(n_splits=cfg.adapt_k_fold, shuffle=True, random_state=cfg.seed)
    
    fold_auroc = []
    fold_acc = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_final, y_test)):
        X_adapt, y_adapt = X_final[train_idx], y_test[train_idx]
        X_eval, y_eval = X_final[val_idx], y_test[val_idx]
        
        # Logistic Regression (Learn user weights w)
        clf = LogisticRegression(class_weight='balanced', solver='liblinear')
        clf.fit(X_adapt, y_adapt)
        
        # Predict
        y_pred_prob = clf.predict_proba(X_eval)[:, 1]
        y_pred = clf.predict(X_eval)
        
        auroc = roc_auc_score(y_eval, y_pred_prob)
        acc = accuracy_score(y_eval, y_pred)
        
        fold_auroc.append(auroc)
        fold_acc.append(acc)
        print(f" Fold {fold_idx+1}/{cfg.adapt_k_fold} | AUROC: {auroc:.4f}, Acc: {acc:.4f}")
        
        # Save ROC curve for first fold
        if fold_idx == 0:
            fpr, tpr, _ = roc_curve(y_eval, y_pred_prob)
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC={auroc:.3f}")
            plt.plot([0, 1], [0, 1], 'k--')
            plt.title(f"ROC Fold 0 - {cfg.test_driver_name}")
            plt.legend()
            plt.savefig(log_dir / f"roc_fold0.png")
            plt.close()
            
            # Save User Weights (Preference Profile)
            user_weights = clf.coef_.flatten()
            plt.figure(figsize=(8, 4))
            plt.bar(range(len(user_weights)), user_weights)
            plt.title("User Preference Weights on Neural Basis")
            plt.xlabel("Basis Index")
            plt.ylabel("Weight")
            plt.grid(True, alpha=0.3)
            plt.savefig(log_dir / "user_weights.png")
            plt.close()

    print("\n[Final Results]")
    print(f" Mean AUROC: {np.mean(fold_auroc):.4f} (+/- {np.std(fold_auroc):.4f})")
    print(f" Mean Acc:   {np.mean(fold_acc):.4f}")
    
    # Save Results
    results = {
        "test_driver": cfg.test_driver_name,
        "backbone": cfg.backbone,
        "pca_dim": cfg.pca_components,
        "mean_auroc": float(np.mean(fold_auroc)),
        "std_auroc": float(np.std(fold_auroc)),
        "fold_aurocs": fold_auroc
    }
    
    with open(log_dir / "results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Done. Artifacts saved in {log_dir}")

if __name__ == "__main__":
    main()