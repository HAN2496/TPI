import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def plot_latent_space(Z, labels, title="Latent Space", save_path=None):
    """
    Z: (N, 2) numpy array
    labels: array-like of shape (N,)
    """
    plt.figure(figsize=(10, 8))
    
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    
    for lbl in unique_labels:
        mask = (labels == lbl)
        plt.scatter(Z[mask, 0], Z[mask, 1], alpha=0.7, label=str(lbl))
        
    plt.title(title)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_coefficients(coefs, feature_names, title="Coefficients", save_path=None):
    indices = np.arange(len(coefs))
    
    plt.figure(figsize=(10, 6))
    plt.bar(indices, coefs)
    plt.xticks(indices, feature_names, rotation=45)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_roc_curve(y_true, y_score, title="ROC Curve", save_path=None):
    """
    y_true: true binary labels
    y_score: target scores (probability or decision function)
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_reward_distribution(rewards, labels, title="Reward Distribution", save_path=None):
    """
    rewards: predicted reward values (logits)
    labels: true labels (0 or 1)
    """
    rewards = np.array(rewards)
    labels = np.array(labels)
    
    plt.figure(figsize=(10, 6))
    
    # Plot histograms for Good(1) and Bad(0)
    plt.hist(rewards[labels==0], bins=30, alpha=0.5, label='Bad (0)', color='blue', density=True)
    plt.hist(rewards[labels==1], bins=30, alpha=0.5, label='Good (1)', color='red', density=True)
    
    plt.title(title)
    plt.xlabel("Estimated Reward (Logit: w^T z + b)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_reward_scatter(y_true, y_prob, title="Reward Prediction Scatter", save_path=None):
    """
    Scatter plot of predicted probabilities (or rewards).
    y_true: true labels
    y_prob: predicted probability (0~1)
    """
    idx = np.arange(len(y_true))
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    m0 = (y_true == 0)
    m1 = (y_true == 1)

    plt.figure(figsize=(10, 5))
    plt.scatter(idx[m0], y_prob[m0], s=15, alpha=0.6, label="Bad(0)", color='blue')
    plt.scatter(idx[m1], y_prob[m1], s=15, alpha=0.6, label="Good(1)", color='red')
    plt.axhline(0.5, linestyle="--", alpha=0.5, color='gray')
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Sample index")
    plt.ylabel("Predicted Probability P(Good|z)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_kfold_roc_curves(fold_predictions, title="K-Fold ROC Curves", save_path=None):
    """
    fold_predictions: list of tuples (y_true, y_prob) for each fold
    """
    plt.figure(figsize=(8, 6))
    
    # 1. Plot each fold
    aucs = []
    for i, (y_true, y_prob) in enumerate(fold_predictions):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        aucs.append(auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.6, label=f'Fold {i} (AUC = {auc:.3f})')
        
    # 2. Plot OOF (Mean)
    y_true_all = np.concatenate([y for y, _ in fold_predictions])
    y_prob_all = np.concatenate([p for _, p in fold_predictions])
    
    fpr_oof, tpr_oof, _ = roc_curve(y_true_all, y_prob_all)
    auc_oof = roc_auc_score(y_true_all, y_prob_all)
    
    plt.plot(fpr_oof, tpr_oof, color='b', label=f'OOF (AUC = {auc_oof:.3f})', lw=2, alpha=0.8)
    
    # 3. Random guess line
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{title}\nMean AUC = {np.mean(aucs):.3f}")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_feature_contribution(pipeline, feature_names, n_top=10, save_path=None):
    """
    [Viz 1] PC1, PC2 등을 구성하는 상위 TSFresh Feature 확인
    """
    pca = pipeline.pca
    components = pca.components_  # shape: (n_components, n_features)
    
    # PCA 단계 이전에 VarianceThreshold로 선택된 Feature 이름 가져오기
    # pipeline.selected_columns_ 가 fit() 이후에 저장되어 있다고 가정
    selected_feats = pipeline.selected_columns_
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for i in range(min(2, pca.n_components_)): # PC1, PC2만 시각화
        # 각 Feature의 절대값 가중치 정렬
        comp_weights = components[i]
        sorted_idx = np.argsort(np.abs(comp_weights))[::-1][:n_top]
        
        top_feats = selected_feats[sorted_idx]
        top_weights = comp_weights[sorted_idx]
        
        # Plot
        axes[i].barh(top_feats, top_weights, align='center')
        axes[i].set_title(f"Principal Component {i+1} (Explained Var: {pca.explained_variance_ratio_[i]:.2%})")
        axes[i].invert_yaxis()  # 상위 항목이 위로 오게
        axes[i].set_xlabel("Weight Contribution")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_user_preference_radar(user_weights, save_path=None):
    """
    [Viz 2] 사용자가 각 PC(Basis)에 대해 가지는 가중치(성향) Radar Chart
    """
    n_components = len(user_weights)
    
    # 각 축의 라벨 (PC1, PC2 ...)
    labels = [f"PC{i+1}" for i in range(n_components)]
    
    # Radar Chart를 위해 각도 계산
    angles = np.linspace(0, 2 * np.pi, n_components, endpoint=False).tolist()
    
    # 닫힌 도형을 만들기 위해 첫 번째 값을 끝에 추가
    values = np.concatenate((user_weights, [user_weights[0]]))
    angles += [angles[0]]
    labels += [labels[0]]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    # Plot
    ax.plot(angles, values, color='red', linewidth=2, linestyle='solid', label='User Weights')
    ax.fill(angles, values, color='red', alpha=0.25)
    
    # Labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels[:-1])
    
    ax.set_title("User Preference Profile (Sensitivity to Basis)", y=1.1)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()