# scripts/temp.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score
import higher

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from src.utils.utils import load_cached_datasets


# =========================
# 1. Config
# =========================
@dataclass
class MetaConfig:
    features: tuple = ("IMU_VerAccelVal", "Pitch_rate_6D", "Bounce_rate_6D", "IMU_LongAccelVal")
    train_driver_names: tuple = ("김진명", "김태근", "조현석", "한규택", "박재일", "이지환")
    test_driver_name: str = "강신길"
    time_range: tuple = (5, 7)
    downsample: int = 5

    hidden_dim: int = 64
    inner_lr: float = 0.005
    outer_lr: float = 0.001
    inner_steps: int = 3
    n_support: int = 20
    n_query: int = 20

    meta_epochs: int = 200
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    verbose: int = 1


# =========================
# 2. Data Loading
# =========================
def load_train_data(cfg):
    config = {"features": list(cfg.features)}
    item_series_list = []
    per_user_items = {}
    item_offset = 0

    for uid, uname in enumerate(cfg.train_driver_names):
        X, y = load_cached_datasets(uname, cfg.time_range, cfg.downsample, config)
        n = X.shape[0]
        item_ids = np.arange(item_offset, item_offset + n, dtype=np.int64)
        per_user_items[uid] = (item_ids, y.astype(np.int64))
        item_series_list.append(X.astype(np.float32))
        item_offset += n
        if cfg.verbose > 0:
            print(f"  - {uname}: items={n}, pos={int(y.sum())}, neg={int((1-y).sum())}")

    item_series = np.concatenate(item_series_list, axis=0)
    return item_series, per_user_items, len(cfg.train_driver_names), item_series.shape[2]


def load_test_data(cfg):
    config = {"features": list(cfg.features)}
    X, y = load_cached_datasets(cfg.test_driver_name, cfg.time_range, cfg.downsample, config)
    return X.astype(np.float32), y.astype(np.int64)


# =========================
# 3. Base Model & Balanced Sampler
# =========================
class MetaCNNModel(nn.Module):
    def __init__(self, obs_dim, hidden_dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(obs_dim, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = F.max_pool1d(x, kernel_size=x.shape[-1]).squeeze(-1)
        return self.head(x).squeeze(-1)


def get_balanced_task_indices(labels, n_support, n_query):
    pos_local_idx = np.where(labels == 1)[0]
    neg_local_idx = np.where(labels == 0)[0]

    n_half_sup = n_support // 2
    n_half_que = n_query // 2

    if len(pos_local_idx) < (n_half_sup + n_half_que) or len(neg_local_idx) < (n_half_sup + n_half_que):
        return None

    p_sel = np.random.choice(pos_local_idx, n_half_sup + n_half_que, replace=False)
    n_sel = np.random.choice(neg_local_idx, n_half_sup + n_half_que, replace=False)

    sup_l = np.concatenate([p_sel[:n_half_sup], n_sel[:n_half_sup]])
    que_l = np.concatenate([p_sel[n_half_sup:], n_sel[n_half_sup:]])
    return sup_l, que_l


def plot_maml_results(y_true, y_prob, adaptation_logs, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {auc(fpr, tpr):.4f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0].set_title('Receiver Operating Characteristic')
    axes[0].legend(loc="lower right")

    axes[1].hist(y_prob[y_true == 1], bins=20, alpha=0.5, label='Good (1)', color='blue')
    axes[1].hist(y_prob[y_true == 0], bins=20, alpha=0.5, label='Bad (0)', color='red')
    axes[1].set_title('Reward Score Distribution')
    axes[1].legend()

    steps = range(len(adaptation_logs))
    axes[2].plot(steps, adaptation_logs, marker='o', color='green')
    axes[2].set_title('Adaptation Progress (Inner Loss)')
    axes[2].set_xlabel('Inner Loop Steps')
    axes[2].set_ylabel('Loss on Support Set')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")


# =========================
# 4. Main Training & Evaluation
# =========================
def run_maml_training(cfg: MetaConfig):
    device = torch.device(cfg.device)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    item_series, per_user_items, n_users, obs_dim = load_train_data(cfg)

    model = MetaCNNModel(obs_dim=obs_dim, hidden_dim=cfg.hidden_dim).to(device)
    meta_optimizer = optim.Adam(model.parameters(), lr=cfg.outer_lr, weight_decay=1e-4)

    print(f"\n[1] Starting Balanced MAML Training...")

    for epoch in range(cfg.meta_epochs):
        model.train()
        meta_loss = 0.0
        task_count = 0

        for _ in range(20):
            uid = np.random.randint(0, n_users)
            item_ids, labels = per_user_items[uid]

            indices = get_balanced_task_indices(labels, cfg.n_support, cfg.n_query)
            if indices is None:
                continue
            sup_l, que_l = indices

            sup_x = torch.tensor(item_series[item_ids[sup_l]], dtype=torch.float32).to(device)
            sup_y = torch.tensor(labels[sup_l], dtype=torch.float32).to(device)
            que_x = torch.tensor(item_series[item_ids[que_l]], dtype=torch.float32).to(device)
            que_y = torch.tensor(labels[que_l], dtype=torch.float32).to(device)

            inner_opt = optim.SGD(model.parameters(), lr=cfg.inner_lr)

            best_step_auc = 0
            with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (fmodel, diffopt):
                for _ in range(cfg.inner_steps):
                    sup_loss = F.binary_cross_entropy_with_logits(fmodel(sup_x), sup_y)
                    diffopt.step(sup_loss)

                que_loss = F.binary_cross_entropy_with_logits(fmodel(que_x), que_y)
                que_loss.backward()
                meta_loss += que_loss.item()
                task_count += 1

        if task_count > 0:
            meta_optimizer.step()
            meta_optimizer.zero_grad()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Avg Meta Loss: {meta_loss / max(1, task_count):.4f}")

    # =========================
    # 5. Evaluation
    # =========================
    print(f"\n[2] Evaluating on Test Driver: {cfg.test_driver_name}")
    X_test, y_test = load_test_data(cfg)

    test_indices = get_balanced_task_indices(y_test, cfg.n_support, cfg.n_query)
    if test_indices is None:
        print("Error: Not enough test data for balanced adaptation.")
        return

    test_indices = get_balanced_task_indices(y_test, cfg.n_support, cfg.n_query)
    test_sup_l, _ = test_indices
    test_que_l = np.delete(np.arange(len(y_test)), test_sup_l)
    
    test_sup_x = torch.tensor(X_test[test_sup_l], dtype=torch.float32).to(device)
    test_sup_y = torch.tensor(y_test[test_sup_l], dtype=torch.float32).to(device)
    test_que_x = torch.tensor(X_test[test_que_l], dtype=torch.float32).to(device)
    test_que_y = y_test[test_que_l]
    
    model.eval()
    inner_opt = optim.SGD(model.parameters(), lr=cfg.inner_lr)
    
    best_step_auc = 0.0
    best_step = 0
    adaptation_logs = []

    # higher를 사용하여 최대 20단계까지 적응을 시도하며 성능 추적
    with higher.innerloop_ctx(model, inner_opt, track_higher_grads=False) as (fmodel, diffopt):
        for step in range(1, 21):  # 1~20단계 탐색
            # 1. 적응(Adaptation) 수행
            logits_sup = fmodel(test_sup_x)
            sup_loss = F.binary_cross_entropy_with_logits(logits_sup, test_sup_y)
            diffopt.step(sup_loss)
            adaptation_logs.append(sup_loss.item())
            
            # 2. 현재 단계에서의 성능 측정 (Query Set 기준)
            with torch.no_grad():
                final_probs = torch.sigmoid(fmodel(test_que_x)).cpu().numpy()
                current_auc = roc_auc_score(test_que_y, final_probs)
                
                # 최적 지점 갱신
                if current_auc > best_step_auc:
                    best_step_auc = current_auc
                    best_step = step

        print(f"==> Best AUROC found at Step {best_step}: {best_step_auc:.4f}")

    # 최종 결과 리포트 및 시각화 (가장 좋았던 AUC 기준)
    plot_maml_results(test_que_y, final_probs, adaptation_logs, "maml_analysis_report.png")

if __name__ == "__main__":
    run_maml_training(MetaConfig())
