import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset




class MoETrainer:
    def __init__(self, model, config, log_dir=None):
        self.model = model
        self.config = config
        self.device = torch.device(config['device'])
        self.log_dir = log_dir
        self.model.to(self.device)

        self.optimizer = optim.AdamW(
            model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    def train(self, item_series, per_user_items, val_x=None, val_y=None, val_uids=None, verbose=1):
        cfg = self.config

        all_x, all_y, all_uids = [], [], []
        for uid, (item_ids, labels) in per_user_items.items():
            all_x.append(item_series[item_ids])
            all_y.append(labels)
            all_uids.append(np.full(len(labels), uid))

        X_t = torch.tensor(np.concatenate(all_x), dtype=torch.float32)
        y_t = torch.tensor(np.concatenate(all_y), dtype=torch.float32)
        u_t = torch.tensor(np.concatenate(all_uids), dtype=torch.long)

        dataset = TensorDataset(X_t, y_t, u_t)
        loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True)

        val_x_t = torch.tensor(val_x, dtype=torch.float32).to(self.device) if val_x is not None else None
        val_uids_t = torch.tensor(val_uids, dtype=torch.long).to(self.device) if val_uids is not None else None

        metrics = defaultdict(list)
        best_val_auroc = -1.0
        best_state_dict = None

        for epoch in range(cfg['epochs']):
            self.model.train()
            total_loss = 0.0
            for obs_b, y_b, u_b in loader:
                obs_b, y_b, u_b = obs_b.to(self.device), y_b.to(self.device), u_b.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(obs_b, uids=u_b)
                loss = F.binary_cross_entropy_with_logits(logits, y_b)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            metrics['train/loss'].append(avg_loss)

            val_auroc = float('nan')
            if val_x_t is not None:
                self.model.eval()
                with torch.no_grad():
                    val_probs = torch.sigmoid(self.model(val_x_t, uids=val_uids_t)).cpu().numpy()
                if len(np.unique(val_y)) > 1:
                    val_auroc = roc_auc_score(val_y, val_probs)
            metrics['val/auroc'].append(val_auroc)

            if not np.isnan(val_auroc) and val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}

            if verbose > 0 and (epoch % 10 == 0 or epoch == cfg['epochs'] - 1):
                print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f}  Val AUROC: {val_auroc:.4f}  best={best_val_auroc:.4f}")

        if best_state_dict is not None:
            if self.log_dir is not None:
                torch.save(best_state_dict, self.log_dir / "best_moe.pt")
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state_dict.items()})

        return metrics

    def adapt_sequential(self, X_test, y_test):
        cfg = self.config
        split_idx = len(X_test) // 2
        if split_idx < 1 or len(X_test) - split_idx < 1:
            return None, None, None, None, {}

        holdout_X = X_test[split_idx:]
        holdout_y = y_test[split_idx:]

        if len(np.unique(holdout_y)) < 2:
            return None, None, None, None, {}

        holdout_x = torch.tensor(holdout_X, dtype=torch.float32).to(self.device)

        target_pcts = [0.1, 0.2, 0.3, 0.4, 0.5]
        snapshot_steps = {max(1, int(len(X_test) * p)): int(p * 100) for p in target_pcts}
        snapshot_steps = {k: v for k, v in snapshot_steps.items() if k <= split_idx}

        ctx_sizes, test_aurocs = [], []
        final_probs = None
        snapshots = {}

        # MAML과 달리 공유 모델은 Freeze 하고, 새로운 유저 임베딩만 학습
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        for t in range(1, split_idx + 1):
            sup_x = torch.tensor(X_test[:t], dtype=torch.float32).to(self.device)
            sup_y = torch.tensor(y_test[:t], dtype=torch.float32).to(self.device)

            # 매 스텝마다 새 유저 임베딩 초기화 & 최적화 (Independent context adaptation)
            test_emb = nn.Parameter(torch.randn(1, self.model.gate[0].in_features).to(self.device))
            opt = optim.Adam([test_emb], lr=cfg['adapt_lr'])

            for _ in range(cfg['adapt_steps']):
                opt.zero_grad()
                emb_batch = test_emb.expand(len(sup_x), -1)
                logits = self.model(sup_x, custom_user_emb=emb_batch)
                loss = F.binary_cross_entropy_with_logits(logits, sup_y)
                loss.backward()
                opt.step()

            # Predict on holdout
            with torch.no_grad():
                holdout_emb_batch = test_emb.expand(len(holdout_x), -1)
                probs = torch.sigmoid(self.model(holdout_x, custom_user_emb=holdout_emb_batch)).cpu().numpy()

            auroc = roc_auc_score(holdout_y, probs)
            ctx_sizes.append(t)
            test_aurocs.append(auroc)
            final_probs = probs

            if t in snapshot_steps:
                snapshots[snapshot_steps[t]] = probs

        # 원상복구
        for param in self.model.parameters():
            param.requires_grad = True

        return ctx_sizes, test_aurocs, final_probs, holdout_y, snapshots