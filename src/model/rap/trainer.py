import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

class RAPTrainer:
    def __init__(self, model, config, log_dir=None):
        self.model = model
        self.config = config
        self.device = torch.device(config['device'])
        self.log_dir = log_dir
        self.model.to(self.device)

        # Weight Decay를 높여서 파라미터 암기를 방해함
        self.optimizer = optim.AdamW(
            model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    def _sample_memory(self, full_x, full_y, full_uids, size=256):
        """메모리 뱅크에서 무작위로 샘플링"""
        idx = np.random.choice(len(full_y), size=min(size, len(full_y)), replace=False)
        return (
            torch.tensor(full_x[idx], dtype=torch.float32).to(self.device),
            torch.tensor(full_uids[idx], dtype=torch.long).to(self.device),
            torch.tensor(full_y[idx], dtype=torch.float32).to(self.device)
        )

    def train(self, item_series, per_user_items, val_x=None, val_y=None, val_uids=None, verbose=1):
        cfg = self.config

        all_x, all_y, all_uids = [], [], []
        for uid, (item_ids, labels) in per_user_items.items():
            all_x.append(item_series[item_ids])
            all_y.append(labels)
            all_uids.append(np.full(len(labels), uid))

        full_x = np.concatenate(all_x)
        full_y = np.concatenate(all_y)
        full_uids = np.concatenate(all_uids)

        dataset = TensorDataset(
            torch.tensor(full_x, dtype=torch.float32),
            torch.tensor(full_y, dtype=torch.float32),
            torch.tensor(full_uids, dtype=torch.long)
        )
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

                # 매 배치마다 랜덤하게 과거 기억을 떠올림(Memory Sampling)
                mem_obs, mem_uids, mem_y = self._sample_memory(full_x, full_y, full_uids, size=cfg['mem_size'])

                self.optimizer.zero_grad()
                logits = self.model(obs_b, uids=u_b, mem_obs=mem_obs, mem_uids=mem_uids, mem_y=mem_y)
                loss = F.binary_cross_entropy_with_logits(logits, y_b)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            metrics['train/loss'].append(avg_loss)

            val_auroc = float('nan')
            if val_x_t is not None:
                self.model.eval()
                mem_obs, mem_uids, mem_y = self._sample_memory(full_x, full_y, full_uids, size=cfg['mem_size'])
                with torch.no_grad():
                    val_probs = torch.sigmoid(
                        self.model(val_x_t, uids=val_uids_t, mem_obs=mem_obs, mem_uids=mem_uids, mem_y=mem_y)
                    ).cpu().numpy()
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
                torch.save(best_state_dict, self.log_dir / "best_rap.pt")
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state_dict.items()})

        return metrics

    def adapt_sequential(self, X_test, y_test, full_train_x, full_train_y, full_train_uids):
        cfg = self.config
        split_idx = len(X_test) // 2
        if split_idx < 1 or len(X_test) - split_idx < 1:
            return None, None, None, None, {}

        holdout_X = X_test[split_idx:]
        holdout_y = y_test[split_idx:]

        if len(np.unique(holdout_y)) < 2:
            return None, None, None, None, {}

        holdout_x = torch.tensor(holdout_X, dtype=torch.float32).to(self.device)

        # 평가 시에는 학습 데이터 전체(또는 대규모 샘플)를 고정 메모리로 사용
        mem_obs, mem_uids, mem_y = self._sample_memory(full_train_x, full_train_y, full_train_uids, size=1024)

        ctx_sizes, test_aurocs = [], []
        final_probs = None
        snapshots = {}

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        for t in range(1, split_idx + 1):
            sup_x = torch.tensor(X_test[:t], dtype=torch.float32).to(self.device)
            sup_y = torch.tensor(y_test[:t], dtype=torch.float32).to(self.device)

            # 새 유저 임베딩 초기화 & 최적화
            test_emb = nn.Parameter(torch.randn(1, self.model.user_emb.embedding_dim).to(self.device))
            opt = optim.Adam([test_emb], lr=cfg['adapt_lr'])

            # RAP의 핵심: 새 유저가 평가한 데이터도 Memory에 실시간으로 추가됨 (Zero/Few-shot Memory Expansion)
            current_mem_obs = torch.cat([mem_obs, sup_x], dim=0)
            current_mem_y = torch.cat([mem_y, sup_y], dim=0)
            
            for _ in range(cfg['adapt_steps']):
                opt.zero_grad()
                emb_batch = test_emb.expand(len(sup_x), -1)
                
                # 메모리 속 유저 임베딩 조립 (기존 유저들 + 현재 테스트 유저)
                mem_u_emb = torch.cat([self.model.user_emb(mem_uids), test_emb.expand(len(sup_x), -1)], dim=0)
                
                logits = self.model(sup_x, uids=None, mem_obs=current_mem_obs, mem_uids=None, mem_y=current_mem_y, 
                                    custom_user_emb=emb_batch, custom_mem_u_emb=mem_u_emb)
                loss = F.binary_cross_entropy_with_logits(logits, sup_y)
                loss.backward()
                opt.step()

            with torch.no_grad():
                holdout_emb_batch = test_emb.expand(len(holdout_x), -1)
                mem_u_emb = torch.cat([self.model.user_emb(mem_uids), test_emb.expand(len(sup_x), -1)], dim=0)
                
                probs = torch.sigmoid(self.model(holdout_x, uids=None, mem_obs=current_mem_obs, mem_uids=None, mem_y=current_mem_y, 
                                                 custom_user_emb=holdout_emb_batch, custom_mem_u_emb=mem_u_emb)).cpu().numpy()

            auroc = roc_auc_score(holdout_y, probs)
            ctx_sizes.append(t)
            test_aurocs.append(auroc)
            final_probs = probs

        for param in self.model.parameters():
            param.requires_grad = True

        return ctx_sizes, test_aurocs, final_probs, holdout_y, snapshots