import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from sklearn.metrics import roc_auc_score


class CoPLGCFTrainer:
    def __init__(self, model, config, log_dir=None):
        self.model = model
        self.config = config
        self.device = config['device']
        self.log_dir = log_dir

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['gcf_lr'],
            weight_decay=config['gcf_weight_decay']
        )

        self.best_val_loss = float('inf')
        self.best_auc = -1.0
        self.best_state_dict = None

    def train_epoch(self, tr_u_t, tr_i_t, tr_y_t, pos_weight=None):
        self.model.train()
        self.optimizer.zero_grad()
        loss, _ = self.model.forward_pointwise(
            tr_u_t, tr_i_t, tr_y_t,
            pos_weight=pos_weight, sample_weight=None,
            test=False, lambda_reg=self.config['gcf_lambda_reg']
        )
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, va_u_t, va_i_t, va_y_np):
        self.model.eval()
        with torch.no_grad():
            va_y_t = torch.tensor(va_y_np, dtype=torch.float32, device=self.device)
            val_loss, val_logits = self.model.forward_pointwise(
                va_u_t, va_i_t, va_y_t,
                pos_weight=None, sample_weight=None,
                test=True, lambda_reg=0.0
            )
            val_prob = torch.sigmoid(val_logits).cpu().numpy()
            val_auc = roc_auc_score(va_y_np, val_prob)
        return val_loss.item(), val_auc

    def train(self, tr_u, tr_i, tr_y, va_u, va_i, va_y, verbose=1):
        tr_u_t = torch.tensor(tr_u, dtype=torch.long, device=self.device)
        tr_i_t = torch.tensor(tr_i, dtype=torch.long, device=self.device)
        tr_y_t = torch.tensor(tr_y, dtype=torch.float32, device=self.device)
        va_u_t = torch.tensor(va_u, dtype=torch.long, device=self.device)
        va_i_t = torch.tensor(va_i, dtype=torch.long, device=self.device)
        va_y_np = va_y.astype(np.int64)

        pos_weight = None
        if self.config.get('use_pos_weight', True):
            pos_cnt = int(tr_y.sum())
            neg_cnt = int((1 - tr_y).sum())
            pos_weight = torch.tensor([neg_cnt / max(1, pos_cnt)], dtype=torch.float32, device=self.device)

        n_epochs = self.config['gcf_epochs']
        metrics = defaultdict(list)

        for epoch in range(n_epochs):
            loss = self.train_epoch(tr_u_t, tr_i_t, tr_y_t, pos_weight)
            val_loss, val_auc = self.evaluate(va_u_t, va_i_t, va_y_np)

            metrics['train/loss'].append(loss)
            metrics['val/loss'].append(val_loss)
            metrics['val/auc'].append(val_auc)

            if val_auc > self.best_auc:
                self.best_auc = val_auc

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}

            if verbose > 0 and (epoch % 10 == 0 or epoch == n_epochs - 1):
                print(f"  [GCF] epoch={epoch:03d} loss={loss:.4f} val_auc={val_auc:.4f} best={self.best_auc:.4f}")

        if self.best_state_dict is not None:
            if self.log_dir is not None:
                torch.save(self.best_state_dict, self.log_dir / "best_gcf.pt")
            self.model.load_state_dict({k: v.to(self.device) for k, v in self.best_state_dict.items()})

        self.model.eval()
        with torch.no_grad():
            E_u, E_i = self.model.encode_graph(test=True)

        return self.best_auc, self.best_val_loss, E_u, E_i, metrics


class CoPLRMTrainer:
    def __init__(self, model, config, log_dir=None):
        self.model = model
        self.config = config
        self.device = config['device']
        self.log_dir = log_dir

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['rm_lr'],
            weight_decay=config['rm_weight_decay']
        )

        self.best_auc = -1.0
        self.best_state_dict = None

    def train_epoch(self, loader, E_u_train, pos_weight=None):
        self.model.train()
        total_loss, n_seen = 0.0, 0

        for uids_b, obs_b, y_b in loader:
            uids_b, obs_b, y_b = uids_b.to(self.device), obs_b.to(self.device), y_b.to(self.device)
            user_emb = E_u_train[uids_b]
            logits = self.model(user_emb, obs_b)
            loss_bce = F.binary_cross_entropy_with_logits(logits, y_b.float(), pos_weight=pos_weight)
            loss_reg = user_emb.norm(2).pow(2).mean()
            loss = loss_bce + self.config['rm_lambda_reg'] * loss_reg

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * len(uids_b)
            n_seen += len(uids_b)

        return total_loss / max(1, n_seen)

    def evaluate(self, loader, E_u_train):
        self.model.eval()
        all_prob, all_y = [], []

        with torch.no_grad():
            for uids_b, obs_b, y_b in loader:
                uids_b, obs_b = uids_b.to(self.device), obs_b.to(self.device)
                user_emb = E_u_train[uids_b]
                logits = self.model(user_emb, obs_b)
                all_prob.append(torch.sigmoid(logits).cpu().numpy())
                all_y.append(y_b.numpy())

        all_prob = np.concatenate(all_prob) if all_prob else np.array([])
        all_y = np.concatenate(all_y).astype(np.int64) if all_y else np.array([])
        return roc_auc_score(all_y, all_prob) if len(np.unique(all_y)) > 1 else 0.0

    def train(self, train_loader, val_loader, E_u_train, tr_y, verbose=1):
        pos_weight = None
        if self.config.get('use_pos_weight', True):
            pos_cnt = int(tr_y.sum())
            neg_cnt = int((1 - tr_y).sum())
            pos_weight = torch.tensor([neg_cnt / max(1, pos_cnt)], dtype=torch.float32, device=self.device)

        n_epochs = self.config['rm_epochs']
        metrics = defaultdict(list)

        for epoch in range(n_epochs):
            train_loss = self.train_epoch(train_loader, E_u_train, pos_weight)
            val_auc = self.evaluate(val_loader, E_u_train)

            metrics['train/loss'].append(train_loss)
            metrics['val/auc'].append(val_auc)

            if val_auc > self.best_auc:
                self.best_auc = val_auc
                self.best_state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}

            if verbose > 0 and (epoch % 5 == 0 or epoch == n_epochs - 1):
                print(f"  [RM ] epoch={epoch:03d} train_loss={train_loss:.4f} val_auc={val_auc:.4f} best={self.best_auc:.4f}")

        if self.best_state_dict is not None:
            if self.log_dir is not None:
                torch.save(self.best_state_dict, self.log_dir / "best_rm.pt")
            self.model.load_state_dict({k: v.to(self.device) for k, v in self.best_state_dict.items()})

        self.model.eval()
        return self.best_auc, metrics
