import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
from sklearn.metrics import roc_auc_score
import higher


def get_balanced_task_indices(labels, n_support, n_query):
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    n_half_sup = n_support // 2
    n_half_que = n_query // 2
    if len(pos_idx) < (n_half_sup + n_half_que) or len(neg_idx) < (n_half_sup + n_half_que):
        return None
    p_sel = np.random.choice(pos_idx, n_half_sup + n_half_que, replace=False)
    n_sel = np.random.choice(neg_idx, n_half_sup + n_half_que, replace=False)
    sup_l = np.concatenate([p_sel[:n_half_sup], n_sel[:n_half_sup]])
    que_l = np.concatenate([p_sel[n_half_sup:], n_sel[n_half_sup:]])
    return sup_l, que_l


class MAMLTrainer:
    def __init__(self, model, config, log_dir=None):
        self.model = model
        self.config = config
        self.device = torch.device(config['device'])
        self.log_dir = log_dir
        self.model.to(self.device)

        self.meta_optimizer = optim.Adam(
            model.parameters(), lr=config['outer_lr'], weight_decay=1e-4)

    def train(self, item_series, per_user_items, n_users, verbose=1):
        cfg = self.config
        n_tasks_per_epoch = cfg.get('n_tasks_per_epoch', 20)
        metrics = defaultdict(list)
        best_loss = float('inf')
        best_state_dict = None

        for epoch in range(cfg['meta_epochs']):
            self.model.train()
            meta_loss = 0.0
            task_count = 0

            for _ in range(n_tasks_per_epoch):
                uid = np.random.randint(0, n_users)
                item_ids, labels = per_user_items[uid]
                indices = get_balanced_task_indices(labels, cfg['n_support'], cfg['n_query'])
                if indices is None:
                    continue
                sup_l, que_l = indices

                sup_x = torch.tensor(item_series[item_ids[sup_l]], dtype=torch.float32).to(self.device)
                sup_y = torch.tensor(labels[sup_l], dtype=torch.float32).to(self.device)
                que_x = torch.tensor(item_series[item_ids[que_l]], dtype=torch.float32).to(self.device)
                que_y = torch.tensor(labels[que_l], dtype=torch.float32).to(self.device)

                inner_opt = optim.SGD(self.model.parameters(), lr=cfg['inner_lr'])
                with higher.innerloop_ctx(self.model, inner_opt, copy_initial_weights=False) as (fmodel, diffopt):
                    for _ in range(cfg['inner_steps']):
                        diffopt.step(F.binary_cross_entropy_with_logits(fmodel(sup_x), sup_y))
                    que_loss = F.binary_cross_entropy_with_logits(fmodel(que_x), que_y)
                    que_loss.backward()
                    meta_loss += que_loss.item()
                    task_count += 1

            if task_count > 0:
                self.meta_optimizer.step()
                self.meta_optimizer.zero_grad()

            avg_loss = meta_loss / max(1, task_count)
            metrics['train/meta_loss'].append(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}

            if verbose > 0 and epoch % 10 == 0:
                print(f"Epoch {epoch:03d} | Avg Meta Loss: {avg_loss:.4f}  best={best_loss:.4f}")

        if best_state_dict is not None:
            if self.log_dir is not None:
                torch.save(best_state_dict, self.log_dir / "best_maml.pt")
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

        for t in range(1, split_idx + 1):
            sup_x = torch.tensor(X_test[:t], dtype=torch.float32).to(self.device)
            sup_y = torch.tensor(y_test[:t], dtype=torch.float32).to(self.device)

            self.model.eval()
            inner_opt = optim.SGD(self.model.parameters(), lr=cfg['inner_lr'])

            with higher.innerloop_ctx(self.model, inner_opt, track_higher_grads=False) as (fmodel, diffopt):
                for _ in range(cfg['inner_steps']):
                    diffopt.step(F.binary_cross_entropy_with_logits(fmodel(sup_x), sup_y))
                with torch.no_grad():
                    probs = torch.sigmoid(fmodel(holdout_x)).cpu().numpy()

            auroc = roc_auc_score(holdout_y, probs)
            ctx_sizes.append(t)
            test_aurocs.append(auroc)
            final_probs = probs

            if t in snapshot_steps:
                snapshots[snapshot_steps[t]] = probs

        return ctx_sizes, test_aurocs, final_probs, holdout_y, snapshots
