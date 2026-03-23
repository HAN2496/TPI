import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
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

            if verbose > 0 and epoch % 10 == 0:
                print(f"Epoch {epoch:03d} | Avg Meta Loss: {meta_loss / max(1, task_count):.4f}")

        if self.log_dir is not None:
            torch.save(self.model.state_dict(), self.log_dir / "best_maml.pt")

    def adapt_and_evaluate(self, X_test, y_test, max_adapt_steps=20, verbose=1):
        cfg = self.config
        indices = get_balanced_task_indices(y_test, cfg['n_support'], cfg['n_query'])
        if indices is None:
            return None, None, None, []

        test_sup_l, _ = indices
        test_que_l = np.delete(np.arange(len(y_test)), test_sup_l)

        sup_x = torch.tensor(X_test[test_sup_l], dtype=torch.float32).to(self.device)
        sup_y = torch.tensor(y_test[test_sup_l], dtype=torch.float32).to(self.device)
        que_x = torch.tensor(X_test[test_que_l], dtype=torch.float32).to(self.device)
        que_y = y_test[test_que_l]

        self.model.eval()
        inner_opt = optim.SGD(self.model.parameters(), lr=cfg['inner_lr'])

        best_auc = 0.0
        best_step = 0
        adaptation_logs = []
        final_probs = None

        with higher.innerloop_ctx(self.model, inner_opt, track_higher_grads=False) as (fmodel, diffopt):
            for step in range(1, max_adapt_steps + 1):
                sup_loss = F.binary_cross_entropy_with_logits(fmodel(sup_x), sup_y)
                diffopt.step(sup_loss)
                adaptation_logs.append(sup_loss.item())

                with torch.no_grad():
                    probs = torch.sigmoid(fmodel(que_x)).cpu().numpy()
                    auc = roc_auc_score(que_y, probs)
                    if auc > best_auc:
                        best_auc = auc
                        best_step = step
                        final_probs = probs

        if verbose > 0:
            print(f"==> Best AUROC found at Step {best_step}: {best_auc:.4f}")

        return best_auc, final_probs, que_y, adaptation_logs
