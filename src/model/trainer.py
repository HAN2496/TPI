from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

class BaseTrainer:
    def __init__(self, model, config, best_model_path=None, device="cpu"):
        self.model = model
        self.config = config
        self.best_model_path = best_model_path
        self.device = device
        self.model.to(self.device)

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        self.best_auroc = float('-inf')
        self.best_val_loss = float('inf')

    def compute_loss(self, batch, return_logits=False):
        xb, yb = batch
        xb = xb.to(self.device)
        yb = yb.to(self.device)
        logits = self.model(xb)
        loss = self.criterion(logits, yb)
        if return_logits:
            return loss, logits.detach(), yb.detach()
        return loss

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        for batch in train_loader:
            self.optimizer.zero_grad()
            loss = self.compute_loss(batch)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * batch[0].size(0)
        return total_loss / len(train_loader.dataset)

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_probs  = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                loss, logits, yb = self.compute_loss(batch, return_logits=True)
                total_loss += loss.item() * batch[0].size(0)

                prob = torch.sigmoid(logits)
                all_probs.append(prob.cpu())
                all_labels.append(yb.cpu())

        all_probs  = torch.cat(all_probs, dim=0).view(-1).numpy()
        all_labels = torch.cat(all_labels, dim=0).view(-1).numpy()
        auroc = roc_auc_score(all_labels, all_probs)
        return total_loss / len(val_loader.dataset), auroc

    def train(self, train_loader, val_loader, epochs=None, verbose=0):
        if epochs is None:
            epochs = self.config['epochs']

        history = {'train_loss': [], 'val_loss': [], 'val_auroc': []}

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_auroc = self.evaluate(val_loader)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_auroc'].append(val_auroc)

            # self.scheduler.step(train_loss)
            self.scheduler.step(val_loss)
            # self.scheduler.step(val_auroc)

            if val_auroc > self.best_auroc:
                self.best_auroc = val_auroc
            if val_loss < self.best_val_loss:
                # self.best_auroc = val_auroc
                if self.best_model_path:
                    Path(self.best_model_path).parent.mkdir(parents=True, exist_ok=True)
                    torch.save(self.model.state_dict(), self.best_model_path)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss

            if verbose >= 1 and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

            # torch.save(self.model.state_dict(), self.best_model_path)

        return history, self.best_auroc


class RegressionTrainer:
    def __init__(self, model, best_model_path=None):
        self.model = model
        self.best_model_path = best_model_path

    def _preproces_loader(self, data_loader):
        X_list = []
        y_list = []
        for batch in data_loader:
            xb, yb = batch
            X_list.append(xb)
            y_list.append(yb)
        X = torch.cat(X_list, dim=0)
        y = torch.cat(y_list, dim=0)
        return X.numpy(), y.numpy()

    def train(self, train_loader, val_loader, **kwargs):
        X_train, y_train = self._preproces_loader(train_loader)
        X_val, y_val = self._preproces_loader(val_loader)

        self.model.fit(X_train, y_train)

        y_probs = self.model.predict_proba(X_val)

        val_auroc = roc_auc_score(y_val, y_probs)
        return {}, val_auroc

def create_trainer(model, config, best_model_path=None, device="cpu", is_neural=True):
    if is_neural:
        return BaseTrainer(model, config['trainer'], best_model_path, device)
    else:
        return RegressionTrainer(model, best_model_path)