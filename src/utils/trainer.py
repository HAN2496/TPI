from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score

class BaseTrainer:
    def __init__(self, model, config, best_model_path=None):
        self.model = model
        self.config = config
        self.best_model_path = best_model_path

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )

        self.best_auroc = 0.0

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        for xb, yb in train_loader:
            self.optimizer.zero_grad()
            logits = self.model(xb)
            loss = self.criterion(logits, yb)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * xb.size(0)
        return total_loss / len(train_loader.dataset)

    def evaluate(self, X, y):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            acc = (preds == y).float().mean().item()

            y_np = y.cpu().numpy()
            probs_np = probs.cpu().numpy()
            auroc = roc_auc_score(y_np, probs_np)

        return acc, auroc

    def fit(self, train_loader, X_val, y_val, epochs=None, verbose=False):
        if epochs is None:
            epochs = self.config['epochs']

        history = {'train_loss': [], 'val_acc': [], 'val_auroc': []}

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_acc, val_auroc = self.evaluate(X_val, y_val)

            history['train_loss'].append(train_loss)
            history['val_acc'].append(val_acc)
            history['val_auroc'].append(val_auroc)

            self.scheduler.step(val_auroc)

            if val_auroc > self.best_auroc:
                self.best_auroc = val_auroc
                if self.best_model_path:
                    Path(self.best_model_path).parent.mkdir(parents=True, exist_ok=True)
                    torch.save(self.model.state_dict(), self.best_model_path)
                    print(f"Best model saved with AUROC: {val_auroc:.4f}")

            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Val Acc: {val_acc:.4f} - Val AUROC: {val_auroc:.4f}")

        return val_acc, val_auroc, history


class CombinationTrainer:
    def __init__(self, model, config, best_model_path=None):
        self.model = model
        self.config = config
        self.best_model_path = best_model_path

    def fit(self, X_train, y_train, X_val, y_val, verbose=False):
        self.model.model.fit(X_train, y_train)

        y_pred = self.model.model.predict(X_val)
        y_probs = self.model.model.predict_proba(X_val)[:, 1]

        val_acc = accuracy_score(y_val, y_pred)
        val_auroc = roc_auc_score(y_val, y_probs)

        history = {'train_loss': [], 'val_acc': [val_acc], 'val_auroc': [val_auroc]}

        if verbose:
            print(f"Val Acc: {val_acc:.4f} - Val AUROC: {val_auroc:.4f}")

        return val_acc, val_auroc, history
