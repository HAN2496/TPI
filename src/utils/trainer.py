from pathlib import Path
import pickle
import numpy as np

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

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

        self.best_auroc = 0.0
        self.best_val_loss = float('inf')

    def find_best_threshold(self, y_true, y_probs):
        thresholds = np.linspace(0.1, 0.9, 81)
        best_f1 = 0
        best_thresh = 0.5

        for thresh in thresholds:
            preds = (y_probs >= thresh).astype(int)
            f1 = f1_score(y_true, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        return best_thresh

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

        all_probs = torch.cat(all_probs).numpy()
        all_labels = torch.cat(all_labels).numpy()

        preds = (all_probs >= 0.5).astype(int)
        labels_int = (all_labels >= 0.5).astype(int)

        acc = accuracy_score(labels_int, preds)
        auroc = roc_auc_score(labels_int, all_probs)

        return acc, auroc, total_loss / len(val_loader.dataset)

    def train(self, train_loader, val_loader, epochs=None, verbose=False):
        if epochs is None:
            epochs = self.config['epochs']

        history = {'train_loss': [], 'val_acc': [], 'val_auroc': [], 'val_loss': []}

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_acc, val_auroc, val_loss = self.evaluate(val_loader)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_auroc'].append(val_auroc)

            self.scheduler.step(val_loss)
            # self.scheduler.step(val_auroc)

            # if val_auroc > self.best_auroc:
            if val_loss < self.best_val_loss:
                self.best_auroc = val_auroc
                if self.best_model_path:
                    Path(self.best_model_path).parent.mkdir(parents=True, exist_ok=True)
                    torch.save(self.model.state_dict(), self.best_model_path)
                    if verbose >= 2:
                        print(f"Best model saved with AUROC: {val_auroc:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss

            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} - Val AUROC: {val_auroc:.4f}")

        # Training 종료 후 최적 threshold 찾기
        if self.best_model_path:
            self.model.load_state_dict(torch.load(self.best_model_path, weights_only=True))
            self.model.eval()

            all_probs = []
            all_labels = []
            with torch.no_grad():
                for batch in val_loader:
                    xb, yb = batch
                    xb = xb.to(self.device)
                    logits = self.model(xb)
                    prob = torch.sigmoid(logits)
                    all_probs.append(prob.cpu())
                    all_labels.append(yb.cpu())

            all_probs = torch.cat(all_probs).numpy()
            all_labels = torch.cat(all_labels).numpy()
            labels_int = (all_labels >= 0.5).astype(int)

            best_threshold = self.find_best_threshold(labels_int, all_probs)
            self.model.best_threshold = best_threshold

            torch.save(self.model.state_dict(), self.best_model_path)

            if verbose:
                print(f"Best threshold: {best_threshold:.3f}")

        return val_acc, self.best_auroc, history


class CombinationTrainer:
    def __init__(self, model, config, best_model_path=None):
        self.model = model
        self.config = config
        self.best_model_path = best_model_path

    def find_best_threshold(self, y_true, y_probs):
        thresholds = np.linspace(0.1, 0.9, 81)
        best_f1 = 0
        best_thresh = 0.5

        for thresh in thresholds:
            preds = (y_probs >= thresh).astype(int)
            f1 = f1_score(y_true, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        return best_thresh

    def train(self, train_loader, val_loader, verbose=False):
        X_train, y_train = train_loader.dataset.tensors
        X_val, y_val = val_loader.dataset.tensors
        X_train, y_train = X_train.numpy(), y_train.numpy()
        X_val, y_val = X_val.numpy(), y_val.numpy()

        self.model.fit(X_train, y_train)

        y_probs = self.model.predict_probability(X_val)

        best_threshold = self.find_best_threshold(y_val.astype(int), y_probs)
        self.model.best_threshold = best_threshold

        y_pred = self.model.predict_label(X_val)

        val_acc = accuracy_score(y_val, y_pred)
        val_auroc = roc_auc_score(y_val, y_probs)

        if self.best_model_path:
            Path(self.best_model_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.best_model_path, 'wb') as f:
                pickle.dump(self.model.state_dict(), f)
            if verbose:
                print(f"Model saved with AUROC: {val_auroc:.4f}")
                print(f"Best threshold: {best_threshold:.3f}")

        history = {'val_acc': [val_acc], 'val_auroc': [val_auroc]}

        if verbose:
            print(f"Val Acc: {val_acc:.4f} - Val AUROC: {val_auroc:.4f}")

        return val_acc, val_auroc, history
