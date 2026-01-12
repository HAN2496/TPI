import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from src.utils.visualization import (
    plot_roc_curve, 
    plot_prediction_scatter, 
    plot_step_rewards,
)

class Evaluator:
    def __init__(self, driver_name, model, paths, device="cpu"):
        self.driver_name = driver_name
        self.model = model
        self.paths = paths
        self.device = device
        self.is_neural = isinstance(model, nn.Module)
        
        if self.is_neural:
            self.model.to(self.device)
            self.model.eval()

    def get_predictions(self, data_loader):
        y_true_list = []
        y_probs_list = []

        if self.is_neural:
            self.model.eval()
            context = torch.no_grad()
        else:
            from contextlib import nullcontext
            context = nullcontext()

        with context:
            for batch in data_loader:
                inputs, targets = batch
                y_true_list.append(targets.numpy())

                if self.is_neural:
                    inputs = inputs.to(self.device)
                    probs = self.model.predict_proba(inputs)
                    y_probs_list.append(probs.cpu().numpy())
                else:
                    probs = self.model.predict_proba(inputs)
                    y_probs_list.append(probs)

        y_true = np.concatenate(y_true_list)
        y_probs = np.concatenate(y_probs_list)

        return y_true, y_probs

    def find_best_threshold(self, y_true, y_probs, step=0.01):
        thresholds = np.arange(0.1, 0.9, step)
        best_f1 = 0.0
        best_th = 0.5

        for th in thresholds:
            preds = (y_probs >= th).astype(int)
            score = f1_score(y_true, preds, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_th = th
        return best_th

    def compute_metrics(self, y_true, y_probs, threshold=0.5):
        y_pred = (y_probs >= threshold).astype(int)

        auroc = roc_auc_score(y_true, y_probs)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        return {
            "auroc": auroc,
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "threshold": threshold,
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
        }

    def save_plots(self, y_true, y_probs, threshold, data_loader, prefix=''):
        p = f"{prefix}_" if prefix else ""
        title_p = f"{prefix.capitalize()} " if prefix else ""

        # 1. ROC Curve
        plot_roc_curve(y_true, y_probs, save_path=self.paths.get(f'{p}roc_curve.png', create=True),
                       title=f'{title_p}ROC Curve ({self.driver_name})')

        # 2. Scatter Plot
        plot_prediction_scatter(y_true, y_probs, threshold=threshold,
                                save_path=self.paths.get(f'{p}prediction_scatter.png', create=True),
                                title=f'{title_p}Prediction Scatter Plot ({self.driver_name})')

        # 3. Step Rewards
        if hasattr(self.model, 'step_rewards'):
            X_sample, y_sample = data_loader.dataset[:10]
            if self.is_neural:
                X_sample = X_sample.to(self.device)
            plot_step_rewards(self.model, X_sample, y_sample, n_samples=10,
                              save_path=self.paths.get(f'{p}step_rewards.png', create=True))


    def evaluate(self, data_loader, threshold=None, save_plots=False, prefix=''):
        y_true, y_probs = self.get_predictions(data_loader)

        if threshold is None:
            threshold = self.find_best_threshold(y_true, y_probs)

        metrics = self.compute_metrics(y_true, y_probs, threshold)

        if save_plots:
            self.save_plots(y_true, y_probs, threshold, data_loader, prefix)

        return metrics, y_true, y_probs