import math
import numpy as np
import torch
import torch.nn as nn


class Annealer:
    """
    KL annealing scheduler for VAE training.
    Copied from reference VPL workspace.
    """
    def __init__(self, total_steps, shape, baseline=0.0, cyclical=False, disable=False):
        self.total_steps = total_steps
        self.current_step = 0
        self.cyclical = cyclical
        self.shape = shape
        self.baseline = baseline
        if disable:
            self.shape = "none"
            self.baseline = 0.0

    def __call__(self, kld):
        out = kld * self.slope()
        return out

    def slope(self):
        if self.shape == "linear":
            y = self.current_step / self.total_steps
        elif self.shape == "cosine":
            y = (math.cos(math.pi * (self.current_step / self.total_steps - 1)) + 1) / 2
        elif self.shape == "logistic":
            exponent = (self.total_steps / 2) - self.current_step
            y = 1 / (1 + math.exp(exponent))
        elif self.shape == "none":
            y = 1.0
        else:
            raise ValueError(
                "Invalid shape for annealing function. Must be linear, cosine, or logistic."
            )
        y = self.add_baseline(y)
        return y

    def step(self):
        if self.current_step < self.total_steps:
            self.current_step += 1
        if self.cyclical and self.current_step >= self.total_steps:
            self.current_step = 0
        return

    def add_baseline(self, y):
        y_out = y * (1 - self.baseline) + self.baseline
        return y_out

    def cyclical_setter(self, value):
        if value is not bool:
            raise ValueError(
                "Cyclical_setter method requires boolean argument (True/False)"
            )
        else:
            self.cyclical = value
        return


class EarlyStopper:
    """
    Early stopping based on validation loss.
    Copied from reference VPL workspace.
    """
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class VPLTrainer:
    """
    Trainer for VPL (Variational Preference Learning) models.

    Key Point: Training 시 driver_ids는 사용하지 않음!
    Batch에서 observations와 labels만 사용.
    """
    def __init__(self, model, config, best_model_path=None, device='cpu'):
        self.model = model
        self.config = config
        self.device = device
        self.best_model_path = best_model_path

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.0)
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )

        self.best_val_loss = float('inf')

        if config.get('early_stop'):
            self.early_stopper = EarlyStopper(
                patience=config.get('patience', 10),
                min_delta=config.get('min_delta', 3e-4)
            )

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        metrics = {'recon_loss': [], 'kl_loss': [], 'accuracy': []}

        for batch in train_loader:
            obs = batch['observations'].to(self.device).float()
            labels = batch['labels'].to(self.device).float()

            self.optimizer.zero_grad()
            loss, batch_metrics = self.model(obs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * obs.size(0)
            for k, v in batch_metrics.items():
                if k in metrics:
                    metrics[k].append(v)

        return total_loss / len(train_loader.dataset), metrics

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        metrics = {'recon_loss': [], 'kl_loss': [], 'accuracy': []}

        with torch.no_grad():
            for batch in val_loader:
                obs = batch['observations'].to(self.device).float()
                labels = batch['labels'].to(self.device).float()

                loss, batch_metrics = self.model(obs, labels)
                total_loss += loss.item() * obs.size(0)
                for k, v in batch_metrics.items():
                    if k in metrics:
                        metrics[k].append(v)

        return total_loss / len(val_loader.dataset), metrics

    def train(self, train_loader, val_loader, epochs, verbose=1):
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}

        for epoch in range(epochs):
            train_loss, train_metrics = self.train_epoch(train_loader)
            val_loss, val_metrics = self.evaluate(val_loader)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(np.mean(val_metrics['accuracy']))

            self.scheduler.step(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                if self.best_model_path:
                    torch.save(self.model.state_dict(), self.best_model_path)

            if hasattr(self, 'early_stopper'):
                if self.early_stopper.early_stop(val_loss):
                    if verbose >= 1:
                        print(f"Early stopping at epoch {epoch+1}")
                    break

            if self.model.annealer:
                self.model.annealer.step()

            if verbose >= 1 and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - "
                      f"Val Acc: {np.mean(val_metrics['accuracy']):.4f}")

        return history, np.mean(history['val_accuracy'][-5:])
