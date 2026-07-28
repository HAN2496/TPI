import math
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict


class Annealer:
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
        return kld * self.slope()

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
            raise ValueError("Invalid shape for annealing function. Must be linear, cosine, logistic, or none.")
        return y * (1 - self.baseline) + self.baseline

    def step(self):
        if self.current_step < self.total_steps:
            self.current_step += 1
        if self.cyclical and self.current_step >= self.total_steps:
            self.current_step = 0


class EarlyStopper:
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
    def __init__(self, model, log_dir, config):
        self.model = model
        self.config = config
        self.device = config["device"]
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10)

        self.best_val_loss = float('inf')
        if config['early_stop']:
            self.early_stopper = EarlyStopper(patience=config['patience'], min_delta=config['min_delta'])

        self.best_model_path = self.log_dir / "best_model.pt"
        self.n_epochs = config['n_epochs']
        self.metrics = None

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss, total_samples = 0, 0
        batch_metrics = defaultdict(list)

        for batch in train_loader:
            self.optimizer.zero_grad()
            observations = batch["observations"].to(self.device).float()
            observations_2 = batch["observations_2"].to(self.device).float()
            labels = batch["labels"].to(self.device).float()
            loss, metrics = self.model(observations, observations_2, labels)
            loss.backward()
            self.optimizer.step()
            for key, val in metrics.items():
                batch_metrics[key].append(val)
            total_loss += loss.item() * observations.size(0)
            total_samples += observations.size(0)

        return total_loss / total_samples, {k: np.mean(v) for k, v in batch_metrics.items()}

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss, total_samples = 0, 0
        batch_metrics = defaultdict(list)

        with torch.no_grad():
            for batch in val_loader:
                observations = batch["observations"].to(self.device).float()
                observations_2 = batch["observations_2"].to(self.device).float()
                labels = batch["labels"].to(self.device).float()
                loss, metrics = self.model(observations, observations_2, labels)
                for key, val in metrics.items():
                    batch_metrics[key].append(val)
                total_loss += loss.item() * observations.size(0)
                total_samples += observations.size(0)

        return total_loss / total_samples, {k: np.mean(v) for k, v in batch_metrics.items()}

    def train(self, train_loader, val_loader, verbose=1, warmup_epochs=10):
        self.metrics = defaultdict(list)

        for epoch in range(self.n_epochs):
            train_loss, train_metrics = self.train_epoch(train_loader)
            val_loss, val_metrics = self.evaluate(val_loader)

            self.metrics["train/loss"].append(train_loss)
            self.metrics["eval/loss"].append(val_loss)
            for key, val in train_metrics.items():
                self.metrics[f"train/{key}"].append(val)
            for key, val in val_metrics.items():
                self.metrics[f"eval/{key}"].append(val)

            self.scheduler.step(val_loss)

            if epoch >= warmup_epochs and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.best_model_path)
                if verbose >= 1:
                    print(f"  --> New best model saved at epoch {epoch+1} with val loss {val_loss:.4f}")

            if self.model.annealer:
                self.model.annealer.step()

            if (epoch + 1) % int(self.config["eval_freq"]) == 0:
                pass  # log_metrics placeholder
                if verbose >= 1:
                    print(f"Epoch {epoch+1}/{self.n_epochs} - "
                          f"Loss: {train_loss:.4f} - eval Loss: {val_loss:.4f} - "
                          f"eval Acc: {val_metrics['accuracy']:.4f}")

        return self.metrics, np.mean(self.metrics['eval/accuracy'][-5:])


class VPLTrainerBinary(VPLTrainer):
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss, total_samples = 0, 0
        batch_metrics = defaultdict(list)

        for batch in train_loader:
            self.optimizer.zero_grad()
            observations = batch["observations"].to(self.device).float()
            labels = batch["labels"].to(self.device).float()
            loss, metrics = self.model(observations, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            for key, val in metrics.items():
                batch_metrics[key].append(val)
            total_loss += loss.item() * observations.size(0)
            total_samples += observations.size(0)

        return total_loss / total_samples, {k: np.mean(v) for k, v in batch_metrics.items()}

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss, total_samples = 0, 0
        batch_metrics = defaultdict(list)

        with torch.no_grad():
            for batch in val_loader:
                observations = batch["observations"].to(self.device).float()
                labels = batch["labels"].to(self.device).float()
                loss, metrics = self.model(observations, labels)
                for key, val in metrics.items():
                    batch_metrics[key].append(val)
                total_loss += loss.item() * observations.size(0)
                total_samples += observations.size(0)

        return total_loss / total_samples, {k: np.mean(v) for k, v in batch_metrics.items()}
