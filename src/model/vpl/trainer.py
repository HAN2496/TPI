import math
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from .utils import log_metrics

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
    def __init__(self, model, logger, config):
        self.model = model
        self.config = config
        self.device = config["device"]
        self.logger = logger

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )

        self.best_val_loss = float('inf')

        if config['early_stop']:
            self.early_stopper = EarlyStopper(
                patience=config['patience'],
                min_delta=config['min_delta']
            )

        self.best_model_path = self.logger.log_dir / "best_model.pt"
        self.n_epochs = config['n_epochs']
        self.metrics = None

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        total_samples = 0
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

            batch_size = observations.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        avg_loss = total_loss / total_samples
        avg_metrics = {key: np.mean(vals) for key, vals in batch_metrics.items()}

        return avg_loss, avg_metrics

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        total_samples = 0
        batch_metrics = defaultdict(list)

        for batch in val_loader:
            with torch.no_grad():
                observations = batch["observations"].to(self.device).float()
                observations_2 = batch["observations_2"].to(self.device).float()
                labels = batch["labels"].to(self.device).float()
                loss, metrics = self.model(
                    observations, observations_2, labels
                )

                for key, val in metrics.items():
                    batch_metrics[key].append(val)

                batch_size = observations.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

        avg_loss = total_loss / total_samples
        avg_metrics = {key: np.mean(vals) for key, vals in batch_metrics.items()}

        return avg_loss, avg_metrics

    def train(self, train_loader, val_loader, verbose=1, warmup_epochs=10):
        self.metrics = defaultdict(list)

        for epoch in range(self.n_epochs):
            train_loss, train_metrics = self.train_epoch(train_loader)
            val_loss, val_metrics = self.evaluate(val_loader)

            # Store epoch-level metrics
            self.metrics["train/loss"].append(train_loss)
            self.metrics["eval/loss"].append(val_loss)

            for key, val in train_metrics.items():
                self.metrics[f"train/{key}"].append(val)

            for key, val in val_metrics.items():
                self.metrics[f"eval/{key}"].append(val)

            self.scheduler.step(val_loss)

            if epoch >= warmup_epochs:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    torch.save(self.model.state_dict(), self.best_model_path)
                    print(f"  --> New best model saved at epoch {epoch+1} with val loss {val_loss:.4f}")
            else:
                if verbose >= 1 and (epoch + 1) % int(self.config["eval_freq"]) == 0:
                    print(f"  (Warm-up: Skipping best model save)")

            if self.model.annealer:
                self.model.annealer.step()

            if (epoch + 1) % int(self.config["eval_freq"]) == 0:
                log_metrics(self.metrics, epoch, self.logger)
                if verbose >= 1:
                    print(f"Epoch {epoch+1}/{self.n_epochs} - "
                        f"Loss: {train_loss:.4f} - eval Loss: {val_loss:.4f} - "
                        f"eval Acc: {val_metrics['accuracy']:.4f}")

        return self.metrics, np.mean(self.metrics['eval/accuracy'][-5:])


class VPLTrainerBinary(VPLTrainer):
    """
    Trainer for VPL Binary Model (No observations_2)
    """
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        total_samples = 0
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

            batch_size = observations.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        avg_loss = total_loss / total_samples
        avg_metrics = {key: np.mean(vals) for key, vals in batch_metrics.items()}

        return avg_loss, avg_metrics

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        total_samples = 0
        batch_metrics = defaultdict(list)

        for batch in val_loader:
            with torch.no_grad():
                observations = batch["observations"].to(self.device).float()
                labels = batch["labels"].to(self.device).float()
                
                loss, metrics = self.model(observations, labels)

                for key, val in metrics.items():
                    batch_metrics[key].append(val)

                batch_size = observations.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

        avg_loss = total_loss / total_samples
        avg_metrics = {key: np.mean(vals) for key, vals in batch_metrics.items()}

        return avg_loss, avg_metrics
