import numpy as np
import torch

from src.model.vpl.trainer import VPLTrainer


class VPLPairwiseTrainer(VPLTrainer):
    """
    Trainer for VPL pairwise preference learning.
    Inherits from VPLTrainer, modifies batch processing for pairwise format.
    """

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        metrics = {'recon_loss': [], 'kl_loss': [], 'accuracy': []}

        for batch in train_loader:
            obs_a = batch['obs_a'].to(self.device).float()          # (B, set_len, T, d)
            obs_b = batch['obs_b'].to(self.device).float()          # (B, set_len, T, d)
            preferences = batch['preferences'].to(self.device).float()  # (B, set_len, 1)

            self.optimizer.zero_grad()
            loss, batch_metrics = self.model.forward_pairwise(obs_a, obs_b, preferences)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * obs_a.size(0)
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
                obs_a = batch['obs_a'].to(self.device).float()
                obs_b = batch['obs_b'].to(self.device).float()
                preferences = batch['preferences'].to(self.device).float()

                loss, batch_metrics = self.model.forward_pairwise(obs_a, obs_b, preferences)
                total_loss += loss.item() * obs_a.size(0)
                for k, v in batch_metrics.items():
                    if k in metrics:
                        metrics[k].append(v)

        return total_loss / len(val_loader.dataset), metrics
