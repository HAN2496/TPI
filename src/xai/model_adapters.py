"""
Model adapters for TimeSHAP integration.

Wraps TPI models to be compatible with TimeSHAP's expected interface.
TimeSHAP expects a model function f(x) that returns event scores.

For our problem:
- p(y=1|tau) = sigmoid(sum(r(s_t))) or sigmoid(mean(r(s_t)))
- Offline models: R(tau) estimation (entire trajectory reward)
- Online models: r(s_t) estimation (per-step reward)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Union, Optional


class OnlineModelAdapter:
    """
    Adapter for online models (OnlineLSTM, OnlineMLP, OnlineAttention).

    These models have step_rewards() method that returns per-timestep rewards.
    For TimeSHAP, we need to return the final prediction: sigmoid(reduce(r(s_t)))
    """

    def __init__(self, model: nn.Module, device: str = 'cpu', return_rewards: bool = False):
        """
        Args:
            model: Online model with step_rewards() method
            device: Device to run the model on
            return_rewards: If True, return step rewards instead of final prediction
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.return_rewards = return_rewards

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Args:
            x: Input of shape (batch, time, features) or (time, features)

        Returns:
            If return_rewards=False: Event scores of shape (batch,) or scalar
            If return_rewards=True: Step rewards of shape (batch, time) or (time,)
        """
        # Convert to torch tensor
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        # Handle single instance (time, features) -> (1, time, features)
        squeeze_output = False
        if x.ndim == 2:
            x = x.unsqueeze(0)
            squeeze_output = True

        x = x.to(self.device)

        with torch.no_grad():
            if self.return_rewards:
                # Return per-timestep rewards: (B, T)
                rewards = self.model.step_rewards(x, detach=False)
                rewards = rewards.cpu().numpy()
                if squeeze_output:
                    rewards = rewards.squeeze(0)  # (T,)
                return rewards
            else:
                # Return final prediction: (B,)
                logits = self.model(x)
                probs = torch.sigmoid(logits)
                probs = probs.cpu().numpy()
                if squeeze_output:
                    probs = probs.item()  # scalar
                return probs

    def get_step_rewards(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Get per-timestep rewards r(s_t).

        Args:
            x: Input of shape (batch, time, features) or (time, features)

        Returns:
            Step rewards of shape (batch, time) or (time,)
        """
        old_return_rewards = self.return_rewards
        self.return_rewards = True
        rewards = self(x)
        self.return_rewards = old_return_rewards
        return rewards


class OfflineModelAdapter:
    """
    Adapter for offline models (OfflineLSTM, OfflineRegression).

    These models process the entire trajectory and return a single reward R(tau).
    """

    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        Args:
            model: Offline model (OfflineLSTM or OfflineRegression)
            device: Device to run the model on
        """
        if isinstance(model, nn.Module):
            self.model = model.to(device)
            self.model.eval()
            self.is_neural = True
        else:
            # sklearn model (OfflineRegression)
            self.model = model
            self.is_neural = False

        self.device = device

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Args:
            x: Input of shape (batch, time, features) or (time, features)

        Returns:
            Event scores of shape (batch,) or scalar
        """
        # Handle single instance (time, features) -> (1, time, features)
        squeeze_output = False
        if isinstance(x, np.ndarray) and x.ndim == 2:
            x = x[np.newaxis, ...]
            squeeze_output = True
        elif isinstance(x, torch.Tensor) and x.ndim == 2:
            x = x.unsqueeze(0)
            squeeze_output = True

        if self.is_neural:
            # Neural network model
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float()
            x = x.to(self.device)

            with torch.no_grad():
                logits = self.model(x)
                probs = torch.sigmoid(logits)
                probs = probs.cpu().numpy()
        else:
            # sklearn model
            if hasattr(self.model, 'predict_probability'):
                probs = self.model.predict_probability(x)
            else:
                # decision_function output
                logits = self.model.decision_function(x)
                probs = 1 / (1 + np.exp(-logits))  # sigmoid

        if squeeze_output:
            probs = probs.item() if isinstance(probs, np.ndarray) and probs.size == 1 else probs[0]

        return probs


def create_model_adapter(
    model,
    model_type: str = 'online',
    device: str = 'cpu',
    return_rewards: bool = False
) -> Union[OnlineModelAdapter, OfflineModelAdapter]:
    """
    Factory function to create appropriate model adapter.

    Args:
        model: TPI model (OnlineLSTM, OfflineLSTM, etc.)
        model_type: 'online' or 'offline'
        device: Device to run the model on
        return_rewards: For online models, whether to return step rewards

    Returns:
        Model adapter compatible with TimeSHAP
    """
    if model_type == 'online':
        return OnlineModelAdapter(model, device=device, return_rewards=return_rewards)
    elif model_type == 'offline':
        return OfflineModelAdapter(model, device=device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be 'online' or 'offline'")


class RewardEstimator:
    """
    Wrapper to estimate per-timestep rewards for TimeSHAP cell-level explanations.

    For online models: directly use step_rewards()
    For offline models: approximate by computing gradients or finite differences
    """

    def __init__(self, model, model_type: str = 'online', device: str = 'cpu'):
        """
        Args:
            model: TPI model
            model_type: 'online' or 'offline'
            device: Device to run the model on
        """
        self.model = model
        self.model_type = model_type
        self.device = device

        if model_type == 'online':
            self.adapter = OnlineModelAdapter(model, device=device, return_rewards=True)
        else:
            self.adapter = OfflineModelAdapter(model, device=device)

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Estimate per-timestep rewards.

        Args:
            x: Input of shape (batch, time, features) or (time, features)

        Returns:
            Step rewards of shape (batch, time) or (time,)
        """
        if self.model_type == 'online':
            # Directly return step rewards
            return self.adapter(x)
        else:
            # For offline models, return uniform contribution
            # (can be improved with gradient-based attribution)
            if isinstance(x, torch.Tensor):
                x = x.cpu().numpy()

            batch_size, seq_len, _ = x.shape if x.ndim == 3 else (1, x.shape[0], x.shape[1])

            # Get total reward
            total_reward = self.adapter(x)
            if not isinstance(total_reward, np.ndarray):
                total_reward = np.array([total_reward])

            # Distribute uniformly across timesteps
            rewards = np.tile(total_reward[:, np.newaxis], (1, seq_len)) / seq_len

            if x.ndim == 2:
                rewards = rewards.squeeze(0)

            return rewards
