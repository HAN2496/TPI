import numpy as np
import torch
from captum.attr import IntegratedGradients, KernelShap

class IntegratedGradientsExplainer:
    """
    IntegratedGradients-based explainer for time series models.

    Computes attributions by integrating gradients along a path from baseline to input.
    Supports multiple baselines for conditional imputation.
    """
    def __init__(self, model, device='cpu', n_steps=50):
        self.model = model
        self.device = device
        self.n_steps = n_steps
        self.model.eval()
        self.ig = IntegratedGradients(self.model.forward)

    def explain_sample(self, x_sample, baseline=None):
        if not isinstance(x_sample, torch.Tensor):
            x_sample = torch.tensor(x_sample, dtype=torch.float32)

        if x_sample.ndim == 2:
            x_sample = x_sample.unsqueeze(0)

        x_sample = x_sample.to(self.device)

        if baseline is None:
            baseline = torch.zeros_like(x_sample)
        elif not isinstance(baseline, torch.Tensor):
            baseline = torch.tensor(baseline, dtype=torch.float32)
            if baseline.ndim == 2:
                baseline = baseline.unsqueeze(0)
            baseline = baseline.to(self.device)

        # Handle multiple baselines (conditional imputation)
        if baseline.ndim == 3 and baseline.shape[0] > 1:
            # baseline shape: (n_baselines, T, F)
            # Compute attribution for each baseline and average
            all_attributions = []
            for i in range(baseline.shape[0]):
                single_baseline = baseline[i:i+1]
                with torch.set_grad_enabled(True):
                    attr = self.ig.attribute(
                        x_sample,
                        baselines=single_baseline,
                        n_steps=self.n_steps,
                        internal_batch_size=1
                    )
                all_attributions.append(attr.detach().cpu().numpy())

            attributions = np.mean(all_attributions, axis=0)
        else:
            # Single baseline
            with torch.set_grad_enabled(True):
                attributions = self.ig.attribute(
                    x_sample,
                    baselines=baseline,
                    n_steps=self.n_steps,
                    internal_batch_size=1
                )
            attributions = attributions.detach().cpu().numpy()

        return attributions


class KernelShapExplainer:
    """
    KernelShap-based explainer for time series models.

    Uses coalition-based Shapley values via the LIME framework.
    More theoretically aligned with TimeSHAP but computationally intensive.
    """
    def __init__(self, model, device='cpu', n_samples=128):
        self.model = model
        self.device = device
        self.n_samples = n_samples
        self.model.eval()
        self.ks = KernelShap(self.model.forward)

    def explain_sample(self, x_sample, baseline=None):
        if not isinstance(x_sample, torch.Tensor):
            x_sample = torch.tensor(x_sample, dtype=torch.float32)

        if x_sample.ndim == 2:
            x_sample = x_sample.unsqueeze(0)

        x_sample = x_sample.to(self.device)

        if baseline is None:
            baseline = torch.zeros_like(x_sample)
        elif not isinstance(baseline, torch.Tensor):
            baseline = torch.tensor(baseline, dtype=torch.float32)
            if baseline.ndim == 2:
                baseline = baseline.unsqueeze(0)
            baseline = baseline.to(self.device)

        # Handle multiple baselines (conditional imputation)
        if baseline.ndim == 3 and baseline.shape[0] > 1:
            # baseline shape: (n_baselines, T, F)
            # Use multiple baselines for conditional expectation
            all_attributions = []
            for i in range(baseline.shape[0]):
                single_baseline = baseline[i:i+1]
                with torch.set_grad_enabled(True):
                    attr = self.ks.attribute(
                        x_sample,
                        baselines=single_baseline,
                        n_samples=self.n_samples,
                        perturbations_per_eval=1
                    )
                all_attributions.append(attr.detach().cpu().numpy())

            attributions = np.mean(all_attributions, axis=0)
        else:
            # Single baseline
            with torch.set_grad_enabled(True):
                attributions = self.ks.attribute(
                    x_sample,
                    baselines=baseline,
                    n_samples=self.n_samples,
                    perturbations_per_eval=1
                )
            attributions = attributions.detach().cpu().numpy()

        return attributions


def create_explainer(model, method='ig', device='cpu', **kwargs):
    """
    Factory function to create explainers.

    Args:
        model: PyTorch model
        method: 'ig' for IntegratedGradients, 'kernelshap' for KernelShap
        device: 'cpu' or 'cuda'
        **kwargs: Additional arguments for explainer (n_steps, n_samples, etc.)

    Returns:
        Explainer instance
    """
    if method == 'ig':
        return IntegratedGradientsExplainer(model, device, **kwargs)
    elif method == 'kernelshap':
        return KernelShapExplainer(model, device, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'ig' or 'kernelshap'.")