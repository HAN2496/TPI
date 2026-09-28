"""Offline fixed-gain models and continuous-action value diagnostics.

These estimators only consume logged episodes. COMs is a contextual adaptation
of Trabucco et al. (2021), Eq. 3: contexts and user weights stay fixed while
adversarial ascent changes the gain. It is not the paper's full design optimizer.
"""
from __future__ import annotations

import copy
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


def gp_optimizer(objective, initial_theta, bounds):
    # A longer line search avoids premature termination on nearly deterministic
    # simulator responses. A failed optimizer is surfaced, never silently used.
    result = minimize(objective, initial_theta, method="L-BFGS-B", jac=True,
                      bounds=bounds, options={"maxiter": 300, "maxls": 100, "gtol": 1e-4})
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"GP hyperparameter fit failed: {result.message}")
    return result.x, result.fun


class FeatureRegressor:
    def __init__(self, kind="poly", seed=0):
        if kind not in ("poly", "gp"):
            raise ValueError(f"Unknown feature model: {kind}")
        self.kind, self.seed = kind, seed

    def fit(self, inputs, outcomes, kernel=None):
        inputs, outcomes = np.asarray(inputs), np.asarray(outcomes)
        self.lower = inputs.min(axis=0)
        self.span = np.maximum(np.ptp(inputs, axis=0), 1e-12)
        if self.kind == "poly":
            # Classical response-surface baseline: a full quadratic model
            # estimated by ordinary least squares. Input normalization is kept
            # for numerical conditioning, but no regularization is applied.
            self.model = make_pipeline(
                PolynomialFeatures(2, include_bias=False),
                LinearRegression(),
            )
        else:
            kernel = kernel if kernel is not None else (
                ConstantKernel(1., (1e-3, 1e3))
                * Matern(np.ones(inputs.shape[1]), (0.03, 30.), nu=2.5)
                + WhiteKernel(1e-4, (1e-8, 0.1))
            )
            self.model = GaussianProcessRegressor(
                kernel=kernel, normalize_y=True, alpha=1e-6, optimizer=gp_optimizer,
                n_restarts_optimizer=0, random_state=self.seed,
            )
        self.model.fit((inputs - self.lower) / self.span, outcomes)
        return self

    def predict(self, inputs):
        inputs = np.asarray(inputs)
        return self.model.predict((inputs - self.lower) / self.span)

    def mean_features(self, gains, contexts, chunk_size=2048):
        inputs = candidate_inputs(gains, contexts)
        predictions = np.concatenate([
            self.predict(inputs[i:i + chunk_size])
            for i in range(0, len(inputs), chunk_size)
        ])
        return predictions.reshape(len(gains), len(contexts), -1).mean(axis=1)


def candidate_inputs(gains, contexts):
    gains = np.asarray(gains).reshape(-1)
    return np.column_stack([
        np.repeat(gains, len(contexts)), np.tile(contexts, (len(gains), 1)),
    ])


def bounded_kernel_weights(gains, logged_gains, bounds, bandwidth, density):
    """Gaussian kernel / logging density, with analytic boundary correction.

    bandwidth and density are in *physical gain units*. The normalization is
    the kernel's integral over the action domain, not the sample weight sum.
    """
    gains = np.asarray(gains, dtype=float).reshape(-1)
    actions = np.asarray(logged_gains, dtype=float).reshape(-1)
    density = np.broadcast_to(np.asarray(density, dtype=float), actions.shape)
    lo, hi = map(float, bounds)
    if bandwidth <= 0 or lo >= hi or np.any(density <= 0):
        raise ValueError("Require positive bandwidth/density and ordered bounds")
    if np.any((gains < lo) | (gains > hi)) or np.any((actions < lo) | (actions > hi)):
        raise ValueError("Gain outside the logging support")
    z = (actions[None, :] - gains[:, None]) / bandwidth
    mass = ndtr((hi - gains) / bandwidth) - ndtr((lo - gains) / bandwidth)
    kernel = np.exp(-0.5 * z * z) / (np.sqrt(2 * np.pi) * bandwidth)
    return kernel / mass[:, None] / density[None, :]


def continuous_values(dm_features, observed, predicted_logged, weights):
    """Kernel IPW and DM + kernel-weighted residuals on an independent log.

    dm_features averages predictions over the *evaluation log's* contexts.
    At finite bandwidth DR targets the fixed action only approximately. It has
    smoothing bias; this function does not claim exact finite-sample robustness.
    """
    n = len(observed)
    if n == 0 or weights.shape[1] != n or predicted_logged.shape != observed.shape:
        raise ValueError("Inconsistent or empty evaluation log")
    residual = observed - predicted_logged
    return {
        "dm": dm_features,
        "ipw": weights @ observed / n,
        "dr": dm_features + weights @ residual / n,
        "ess": weights.sum(axis=1) ** 2 / np.maximum((weights ** 2).sum(axis=1), 1e-30),
    }


class ConservativeRewardModels:
    """Independent user reward MLPs, batched for efficient identical training.

    alpha=0 supplies the matched ordinary MLP ablation. Reward standardization
    is per user and uses training data only. No true preference weights are used.
    """
    def __init__(self, alpha=0.1, seed=0, steps=5000, hidden=64, device=None):
        self.alpha, self.seed, self.steps, self.hidden = alpha, seed, steps, hidden
        self.device = device

    def fit(self, inputs, rewards, gain_bounds, validation=None):
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(2)
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.lower = np.asarray(inputs).min(axis=0)
        self.span = np.maximum(np.ptp(inputs, axis=0), 1e-12)
        self.ymean, self.yscale = rewards.mean(axis=0), np.maximum(rewards.std(axis=0), 1e-8)
        n_users, width = rewards.shape[1], self.hidden

        class BatchedMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.weights, self.biases = nn.ParameterList(), nn.ParameterList()
                for nin, nout in zip([inputs.shape[1], width, width], [width, width, 1]):
                    self.weights.append(nn.Parameter(torch.randn(n_users, nin, nout) / np.sqrt(nin)))
                    self.biases.append(nn.Parameter(torch.zeros(n_users, 1, nout)))

            def forward(self, x):
                if x.ndim == 2:
                    x = x.unsqueeze(0).expand(n_users, -1, -1)
                for i, (w, b) in enumerate(zip(self.weights, self.biases)):
                    x = torch.bmm(x, w) + b
                    if i < 2:
                        x = torch.tanh(x)
                return x.squeeze(-1)

        net = BatchedMLP().to(device)
        x = torch.tensor((inputs - self.lower) / self.span, dtype=torch.float32, device=device)
        y = torch.tensor(((rewards - self.ymean) / self.yscale).T, dtype=torch.float32, device=device)
        if validation is not None:
            vx, vy = validation
            vx = torch.tensor((vx - self.lower) / self.span, dtype=torch.float32, device=device)
            vy = torch.tensor(((vy - self.ymean) / self.yscale).T, dtype=torch.float32, device=device)
        lo, hi = (np.asarray(gain_bounds) - self.lower[0]) / self.span[0]
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        rng = np.random.default_rng(self.seed)
        best_loss, best_state, best_step = float("inf"), None, self.steps
        self.history = []
        start = time.perf_counter()
        for step in range(1, self.steps + 1):
            ix = rng.integers(len(inputs), size=min(256, len(inputs)))
            xb, yb = x[ix], y[:, ix]
            if self.alpha:
                adversary = xb.unsqueeze(0).expand(n_users, -1, -1).clone()
                for _ in range(10):
                    adversary.requires_grad_(True)
                    gradient = torch.autograd.grad(net(adversary).sum(), adversary)[0]
                    updated_gain = (adversary[..., 0] + 0.02 * gradient[..., 0]).clamp(float(lo), float(hi))
                    adversary = torch.cat([updated_gain[..., None], adversary[..., 1:]], dim=-1).detach()
            pred = net(xb)
            mse = ((pred - yb) ** 2).mean()
            gap = (net(adversary) - pred).mean() if self.alpha else mse.new_zeros(())
            loss = 0.5 * mse + self.alpha * gap
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 50 == 0 or step == self.steps:
                with torch.no_grad():
                    score = float(((net(vx) - vy) ** 2).mean()) if validation is not None else float(mse)
                self.history.append({"step": step, "mse": float(mse.detach()), "gap": float(gap.detach()), "validation_mse": score})
                if validation is not None and score < best_loss:
                    best_loss, best_state, best_step = score, copy.deepcopy(net.state_dict()), step
        if best_state is not None:
            net.load_state_dict(best_state)
        self.net, self.best_step = net.eval(), best_step
        self.fit_seconds = time.perf_counter() - start
        return self

    def predict(self, inputs, chunk_size=1024):
        import torch
        result = []
        with torch.no_grad():
            for i in range(0, len(inputs), chunk_size):
                x = torch.tensor((inputs[i:i + chunk_size] - self.lower) / self.span, dtype=torch.float32, device=self.device)
                result.append(self.net(x).cpu().numpy().T * self.yscale + self.ymean)
        return np.concatenate(result)

    def mean_rewards(self, gains, contexts):
        prediction = self.predict(candidate_inputs(gains, contexts))
        return prediction.reshape(len(gains), len(contexts), -1).mean(axis=1)
