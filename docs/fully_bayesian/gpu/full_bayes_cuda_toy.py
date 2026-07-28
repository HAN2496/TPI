"""CPU/CUDA toy experiment for the complete hierarchical Bayesian pipeline.

The script implements both algorithms in the accompanying manuscript:

1. Raw trajectories -> manual statistics -> train-only standardization.
2. Population PG-Gibbs + NIW sampling with every posterior draw retained.
3. Held-out new users with a fixed context pool and fixed query set.
4. L sequential PG-Gibbs personalization sweeps for every population draw.
5. Fully Bayesian prediction E_m[sigmoid(theta_*^(m)^T z)].

The posterior-draw dimension and test-user dimension are batched on CPU/CUDA.
Only the L personalization sweeps and population Gibbs iterations are serial.
By default the CPU path uses the exact polyagamma sampler and the CUDA path
uses the K-term truncated approximation to PG(1, psi); pass --cpu-truncated to
run the same truncated sampler on the CPU too (a pure hardware comparison).
Projection-predictive, CMI, and spike-and-slab selection are deliberately not
mixed into this benchmark; this isolates the manuscript's base Algorithms 1--2.

Examples
--------
Small CPU smoke test:

    python full_bayes_cuda_toy.py --quick --cpu-only

CPU/CUDA comparison:

    python full_bayes_cuda_toy.py --require-cuda

Sweep the population size and PG truncation:

    python full_bayes_cuda_toy.py \
        --train-users 5 10 20 \
        --train-items 100 500 1000 \
        --pg-terms 25 50 100 \
        --chains 1 4 \
        --repeats 3

Pure-hardware comparison (same truncated PG sampler on CPU and CUDA):

    python full_bayes_cuda_toy.py --require-cuda --cpu-truncated
"""

import argparse
import csv
import itertools
import json
import math
import statistics
import time
from collections import OrderedDict
from pathlib import Path

try:
    import torch
except ModuleNotFoundError as error:
    raise SystemExit(
        "PyTorch is required. Run this script in a PyTorch/CUDA environment."
    ) from error

import numpy as np
from polyagamma import random_polyagamma


ALL_STATS = (
    "mean",
    "std",
    "abs_peak",
    "energy",
    "rms",
    "p95_abs",
    "p2p",
    "impulse_abs",
    "crest",
    "z_nod",
    "rms_deriv",
    "p2p_deriv",
    "abs_peak_deriv",
    "vdv",
    "vdv_deriv",
    "mtvv",
    "wrms_z",
    "wrms_xy",
    "wrms_rot",
    "wrms_z_d",
    "sigma_sd",
    "band_low",
    "band_mid",
    "rms_dfz",
)

CURRENT_STATS_PATTERN = (
    (
        "abs_peak_deriv",
        "p2p_deriv",
        "rms_deriv",
        "mean",
        "std",
        "abs_peak",
        "energy",
        "impulse_abs",
        "crest",
    ),
    (
        "abs_peak_deriv",
        "p2p_deriv",
        "rms_deriv",
        "mean",
        "std",
        "abs_peak",
        "energy",
        "impulse_abs",
        "crest",
    ),
    (
        "abs_peak",
        "p2p",
        "rms",
        "wrms_z",
        "vdv",
        "mean",
        "std",
        "energy",
        "impulse_abs",
        "crest",
    ),
    (
        "abs_peak",
        "p2p",
        "rms",
        "vdv",
        "mean",
        "std",
        "energy",
        "impulse_abs",
        "crest",
    ),
    (
        "abs_peak",
        "p2p",
        "rms",
        "mean",
        "std",
        "energy",
        "impulse_abs",
        "crest",
    ),
)

BASIC_STATS = (
    "mean",
    "std",
    "abs_peak",
    "energy",
    "impulse_abs",
    "crest",
)


def make_manual_stats(sensors, preset, custom_stats):
    unknown = set(custom_stats or ()) - set(ALL_STATS)
    if unknown:
        raise ValueError(f"unsupported statistics: {sorted(unknown)}")
    result = OrderedDict()
    for sensor in range(sensors):
        if custom_stats:
            methods = custom_stats
        elif preset == "basic":
            methods = BASIC_STATS
        elif preset == "current":
            methods = CURRENT_STATS_PATTERN[sensor % len(CURRENT_STATS_PATTERN)]
        elif preset == "full":
            methods = ALL_STATS
        else:
            raise ValueError("stats preset must be basic, current, or full")
        result[f"sensor_{sensor}"] = tuple(dict.fromkeys(methods))
    return result


def seed_all(seed, device):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(device, function):
    synchronize(device)
    start = time.perf_counter()
    result = function()
    synchronize(device)
    return result, time.perf_counter() - start


def warmup_device(device, dtype):
    seed_all(12345, device)
    matrix = torch.randn((4, 12, 12), device=device, dtype=dtype)
    covariance = matrix @ matrix.transpose(-1, -2)
    covariance = covariance + 1e-3 * torch.eye(12, device=device, dtype=dtype)
    torch.linalg.cholesky(covariance)
    pg_truncated(torch.zeros(256, device=device, dtype=dtype), 4, 4)
    synchronize(device)


def central_derivative(x, fs):
    if x.shape[2] < 2:
        return torch.zeros_like(x)
    derivative = torch.empty_like(x)
    derivative[:, :, 0] = (x[:, :, 1] - x[:, :, 0]) * fs
    derivative[:, :, -1] = (x[:, :, -1] - x[:, :, -2]) * fs
    if x.shape[2] > 2:
        derivative[:, :, 1:-1] = (x[:, :, 2:] - x[:, :, :-2]) * (0.5 * fs)
    return derivative


def iso_weight(frequency, kind):
    s = (2j * math.pi) * frequency
    w1, w2 = 2 * math.pi * 0.4, 2 * math.pi * 100.0
    high = s.square() / (s.square() + (w1 / 0.7071) * s + w1**2)
    low = w2**2 / (s.square() + (w2 / 0.7071) * s + w2**2)
    if kind == "wk":
        w3 = 2 * math.pi * 12.5
        w4, q4 = 2 * math.pi * 12.5, 0.63
        w5, q5 = 2 * math.pi * 2.37, 0.91
        w6, q6 = 2 * math.pi * 3.35, 0.91
        transition = (1 + s / w3) / (1 + s / (q4 * w4) + (s / w4).square())
        step = ((w6 / w5) ** 2) * (
            1 + s / (q5 * w5) + (s / w5).square()
        ) / (1 + s / (q6 * w6) + (s / w6).square())
        return torch.abs(high * low * transition * step)
    if kind == "wd":
        w4, q4 = 2 * math.pi * 2.0, 0.63
    elif kind == "we":
        w4, q4 = 2 * math.pi * 1.0, 0.63
    else:
        raise ValueError(f"unknown ISO weight: {kind}")
    transition = (1 + s / w4) / (1 + s / (q4 * w4) + (s / w4).square())
    return torch.abs(high * low * transition)


class ManualFeatures:
    def __init__(self, manual_stats, fs):
        self.manual_stats = manual_stats
        self.channels = list(manual_stats)
        self.fs = float(fs)
        self.pairs = []
        feature_names = []
        for channel_index, (channel, methods) in enumerate(manual_stats.items()):
            for method in methods:
                if method not in ALL_STATS:
                    raise ValueError(f"unsupported statistic: {method}")
                self.pairs.append((channel_index, method))
                feature_names.append(f"{channel}__{method}")
        self.feature_names = ["bias"] + feature_names
        self.mean = None
        self.scale = None

    def extract(self, raw):
        if raw.ndim != 4 or raw.shape[-1] != len(self.channels):
            raise ValueError("raw must have shape [users, items, time, sensors]")
        needed = {method for _, method in self.pairs}
        method_channels = {
            method: sorted({channel for channel, name in self.pairs if name == method})
            for method in needed
        }
        derivative_methods = {
            "rms_deriv",
            "p2p_deriv",
            "abs_peak_deriv",
            "vdv_deriv",
            "wrms_rot",
            "wrms_z_d",
        }
        raw_fft_methods = {"wrms_z", "wrms_xy", "sigma_sd", "band_low", "band_mid"}
        derivative_fft_methods = {"wrms_rot", "wrms_z_d"}
        derivative_channels = sorted(
            {
                channel
                for method in derivative_methods & needed
                for channel in method_channels[method]
            }
        )
        raw_fft_channels = sorted(
            {
                channel
                for method in raw_fft_methods & needed
                for channel in method_channels[method]
            }
        )
        derivative_fft_channels = sorted(
            {
                channel
                for method in derivative_fft_methods & needed
                for channel in method_channels[method]
            }
        )
        eps = torch.as_tensor(1e-12, device=raw.device, dtype=raw.dtype)
        values = {}
        cache = {}

        def index(channels):
            return torch.as_tensor(channels, device=raw.device, dtype=torch.long)

        def expand(method, selected):
            full = torch.zeros(
                (*raw.shape[:2], raw.shape[-1]), device=raw.device, dtype=raw.dtype
            )
            return full.index_copy(-1, index(method_channels[method]), selected)

        def raw_abs():
            if "abs" not in cache:
                cache["abs"] = raw.abs()
            return cache["abs"]

        def raw_square():
            if "square" not in cache:
                cache["square"] = raw.square()
            return cache["square"]

        def raw_mean():
            if "mean" not in cache:
                cache["mean"] = raw.mean(dim=2)
            return cache["mean"]

        def derivative():
            if "derivative" not in cache:
                selected = central_derivative(
                    raw.index_select(-1, index(derivative_channels)), self.fs
                )
                cache["derivative"] = torch.zeros_like(raw).index_copy(
                    -1, index(derivative_channels), selected
                )
            return cache["derivative"]

        def frequencies():
            if "frequencies" not in cache:
                cache["frequencies"] = torch.fft.rfftfreq(
                    raw.shape[2], d=1.0 / self.fs, device=raw.device, dtype=raw.dtype
                )
            return cache["frequencies"]

        def spectrum(source):
            key = f"{source}_spectrum"
            if key not in cache:
                if source == "raw":
                    selected = raw.index_select(-1, index(raw_fft_channels))
                else:
                    selected = derivative().index_select(
                        -1, index(derivative_fft_channels)
                    )
                cache[key] = torch.fft.rfft(selected, dim=2)
            return cache[key]

        def select_spectrum(source, method):
            available = raw_fft_channels if source == "raw" else derivative_fft_channels
            positions = {channel: position for position, channel in enumerate(available)}
            selected = [positions[channel] for channel in method_channels[method]]
            return spectrum(source).index_select(-1, index(selected))

        def weighted_rms(source, weight, method):
            selected = select_spectrum(source, method)
            weight_key = f"weight_{weight}"
            if weight_key not in cache:
                cache[weight_key] = iso_weight(frequencies(), weight)
            weighted = selected * cache[weight_key].reshape(1, 1, -1, 1)
            filtered = torch.fft.irfft(weighted, n=raw.shape[2], dim=2)
            return expand(method, filtered.square().mean(dim=2).sqrt())

        def band_sum(method, low, high):
            selected = select_spectrum("raw", method).abs().square()
            mask = (frequencies() >= low) & (frequencies() < high)
            return expand(method, selected[:, :, mask, :].sum(dim=2))

        if "mean" in needed:
            values["mean"] = raw_mean()
        if "std" in needed:
            values["std"] = raw.std(dim=2, unbiased=False)
        if "abs_peak" in needed or "crest" in needed:
            values["abs_peak"] = raw_abs().amax(dim=2)
        if {"energy", "rms", "crest"} & needed:
            values["energy"] = raw_square().mean(dim=2)
        if "rms" in needed:
            values["rms"] = values["energy"].sqrt()
        if "p95_abs" in needed:
            values["p95_abs"] = torch.quantile(raw_abs(), 0.95, dim=2)
        if "p2p" in needed:
            values["p2p"] = raw.amax(dim=2) - raw.amin(dim=2)
        if "impulse_abs" in needed:
            values["impulse_abs"] = raw_abs().sum(dim=2) / self.fs
        if "crest" in needed:
            values["crest"] = values["abs_peak"] / (values["energy"].sqrt() + eps)
        if "z_nod" in needed:
            if raw.shape[2] < 3:
                values["z_nod"] = torch.zeros_like(raw_mean())
            else:
                center = raw[:, :, 1:-1]
                peak = (center > raw[:, :, :-2]) & (center > raw[:, :, 2:]) & (center > 0)
                rank = peak.to(torch.int32).cumsum(dim=2)
                values["z_nod"] = torch.where(
                    peak & (rank == 3), center, torch.zeros_like(center)
                ).sum(dim=2)
        if "rms_deriv" in needed:
            values["rms_deriv"] = derivative().square().mean(dim=2).sqrt()
        if "p2p_deriv" in needed:
            values["p2p_deriv"] = derivative().amax(dim=2) - derivative().amin(dim=2)
        if "abs_peak_deriv" in needed:
            values["abs_peak_deriv"] = derivative().abs().amax(dim=2)
        if "vdv" in needed:
            values["vdv"] = (raw.pow(4).sum(dim=2) / self.fs).pow(0.25)
        if "vdv_deriv" in needed:
            values["vdv_deriv"] = (derivative().pow(4).sum(dim=2) / self.fs).pow(0.25)
        if "mtvv" in needed:
            window = int(round(self.fs))
            if window >= raw.shape[2]:
                values["mtvv"] = raw_square().mean(dim=2).sqrt()
            else:
                cumulative = raw_square().cumsum(dim=2)
                running = (cumulative[:, :, window:] - cumulative[:, :, :-window]) / window
                values["mtvv"] = running.amax(dim=2).sqrt()
        if "wrms_z" in needed:
            values["wrms_z"] = weighted_rms("raw", "wk", "wrms_z")
        if "wrms_xy" in needed:
            values["wrms_xy"] = weighted_rms("raw", "wd", "wrms_xy")
        if "wrms_rot" in needed:
            values["wrms_rot"] = weighted_rms("derivative", "we", "wrms_rot")
        if "wrms_z_d" in needed:
            values["wrms_z_d"] = weighted_rms("derivative", "wk", "wrms_z_d")
        if "sigma_sd" in needed:
            values["sigma_sd"] = band_sum("sigma_sd", 0.5, 5.0).sqrt()
        for method, low, high in (("band_low", 1.0, 2.0), ("band_mid", 4.0, 8.0)):
            if method in needed:
                selected = select_spectrum("raw", method).abs().square()
                mask = (frequencies() >= low) & (frequencies() < high)
                ratio = selected[:, :, mask].sum(dim=2) / (selected.sum(dim=2) + eps)
                values[method] = expand(method, ratio)
        if "rms_dfz" in needed:
            mean = raw_mean().unsqueeze(2)
            values["rms_dfz"] = ((raw - mean) / (mean + eps)).square().mean(dim=2).sqrt()

        return torch.stack(
            [values[method][..., channel] for channel, method in self.pairs], dim=-1
        )

    def fit_transform(self, raw):
        features = self.extract(raw)
        self.mean = features.mean(dim=(0, 1))
        self.scale = features.std(dim=(0, 1), unbiased=False) + 1e-12
        return self._standardize(features)

    def transform(self, raw):
        if self.mean is None:
            raise RuntimeError("fit_transform must be called on train data first")
        return self._standardize(self.extract(raw))

    def _standardize(self, features):
        z = (features - self.mean) / self.scale
        bias = torch.ones((*z.shape[:-1], 1), device=z.device, dtype=z.dtype)
        return torch.cat((bias, z), dim=-1)


def pg_truncated(psi, terms, chunk_size):
    if terms < 1 or chunk_size < 1:
        raise ValueError("PG terms and chunk size must be positive")
    result = torch.zeros_like(psi)
    scaled_psi2 = psi.square() / (4.0 * math.pi**2)
    prefix = (1,) * psi.ndim
    for start in range(0, terms, chunk_size):
        stop = min(start + chunk_size, terms)
        k = torch.arange(start + 1, stop + 1, device=psi.device, dtype=psi.dtype)
        denominator = (k - 0.5).square().reshape(*prefix, -1)
        denominator = denominator + scaled_psi2.unsqueeze(-1)
        exponentials = torch.empty(
            (*psi.shape, stop - start), device=psi.device, dtype=psi.dtype
        ).exponential_(1.0)
        result.add_((exponentials / denominator).sum(dim=-1))
    return result / (2.0 * math.pi**2)


def sample_pg(psi, cfg, rng):
    """Exact polyagamma on CPU when rng is set; truncated approximation otherwise."""
    if rng is not None and psi.device.type == "cpu":
        omega = random_polyagamma(h=1.0, z=psi.detach().numpy(), random_state=rng)
        return torch.from_numpy(omega).to(dtype=psi.dtype)
    return pg_truncated(psi, cfg["pg_terms"], cfg["pg_chunk"])


def pg_mean(psi):
    near_zero = psi.abs() < 1e-6
    safe = torch.where(near_zero, torch.ones_like(psi), psi)
    value = torch.tanh(safe / 2.0) / (2.0 * safe)
    return torch.where(near_zero, torch.full_like(value, 0.25), value)


def sample_inverse_wishart(scale, df, jitter):
    chains, d, _ = scale.shape
    if df <= d - 1:
        raise ValueError("inverse-Wishart requires df > d - 1")
    identity = torch.eye(d, device=scale.device, dtype=scale.dtype)
    scale_chol = torch.linalg.cholesky(scale + jitter * identity)
    wishart_scale = torch.cholesky_inverse(scale_chol)
    wishart_chol = torch.linalg.cholesky(wishart_scale + jitter * identity)

    bartlett = torch.tril(
        torch.randn((chains, d, d), device=scale.device, dtype=scale.dtype),
        diagonal=-1,
    )
    diag_df = df - torch.arange(d, device=scale.device, dtype=scale.dtype)
    gamma = torch.distributions.Gamma(
        (diag_df / 2.0).expand(chains, d),
        torch.full((chains, d), 0.5, device=scale.device, dtype=scale.dtype),
    ).sample()
    bartlett = bartlett + torch.diag_embed(gamma.sqrt())
    factor = wishart_chol @ bartlett
    expanded_identity = identity.expand(chains, d, d)
    factor_inv = torch.linalg.solve_triangular(
        factor, expanded_identity, upper=False
    )
    sigma = factor_inv.transpose(-1, -2) @ factor_inv
    return 0.5 * (sigma + sigma.transpose(-1, -2))


def fit_population(z, y, cfg, seed):
    if z.ndim != 3 or y.shape != z.shape[:2]:
        raise ValueError("z must be [users, items, features] and y [users, items]")
    seed_all(seed, z.device)
    pg_rng = np.random.default_rng(seed) if cfg["pg_exact"] and z.device.type == "cpu" else None
    users, _, d = z.shape
    chains = cfg["chains"]
    samples = cfg["samples"]
    total_steps = cfg["burnin"] + samples * cfg["thin"]
    identity = torch.eye(d, device=z.device, dtype=z.dtype)
    m0 = torch.zeros(d, device=z.device, dtype=z.dtype)
    lambda0 = cfg["niw_lambda0_scale"] * identity
    nu0 = float(d + 2) if cfg["niw_nu0"] is None else float(cfg["niw_nu0"])
    kappa = y - 0.5
    z_transpose = z.transpose(-1, -2)

    theta = 0.1 * torch.randn((chains, users, d), device=z.device, dtype=z.dtype)
    sigma = identity.expand(chains, d, d).clone()
    mu = torch.zeros((chains, d), device=z.device, dtype=z.dtype)
    epsilon = torch.zeros((chains, *y.shape), device=z.device, dtype=z.dtype)
    inv_eps = 0.0 if cfg["eps_var"] is None else 1.0 / cfg["eps_var"]

    mu_samples = torch.empty((chains, samples, d), device=z.device, dtype=z.dtype)
    sigma_samples = torch.empty(
        (chains, samples, d, d), device=z.device, dtype=z.dtype
    )
    theta_samples = torch.empty(
        (chains, samples, users, d), device=z.device, dtype=z.dtype
    )
    save = 0

    for iteration in range(total_steps):
        eta = torch.einsum("und,cud->cun", z, theta)
        omega = sample_pg(eta + epsilon, cfg, pg_rng)

        if cfg["eps_var"] is not None:
            precision_eps = omega + inv_eps
            mean_eps = (kappa.unsqueeze(0) - omega * eta) / precision_eps
            epsilon = mean_eps + torch.randn_like(epsilon) / precision_eps.sqrt()

        sigma_inv = torch.linalg.inv(sigma + cfg["jitter"] * identity)
        sigma_inv_mu = torch.einsum("cij,cj->ci", sigma_inv, mu)
        weighted_z = omega.unsqueeze(-1) * z.unsqueeze(0)
        precision_theta = (
            z_transpose.unsqueeze(0) @ weighted_z
            + sigma_inv.unsqueeze(1)
            + cfg["jitter"] * identity
        )
        residual = kappa.unsqueeze(0) - omega * epsilon
        linear_theta = (
            (z_transpose.unsqueeze(0) @ residual.unsqueeze(-1)).squeeze(-1)
            + sigma_inv_mu.unsqueeze(1)
        )
        chol = torch.linalg.cholesky(precision_theta)
        mean_theta = torch.cholesky_solve(
            linear_theta.unsqueeze(-1), chol
        ).squeeze(-1)
        noise = torch.linalg.solve_triangular(
            chol.transpose(-1, -2),
            torch.randn_like(mean_theta).unsqueeze(-1),
            upper=True,
        ).squeeze(-1)
        theta = mean_theta + noise

        theta_bar = theta.mean(dim=1)
        centered = theta - theta_bar.unsqueeze(1)
        kappa_n = cfg["niw_kappa0"] + users
        nu_n = nu0 + users
        offset = theta_bar - m0
        lambda_n = (
            lambda0.unsqueeze(0)
            + centered.transpose(-1, -2) @ centered
            + (cfg["niw_kappa0"] * users / kappa_n)
            * offset.unsqueeze(-1)
            * offset.unsqueeze(-2)
        )
        sigma = sample_inverse_wishart(lambda_n, nu_n, cfg["jitter"])
        m_n = (cfg["niw_kappa0"] * m0 + users * theta_bar) / kappa_n
        mu_chol = torch.linalg.cholesky(
            sigma / kappa_n + cfg["jitter"] * identity
        )
        mu = m_n + (mu_chol @ torch.randn_like(mu).unsqueeze(-1)).squeeze(-1)

        if iteration >= cfg["burnin"] and (
            iteration - cfg["burnin"]
        ) % cfg["thin"] == 0:
            mu_samples[:, save] = mu
            sigma_samples[:, save] = sigma
            theta_samples[:, save] = theta
            save += 1

    if save != samples:
        raise RuntimeError(f"retained {save} draws per chain, expected {samples}")

    return {
        "mu": mu_samples.reshape(chains * samples, d),
        "sigma": sigma_samples.reshape(chains * samples, d, d),
        "theta": theta_samples.reshape(chains * samples, users, d),
        "chains": chains,
        "samples_per_chain": samples,
    }


def adapt_new_users(mu, sigma, z_context, y_context, cfg, seed):
    if z_context.ndim != 3 or y_context.shape != z_context.shape[:2]:
        raise ValueError("context z must be [users, items, features]")
    if sigma.shape != (mu.shape[0], mu.shape[1], mu.shape[1]):
        raise ValueError("mu and sigma posterior draws must remain paired")
    if z_context.shape[-1] != mu.shape[-1]:
        raise ValueError("population and context feature dimensions differ")
    seed_all(seed, z_context.device)
    pg_rng = np.random.default_rng(seed) if cfg["pg_exact"] and mu.device.type == "cpu" else None
    test_users = z_context.shape[0]
    context_items = z_context.shape[1]
    draws, d = mu.shape
    identity = torch.eye(d, device=mu.device, dtype=mu.dtype)
    sigma_chol = torch.linalg.cholesky(sigma + cfg["jitter"] * identity)
    normal = torch.randn((test_users, draws, d), device=mu.device, dtype=mu.dtype)
    theta = mu.unsqueeze(0) + torch.einsum("mij,vmj->vmi", sigma_chol, normal)

    if context_items == 0 or cfg["newuser_iters"] == 0:
        return theta

    sigma_inv = torch.linalg.inv(sigma + cfg["jitter"] * identity)
    sigma_inv_mu = torch.einsum("mij,mj->mi", sigma_inv, mu)
    kappa = y_context - 0.5
    epsilon = torch.zeros(
        (test_users, draws, context_items), device=mu.device, dtype=mu.dtype
    )
    inv_eps = 0.0 if cfg["eps_var"] is None else 1.0 / cfg["eps_var"]

    for _ in range(cfg["newuser_iters"]):
        eta = torch.einsum("vmd,vnd->vmn", theta, z_context)
        omega = sample_pg(eta + epsilon, cfg, pg_rng)
        if cfg["eps_var"] is not None:
            precision_eps = omega + inv_eps
            mean_eps = (kappa[:, None] - omega * eta) / precision_eps
            epsilon = mean_eps + torch.randn_like(epsilon) / precision_eps.sqrt()

        precision_theta = torch.einsum(
            "vni,vmn,vnj->vmij", z_context, omega, z_context
        )
        precision_theta = (
            precision_theta
            + sigma_inv.unsqueeze(0)
            + cfg["jitter"] * identity
        )
        residual = kappa[:, None] - omega * epsilon
        linear_theta = torch.einsum(
            "vni,vmn->vmi", z_context, residual
        ) + sigma_inv_mu.unsqueeze(0)
        chol = torch.linalg.cholesky(precision_theta)
        mean_theta = torch.cholesky_solve(
            linear_theta.unsqueeze(-1), chol
        ).squeeze(-1)
        noise = torch.linalg.solve_triangular(
            chol.transpose(-1, -2),
            torch.randn_like(mean_theta).unsqueeze(-1),
            upper=True,
        ).squeeze(-1)
        theta = mean_theta + noise

    return theta


def population_predict(theta, z, draw_chunk):
    draws = theta.shape[0]
    probability_sum = torch.zeros(z.shape[:2], device=z.device, dtype=z.dtype)
    square_sum = torch.zeros_like(probability_sum)
    aleatoric_sum = torch.zeros_like(probability_sum)
    for start in range(0, draws, draw_chunk):
        current = theta[start : start + draw_chunk]
        probability = torch.sigmoid(torch.einsum("mud,und->mun", current, z))
        probability_sum += probability.sum(dim=0)
        square_sum += probability.square().sum(dim=0)
        aleatoric_sum += (probability * (1.0 - probability)).sum(dim=0)
    mean = probability_sum / draws
    epistemic = (square_sum / draws - mean.square()).clamp_min(0.0)
    aleatoric = aleatoric_sum / draws
    return mean, epistemic, aleatoric


def new_user_predict(theta, z, draw_chunk):
    draws = theta.shape[1]
    probability_sum = torch.zeros(z.shape[:2], device=z.device, dtype=z.dtype)
    square_sum = torch.zeros_like(probability_sum)
    aleatoric_sum = torch.zeros_like(probability_sum)
    for start in range(0, draws, draw_chunk):
        current = theta[:, start : start + draw_chunk]
        probability = torch.sigmoid(torch.einsum("vmd,vnd->vmn", current, z))
        probability_sum += probability.sum(dim=1)
        square_sum += probability.square().sum(dim=1)
        aleatoric_sum += (probability * (1.0 - probability)).sum(dim=1)
    mean = probability_sum / draws
    epistemic = (square_sum / draws - mean.square()).clamp_min(0.0)
    aleatoric = aleatoric_sum / draws
    return mean, epistemic, aleatoric


def auroc(y, probability):
    y = y.reshape(-1)
    probability = probability.reshape(-1)
    positives = y.sum()
    negatives = y.numel() - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = torch.argsort(probability, descending=True)
    sorted_y = y[order]
    true_positive = torch.cat(
        (torch.zeros(1, device=y.device, dtype=y.dtype), sorted_y.cumsum(0))
    )
    false_positive = torch.cat(
        (
            torch.zeros(1, device=y.device, dtype=y.dtype),
            (1.0 - sorted_y).cumsum(0),
        )
    )
    tpr = true_positive / positives
    fpr = false_positive / negatives
    return float(torch.trapz(tpr, fpr).detach().cpu())


def metrics(y, probability, epistemic, aleatoric):
    probability = probability.clamp(1e-7, 1.0 - 1e-7)
    lpd = y * probability.log() + (1.0 - y) * torch.log1p(-probability)
    return {
        "lpd": float(lpd.mean().detach().cpu()),
        "brier": float((probability - y).square().mean().detach().cpu()),
        "accuracy": float(
            ((probability >= 0.5) == (y >= 0.5)).to(y.dtype).mean().detach().cpu()
        ),
        "auroc": auroc(y, probability),
        "epistemic": float(epistemic.mean().detach().cpu()),
        "aleatoric": float(aleatoric.mean().detach().cpu()),
    }


def generate_raw(users, items, window, sensors, fs, dtype, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    time_axis = torch.arange(window, dtype=dtype).reshape(1, 1, window, 1) / fs
    shape = (users, items, 1, sensors)
    amplitude = torch.exp(0.30 * torch.randn(shape, generator=generator, dtype=dtype))
    frequency = 0.4 + 4.6 * torch.rand(shape, generator=generator, dtype=dtype)
    phase = 2.0 * math.pi * torch.rand(shape, generator=generator, dtype=dtype)
    offset = 0.20 * torch.randn(shape, generator=generator, dtype=dtype)
    user_offset = 0.20 * torch.randn(
        (users, 1, 1, sensors), generator=generator, dtype=dtype
    )
    raw = offset + user_offset
    raw = raw + amplitude * torch.sin(2.0 * math.pi * frequency * time_axis + phase)
    raw = raw + 0.25 * amplitude * torch.sin(
        4.0 * math.pi * frequency * time_axis + 0.5 * phase
    )
    raw = raw + 0.20 * torch.randn(
        (users, items, window, sensors), generator=generator, dtype=dtype
    )
    return raw.contiguous()


def generate_feedback(z_train, z_test, seed, logit_scale):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    train_users, _, d = z_train.shape
    test_users = z_test.shape[0]
    population_mean = 0.12 * torch.randn(d, generator=generator, dtype=z_train.dtype)
    population_mean[0] = 0.0
    rank = min(4, d)
    loading = 0.10 * torch.randn(
        (d, rank), generator=generator, dtype=z_train.dtype
    )
    shared = torch.randn(
        (train_users + test_users, rank), generator=generator, dtype=z_train.dtype
    ) @ loading.T
    independent = 0.18 * torch.randn(
        (train_users + test_users, d), generator=generator, dtype=z_train.dtype
    )
    theta = population_mean + shared + independent

    eta_train = torch.einsum("und,ud->un", z_train, theta[:train_users])
    eta_test = torch.einsum("vnd,vd->vn", z_test, theta[train_users:])
    scale = torch.cat((eta_train.flatten(), eta_test.flatten())).std(
        unbiased=False
    ).clamp_min(1e-6) / logit_scale
    theta = theta / scale
    theta_train = theta[:train_users]
    theta_test = theta[train_users:]
    eta_train = torch.einsum("und,ud->un", z_train, theta_train)
    eta_test = torch.einsum("vnd,vd->vn", z_test, theta_test)
    y_train = torch.bernoulli(torch.sigmoid(eta_train), generator=generator)
    y_test = torch.bernoulli(torch.sigmoid(eta_test), generator=generator)
    for user in range(train_users):
        if y_train[user].sum() == 0:
            y_train[user, 0] = 1.0
        elif y_train[user].sum() == y_train.shape[1]:
            y_train[user, 0] = 0.0
    return y_train, y_test, theta_train, theta_test


def prepare_toy_data(args, train_users, train_items, window, sensors, dtype, seed):
    max_context = max(args.context_sizes)
    test_items = max_context + args.query_items
    raw_train = generate_raw(
        train_users, train_items, window, sensors, args.fs, dtype, seed
    )
    raw_test = generate_raw(
        args.test_users, test_items, window, sensors, args.fs, dtype, seed + 1
    )
    manual_stats = make_manual_stats(sensors, args.stats_preset, args.stats)
    canonical = ManualFeatures(manual_stats, args.fs)
    z_train = canonical.fit_transform(raw_train)
    z_test = canonical.transform(raw_test)
    y_train, y_test, theta_train, theta_test = generate_feedback(
        z_train, z_test, seed + 2, args.logit_scale
    )
    return {
        "raw_train": raw_train,
        "raw_test": raw_test,
        "y_train": y_train,
        "y_test": y_test,
        "theta_train": theta_train,
        "theta_test": theta_test,
        "n_features": z_train.shape[-1],
        "feature_names": canonical.feature_names,
    }


def run_device(data, args, cfg, sensors, device, seed):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def transfer():
        return {
            key: value.to(device)
            if torch.is_tensor(value)
            else value
            for key, value in data.items()
        }

    device_data, transfer_seconds = timed(device, transfer)

    def extract_features():
        manual_stats = make_manual_stats(sensors, args.stats_preset, args.stats)
        extractor = ManualFeatures(manual_stats, args.fs)
        z_train = extractor.fit_transform(device_data["raw_train"])
        z_test = extractor.transform(device_data["raw_test"])
        return z_train, z_test

    (z_train, z_test), feature_seconds = timed(device, extract_features)
    posterior, population_seconds = timed(
        device,
        lambda: fit_population(z_train, device_data["y_train"], cfg, seed + 10),
    )

    (train_probability, train_epistemic, train_aleatoric), train_predict_seconds = timed(
        device,
        lambda: population_predict(
            posterior["theta"], z_train, args.draw_chunk
        ),
    )
    train_metrics = metrics(
        device_data["y_train"],
        train_probability,
        train_epistemic,
        train_aleatoric,
    )
    theta_train_mean = posterior["theta"].mean(dim=0)
    train_metrics["theta_rmse"] = float(
        (theta_train_mean - device_data["theta_train"])
        .square()
        .mean()
        .sqrt()
        .detach()
        .cpu()
    )

    max_context = max(args.context_sizes)
    z_query = z_test[:, max_context : max_context + args.query_items]
    y_query = device_data["y_test"][:, max_context : max_context + args.query_items]
    test_results = {}
    adaptation_total = 0.0
    prediction_total = 0.0

    for context_size in args.context_sizes:
        z_context = z_test[:, :context_size]
        y_context = device_data["y_test"][:, :context_size]
        theta_new, adaptation_seconds = timed(
            device,
            lambda context_size=context_size: adapt_new_users(
                posterior["mu"],
                posterior["sigma"],
                z_context,
                y_context,
                cfg,
                seed + 1000,
            ),
        )
        (probability, epistemic, aleatoric), prediction_seconds = timed(
            device,
            lambda: new_user_predict(theta_new, z_query, args.draw_chunk),
        )
        aggregate = metrics(y_query, probability, epistemic, aleatoric)
        aggregate["theta_rmse"] = float(
            (theta_new.mean(dim=1) - device_data["theta_test"])
            .square()
            .mean()
            .sqrt()
            .detach()
            .cpu()
        )
        per_user = []
        for user in range(args.test_users):
            user_metrics = metrics(
                y_query[user],
                probability[user],
                epistemic[user],
                aleatoric[user],
            )
            user_metrics["user"] = user
            per_user.append(user_metrics)
        test_results[str(context_size)] = {
            "aggregate": aggregate,
            "per_user": per_user,
            "adaptation_seconds": adaptation_seconds,
            "prediction_seconds": prediction_seconds,
        }
        adaptation_total += adaptation_seconds
        prediction_total += prediction_seconds

    peak_memory = (
        torch.cuda.max_memory_allocated(device) / 2**20
        if device.type == "cuda"
        else float("nan")
    )
    total_seconds = (
        transfer_seconds
        + feature_seconds
        + population_seconds
        + train_predict_seconds
        + adaptation_total
        + prediction_total
    )
    result = {
        "device": str(device),
        "posterior_draws": posterior["theta"].shape[0],
        "timing": {
            "transfer": transfer_seconds,
            "features": feature_seconds,
            "population": population_seconds,
            "train_prediction": train_predict_seconds,
            "newuser_adaptation": adaptation_total,
            "newuser_prediction": prediction_total,
            "total": total_seconds,
        },
        "train": train_metrics,
        "test": test_results,
        "peak_memory_mib": peak_memory,
        "z_train": z_train.detach().cpu(),
        "z_test": z_test.detach().cpu(),
    }
    return result


def run_repeated(data, args, cfg, sensors, device, seed):
    results = [
        run_device(data, args, cfg, sensors, device, seed)
        for _ in range(args.repeats)
    ]
    median = statistics.median(result["timing"]["total"] for result in results)
    return min(results, key=lambda result: abs(result["timing"]["total"] - median))


def flatten_row(structure, cfg, cpu, gpu, context_sizes):
    row = dict(structure)
    row.update(
        {
            "pg_terms": cfg["pg_terms"],
            "chains": cfg["chains"],
            "burnin": cfg["burnin"],
            "samples_per_chain": cfg["samples"],
            "posterior_draws": cfg["chains"] * cfg["samples"],
            "newuser_iters": cfg["newuser_iters"],
            "cpu_pg": "exact" if cfg["pg_exact"] else "truncated",
            "cpu_total_s": cpu["timing"]["total"],
            "cpu_feature_s": cpu["timing"]["features"],
            "cpu_population_s": cpu["timing"]["population"],
            "cpu_adaptation_s": cpu["timing"]["newuser_adaptation"],
            "gpu_total_s": float("nan") if gpu is None else gpu["timing"]["total"],
            "gpu_feature_s": float("nan") if gpu is None else gpu["timing"]["features"],
            "gpu_population_s": float("nan") if gpu is None else gpu["timing"]["population"],
            "gpu_adaptation_s": float("nan") if gpu is None else gpu["timing"]["newuser_adaptation"],
            "end_to_end_speedup": float("nan")
            if gpu is None
            else cpu["timing"]["total"] / gpu["timing"]["total"],
            "feature_max_abs_diff": float("nan")
            if gpu is None
            else float(
                (cpu["z_train"] - gpu["z_train"]).abs().max().detach().cpu()
            ),
        }
    )
    for context_size in context_sizes:
        key = str(context_size)
        row[f"cpu_ctx{context_size}_adaptation_s"] = cpu["test"][key]["adaptation_seconds"]
        row[f"cpu_ctx{context_size}_prediction_s"] = cpu["test"][key]["prediction_seconds"]
        row[f"gpu_ctx{context_size}_adaptation_s"] = (
            float("nan") if gpu is None else gpu["test"][key]["adaptation_seconds"]
        )
        row[f"gpu_ctx{context_size}_prediction_s"] = (
            float("nan") if gpu is None else gpu["test"][key]["prediction_seconds"]
        )
        for metric in (
            "lpd",
            "brier",
            "accuracy",
            "auroc",
            "theta_rmse",
            "epistemic",
            "aleatoric",
        ):
            row[f"cpu_ctx{context_size}_{metric}"] = cpu["test"][key]["aggregate"][metric]
            row[f"gpu_ctx{context_size}_{metric}"] = (
                float("nan")
                if gpu is None
                else gpu["test"][key]["aggregate"][metric]
            )
    return row


def strip_tensors(result):
    return {key: value for key, value in result.items() if key not in {"z_train", "z_test"}}


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_csv(rows, path):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


STAGE_COLORS = {"feature": "#4C72B0", "population": "#DD8452", "adaptation": "#55A868"}


def context_sizes_in(rows):
    return sorted(
        int(key.split("ctx")[1].split("_")[0])
        for key in rows[0]
        if key.startswith("cpu_ctx") and key.endswith("_lpd")
    )


def plot_all(rows, pg_rows, output_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plt.rcParams.update({          # 발표용: 기본 대비 글자 크기 +2.5
        "font.size": 12.5,
        "axes.titlesize": 14.5,
        "axes.labelsize": 12.5,
        "legend.fontsize": 12.5,
        "xtick.labelsize": 12.5,
        "ytick.labelsize": 12.5,
    })
    valid = [row for row in rows if math.isfinite(row["end_to_end_speedup"])]
    if valid:
        contexts = context_sizes_in(rows)
        plot_speedup(plt, valid, output_dir)
        plot_time_breakdown(plt, valid, output_dir)
        plot_agreement(plt, valid, max(contexts), output_dir)
        if len(contexts) > 1:
            plot_context_curve(plt, valid, contexts, output_dir)
    if pg_rows:
        plot_pg_validation(plt, pg_rows, output_dir)


def plot_speedup(plt, rows, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) chains lever: x=chains, one line per (U, N) problem
    structures = {}
    for row in rows:
        structures.setdefault((row["train_users"], row["train_items_per_user"]), []).append(row)
    for (users, items), pts in sorted(structures.items()):
        pts.sort(key=lambda row: row["chains"])
        axes[0].plot(
            [row["chains"] for row in pts],
            [row["end_to_end_speedup"] for row in pts],
            marker="o",
            label=f"U={users}, N={items}",
        )
    axes[0].axhline(1.0, color="crimson", linestyle="--", linewidth=1.2)
    axes[0].set_xlabel("chains (parallel MCMC chains)")
    axes[0].set_ylabel("speedup (CPU time / GPU time)")
    axes[0].set_title("(a) Speedup vs chains")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # (b) scale lever: x=observations at fixed chains, one line per U (no cross-U break)
    base_chains = min(row["chains"] for row in rows)
    per_user = {}
    for row in rows:
        if row["chains"] == base_chains:
            per_user.setdefault(row["train_users"], []).append(row)
    for users, pts in sorted(per_user.items()):
        pts.sort(key=lambda row: row["observations"])
        axes[1].plot(
            [row["observations"] for row in pts],
            [row["end_to_end_speedup"] for row in pts],
            marker="o",
            label=f"U={users}",
        )
    axes[1].axhline(1.0, color="crimson", linestyle="--", linewidth=1.2)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("training observations (U x N)")
    axes[1].set_ylabel("speedup (CPU time / GPU time)")
    axes[1].set_title(f"(b) Speedup vs problem size (chains={base_chains})")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "speedup.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_time_breakdown(plt, rows, output_dir):
    row = max(rows, key=lambda row: row["observations"])
    fig, ax = plt.subplots(figsize=(6, 5))
    bottom_cpu = 0.0
    bottom_gpu = 0.0
    for stage in ("feature", "population", "adaptation"):
        cpu_value = row[f"cpu_{stage}_s"]
        gpu_value = row[f"gpu_{stage}_s"]
        ax.bar(0, cpu_value, bottom=bottom_cpu, color=STAGE_COLORS[stage], label=stage)
        ax.bar(1, gpu_value, bottom=bottom_gpu, color=STAGE_COLORS[stage])
        bottom_cpu += cpu_value
        bottom_gpu += gpu_value
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["CPU", "GPU"])
    ax.set_ylabel("seconds")
    ax.set_title(
        f"Time budget  (U={row['train_users']}, N={row['train_items_per_user']}, "
        f"d={row['n_features']}, chains={row['chains']})"
    )
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / "time_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_agreement(plt, rows, context, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric, name in (
        (axes[0], "lpd", "new-user LPD"),
        (axes[1], "auroc", "new-user AUROC"),
    ):
        pairs = [
            (row[f"cpu_ctx{context}_{metric}"], row[f"gpu_ctx{context}_{metric}"])
            for row in rows
            if math.isfinite(row[f"cpu_ctx{context}_{metric}"])
            and math.isfinite(row[f"gpu_ctx{context}_{metric}"])
        ]
        if not pairs:
            continue
        xs = [pair[0] for pair in pairs]
        ys = [pair[1] for pair in pairs]
        ax.scatter(xs, ys)
        low = min(min(xs), min(ys))
        high = max(max(xs), max(ys))
        ax.plot([low, high], [low, high], color="gray", linestyle="--")
        ax.set_xlabel(f"CPU {name}")
        ax.set_ylabel(f"GPU {name}")
        ax.set_title(f"{name} agreement (context={context})")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "agreement.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_context_curve(plt, rows, contexts, output_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    for metric in ("lpd", "auroc"):
        means = []
        for context in contexts:
            values = [
                row[f"cpu_ctx{context}_{metric}"]
                for row in rows
                if math.isfinite(row[f"cpu_ctx{context}_{metric}"])
            ]
            means.append(sum(values) / len(values) if values else float("nan"))
        ax.plot(contexts, means, marker="o", label=metric)
    ax.set_xlabel("new-user context size")
    ax.set_ylabel("metric (mean over configs, CPU)")
    ax.set_title("Personalization vs context size")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "context_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pg_validation(plt, pg_rows, output_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    groups = {}
    for row in pg_rows:
        groups.setdefault((row["device"], row["terms"]), []).append(row)
    for (device, terms), points in sorted(groups.items()):
        points.sort(key=lambda row: row["psi"])
        ax.plot(
            [point["psi"] for point in points],
            [point["relative_error"] for point in points],
            marker="o",
            linestyle="--" if "cuda" in device else "-",
            label=f"{device.split(':')[0]} K={terms}",
        )
    ax.set_yscale("log")
    ax.set_xlabel("psi (linear predictor)")
    ax.set_ylabel("relative error vs analytic mean")
    ax.set_title("Truncated PG(1, psi) mean accuracy")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "pg_validation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def validate_pg(args, devices, dtype):
    rows = []
    for device in devices:
        for terms in args.pg_terms:
            for value in (0.0, 0.5, 1.0, 3.0, 5.0):
                seed_all(args.seed, device)
                psi = torch.full(
                    (args.pg_validation_samples,), value, device=device, dtype=dtype
                )
                draw, seconds = timed(
                    device,
                    lambda: pg_truncated(
                        psi, terms, min(args.pg_chunk, terms)
                    ),
                )
                analytic = float(pg_mean(psi[:1]).detach().cpu())
                empirical = float(draw.mean().detach().cpu())
                rows.append(
                    {
                        "device": str(device),
                        "terms": terms,
                        "psi": value,
                        "analytic_mean": analytic,
                        "sample_mean": empirical,
                        "relative_error": abs(empirical - analytic) / analytic,
                        "seconds": seconds,
                    }
                )
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-users", nargs="+", type=int, default=[5])
    parser.add_argument("--train-items", nargs="+", type=int, default=[200])
    parser.add_argument("--test-users", type=int, default=5)
    parser.add_argument("--context-sizes", nargs="+", type=int, default=[0, 5, 20])
    parser.add_argument("--query-items", type=int, default=100)
    parser.add_argument("--newuser-iters", type=int, default=20)
    parser.add_argument("--window-lengths", nargs="+", type=int, default=[128])
    parser.add_argument("--sensors", nargs="+", type=int, default=[5])
    parser.add_argument(
        "--stats-preset", choices=("basic", "current", "full"), default="current"
    )
    parser.add_argument("--stats", nargs="+")
    parser.add_argument("--fs", type=float, default=100.0)
    parser.add_argument("--logit-scale", type=float, default=1.5)
    parser.add_argument("--burnin", type=int, default=500)
    parser.add_argument("--samples", type=int, default=1500)
    parser.add_argument("--thin", type=int, default=1)
    parser.add_argument("--chains", nargs="+", type=int, default=[1])
    parser.add_argument("--pg-terms", nargs="+", type=int, default=[50])
    parser.add_argument("--pg-chunk", type=int, default=25)
    parser.add_argument("--niw-kappa0", type=float, default=1.0)
    parser.add_argument("--niw-nu0", type=float)
    parser.add_argument("--niw-lambda0-scale", type=float, default=1.0)
    parser.add_argument("--eps-var", type=float)
    parser.add_argument("--jitter", type=float, default=1e-5)
    parser.add_argument("--draw-chunk", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-raw-gib", type=float, default=4.0)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--cuda-device", default="cuda:0")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--cpu-truncated", action="store_true")
    parser.add_argument("--validate-pg", action="store_true")
    parser.add_argument("--pg-validation-samples", type=int, default=20000)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/toy_result"))
    return parser.parse_args()


def apply_quick(args):
    if not args.quick:
        return
    args.train_users = [3]
    args.train_items = [30]
    args.test_users = 2
    args.context_sizes = [0, 3]
    args.query_items = 12
    args.newuser_iters = 3
    args.window_lengths = [32]
    args.sensors = [2]
    args.stats = ["mean", "std", "p2p", "rms"]
    args.burnin = 3
    args.samples = 4
    args.chains = [1]
    args.pg_terms = [8]
    args.pg_chunk = 4
    args.draw_chunk = 8
    args.repeats = 1
    args.pg_validation_samples = 1000


def main():
    args = parse_args()
    apply_quick(args)
    args.context_sizes = sorted(set(args.context_sizes))
    if min(args.context_sizes) < 0:
        raise ValueError("context sizes must be nonnegative")
    if args.samples < 1 or args.thin < 1 or args.newuser_iters < 0:
        raise ValueError("samples/thin must be positive and newuser-iters nonnegative")
    if args.repeats < 1 or args.burnin < 0:
        raise ValueError("repeats must be positive and burnin nonnegative")
    if args.test_users < 1 or args.query_items < 1:
        raise ValueError("test-users and query-items must be positive")
    if args.fs <= 0 or args.logit_scale <= 0 or args.max_raw_gib <= 0:
        raise ValueError("fs, logit-scale, and max-raw-gib must be positive")
    if args.pg_chunk < 1 or args.draw_chunk < 1:
        raise ValueError("PG and posterior-draw chunks must be positive")
    if args.eps_var is not None and args.eps_var <= 0:
        raise ValueError("eps-var must be positive when enabled")
    if min(args.train_users + args.train_items + args.window_lengths + args.sensors) < 1:
        raise ValueError("all structural sweep values must be positive")
    if min(args.pg_terms + args.chains) < 1:
        raise ValueError("PG terms and chains must be positive")

    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    cuda_available = torch.cuda.is_available() and not args.cpu_only
    if args.require_cuda and not cuda_available:
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    devices = [torch.device("cpu")]
    if cuda_available:
        devices.append(torch.device(args.cuda_device))
    for device in devices:
        warmup_device(device, dtype)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(vars(args)), handle, indent=2, ensure_ascii=False)
    pg_rows = []
    if args.validate_pg:
        pg_rows = validate_pg(args, devices, dtype)
        with (args.output_dir / "pg_validation.json").open("w", encoding="utf-8") as handle:
            json.dump(json_safe(pg_rows), handle, indent=2)

    rows = []
    details = []
    structures = itertools.product(
        args.train_users, args.train_items, args.window_lengths, args.sensors
    )
    for train_users, train_items, window, sensors in structures:
        test_items = max(args.context_sizes) + args.query_items
        raw_elements = (
            train_users * train_items + args.test_users * test_items
        ) * window * sensors
        raw_gib = raw_elements * torch.empty((), dtype=dtype).element_size() / 2**30
        if raw_gib > args.max_raw_gib:
            print(
                f"\n[skip] U={train_users}, N/U={train_items}, T={window}, "
                f"sensors={sensors}: raw={raw_gib:.2f} GiB > {args.max_raw_gib:.2f} GiB"
            )
            continue
        structure_seed = (
            args.seed
            + train_users * 1_000_003
            + train_items * 101
            + window * 17
            + sensors
        )
        data = prepare_toy_data(
            args, train_users, train_items, window, sensors, dtype, structure_seed
        )
        n_features = data["n_features"]
        print(
            f"\n[data] train U={train_users}, N/U={train_items}, "
            f"test U={args.test_users}, d={n_features}, T={window}, sensors={sensors}"
        )
        for pg_terms, chains in itertools.product(args.pg_terms, args.chains):
            cfg = {
                "burnin": args.burnin,
                "samples": args.samples,
                "thin": args.thin,
                "chains": chains,
                "pg_terms": pg_terms,
                "pg_chunk": min(args.pg_chunk, pg_terms),
                "niw_kappa0": args.niw_kappa0,
                "niw_nu0": args.niw_nu0,
                "niw_lambda0_scale": args.niw_lambda0_scale,
                "eps_var": args.eps_var,
                "jitter": args.jitter,
                "newuser_iters": args.newuser_iters,
                "pg_exact": not args.cpu_truncated,
            }
            print(
                f"  [run] K={pg_terms}, chains={chains}, "
                f"steps={args.burnin + args.samples * args.thin}, "
                f"draws={chains * args.samples}, L={args.newuser_iters}"
            )
            cpu = run_repeated(
                data, args, cfg, sensors, torch.device("cpu"), structure_seed + 100
            )
            gpu = None
            if cuda_available:
                gpu = run_repeated(
                    data,
                    args,
                    cfg,
                    sensors,
                    torch.device(args.cuda_device),
                    structure_seed + 100,
                )

            structure = {
                "train_users": train_users,
                "train_items_per_user": train_items,
                "observations": train_users * train_items,
                "test_users": args.test_users,
                "query_items_per_user": args.query_items,
                "window_length": window,
                "sensors": sensors,
                "n_features": n_features,
                "dtype": args.dtype,
                "stats": "custom" if args.stats else args.stats_preset,
                "raw_gib": raw_gib,
                "timing_repeats": args.repeats,
            }
            rows.append(flatten_row(structure, cfg, cpu, gpu, args.context_sizes))
            details.append(
                {
                    "structure": structure,
                    "config": cfg,
                    "feature_names": data["feature_names"],
                    "cpu": strip_tensors(cpu),
                    "gpu": None if gpu is None else strip_tensors(gpu),
                }
            )
            last = str(max(args.context_sizes))
            if gpu is None:
                print(
                    f"    CPU total={cpu['timing']['total']:.3f}s, "
                    f"test LPD={cpu['test'][last]['aggregate']['lpd']:+.4f}"
                )
            else:
                speedup = cpu["timing"]["total"] / gpu["timing"]["total"]
                print(
                    f"    CPU={cpu['timing']['total']:.3f}s, "
                    f"CUDA={gpu['timing']['total']:.3f}s, speedup={speedup:.2f}x"
                )
                print(
                    f"    context={last}: CPU LPD={cpu['test'][last]['aggregate']['lpd']:+.4f}, "
                    f"CUDA LPD={gpu['test'][last]['aggregate']['lpd']:+.4f}"
                )

            del cpu, gpu
            if cuda_available:
                torch.cuda.empty_cache()

    write_csv(rows, args.output_dir / "benchmark.csv")
    with (args.output_dir / "details.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(details), handle, indent=2, ensure_ascii=False)
    plot_all(rows, pg_rows, args.output_dir)
    print(f"\nresults: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
