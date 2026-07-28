"""CPU-vs-CUDA benchmark for manual features and hierarchical PG-Gibbs.

This is a standalone toy benchmark that preserves the computational structure
of the fully Bayesian reward model:

    raw trajectories -> manual sensor statistics -> standardization (+ bias)
    y_ui ~ Bernoulli(sigmoid(theta_u^T z_ui))
    theta_u | mu, Sigma ~ Normal(mu, Sigma)
    (mu, Sigma) ~ Normal-Inverse-Wishart

Both CPU and CUDA runs use the same K-term truncated-series approximation to
PG(1, psi), so the timing comparison changes hardware rather than inference
algorithms.  PCA, autoencoders, feature selection, CSV parsing, and plotting
from the main project are intentionally outside the timed pipeline.

The timed end-to-end GPU path includes host-to-device transfer, manual feature
extraction, standardization, and population Gibbs sampling. Synthetic data and
label generation are excluded from both CPU and GPU timings.

Examples
--------
Quick smoke test:

    python gpu_feature_pg_benchmark.py --quick --output-dir bench_quick

Sweep the likely crossover region:

python gpu_feature_pg_benchmark.py --users 5 10 20 --items-per-user 100 500 1000 2000 --window-lengths 128 512 --sensor-counts 5 --pg-terms 25 50 100 --chains 1 4 --burnin 5 --samples 15 --stats-preset current --dtype float32

Validate the truncated PG sample mean before benchmarking:

    python gpu_feature_pg_benchmark.py --quick --validate-pg

Notes
-----
* A retained draw from one Markov chain cannot be parallelized over Gibbs
  iteration. ``--chains`` runs genuinely independent chains in parallel.
* The inverse-Wishart update is sampled with a Bartlett decomposition on both
  CPU and CUDA, avoiding a SciPy/CPU round trip in the CUDA loop.
* ``float64`` is useful for numerical validation. ``float32`` is usually the
  meaningful throughput benchmark, but posterior quality should be checked.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
    raise SystemExit(
        "PyTorch is required. Run this script inside the project's PyTorch/CUDA "
        "environment (for example: uv run python gpu_feature_pg_benchmark.py ...)."
    ) from exc


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

BASIC_STATS = (
    "mean",
    "std",
    "abs_peak",
    "energy",
    "impulse_abs",
    "crest",
)

# Repeats the five-channel pattern in the current project when more toy
# sensors are requested. Duplicate (sensor, statistic) pairs are removed,
# matching the main Features class.
CURRENT_STATS_PATTERN = (
    ("abs_peak_deriv", "p2p_deriv", "rms_deriv", "mean", "std", "abs_peak", "energy", "impulse_abs", "crest"),
    ("abs_peak_deriv", "p2p_deriv", "rms_deriv", "mean", "std", "abs_peak", "energy", "impulse_abs", "crest"),
    (
        "abs_peak",
        "p2p",
        "rms",
        "wrms_z",
        "vdv",
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
        "vdv",
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
        "mean",
        "std",
        "abs_peak",
        "energy",
        "impulse_abs",
        "crest",
    ),
)


# run_fully_bayesian.py Config에 맞춘 프리셋 (--fb-preset로 활성화).
# population fit(train) 타이밍 비교용. new-user 적응(newuser_n_iters)은 범위 밖 (main의 NOTE 참고).
# 실제 데이터에 맞춰 아래 숫자만 조정하면 된다.
FB_USERS = 5                                    # run_fully_bayesian train 드라이버 수
FB_ITEMS_PER_USER = 200                         # 드라이버당 세그먼트 수 (대략치 — 실제 데이터에 맞춰 조정)
FB_WINDOW = 200                                 # around=(-2,2)+downsample 윈도우 근사 (fs=50 기준)
FB_SENSORS = 3                                  # 활성 채널 수 (Bounce/VerAccel/LongAccel)
FB_STATS = ("p2p", "rms", "std", "abs_peak")    # 채널당 4 stat -> d=3*4+bias=13 (Config와 동일)
FB_PG_TERMS = 100                               # 실제 모델은 정확한 polyagamma; 토이는 truncated K항
FB_CHAINS = 1
FB_BURNIN = 500
FB_SAMPLES = 1500
FB_THIN = 1
FB_NIW_KAPPA0 = 1.0
FB_NIW_LAMBDA0_SCALE = 1.0
FB_EPS_VAR = None
FB_DTYPE = "float32"


def _deduplicate(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def make_manual_stats(
    sensor_count: int,
    preset: str,
    custom_stats: Sequence[str] | None = None,
) -> OrderedDict[str, tuple[str, ...]]:
    """Construct the same sensor -> statistic mapping used by manual features."""
    if sensor_count < 1:
        raise ValueError("sensor_count must be positive")
    unknown = set(custom_stats or ()) - set(ALL_STATS)
    if unknown:
        raise ValueError(f"Unknown statistics: {sorted(unknown)}")

    result: OrderedDict[str, tuple[str, ...]] = OrderedDict()
    for sensor in range(sensor_count):
        name = f"sensor_{sensor}"
        if custom_stats:
            methods = tuple(custom_stats)
        elif preset == "basic":
            methods = BASIC_STATS
        elif preset == "current":
            methods = CURRENT_STATS_PATTERN[sensor % len(CURRENT_STATS_PATTERN)]
        elif preset == "full":
            methods = ALL_STATS
        else:
            raise ValueError("stats preset must be 'basic', 'current', or 'full'")
        result[name] = _deduplicate(methods)
    return result


def _central_derivative(x: torch.Tensor, fs: float) -> torch.Tensor:
    """NumPy-gradient-compatible first derivative along the time axis (dim=2)."""
    if x.shape[2] < 2:
        return torch.zeros_like(x)
    dx = torch.empty_like(x)
    dx[:, :, 0, :] = (x[:, :, 1, :] - x[:, :, 0, :]) * fs
    dx[:, :, -1, :] = (x[:, :, -1, :] - x[:, :, -2, :]) * fs
    if x.shape[2] > 2:
        dx[:, :, 1:-1, :] = (x[:, :, 2:, :] - x[:, :, :-2, :]) * (0.5 * fs)
    return dx


def _iso_weight(f: torch.Tensor, kind: str) -> torch.Tensor:
    """ISO 2631 magnitude weights matching the NumPy feature implementation."""
    pi = math.pi
    s = (2j * pi) * f
    w1, w2 = 2 * pi * 0.4, 2 * pi * 100.0
    h_high = s.square() / (s.square() + (w1 / 0.7071) * s + w1**2)
    h_low = w2**2 / (s.square() + (w2 / 0.7071) * s + w2**2)

    if kind == "wk":
        w3 = 2 * pi * 12.5
        w4, q4 = 2 * pi * 12.5, 0.63
        w5, q5 = 2 * pi * 2.37, 0.91
        w6, q6 = 2 * pi * 3.35, 0.91
        h_transition = (1 + s / w3) / (1 + s / (q4 * w4) + (s / w4).square())
        h_step = ((w6 / w5) ** 2) * (
            1 + s / (q5 * w5) + (s / w5).square()
        ) / (1 + s / (q6 * w6) + (s / w6).square())
        return torch.abs(h_high * h_low * h_transition * h_step)

    if kind == "wd":
        w4, q4 = 2 * pi * 2.0, 0.63
    elif kind == "we":
        w4, q4 = 2 * pi * 1.0, 0.63
    else:
        raise ValueError(f"Unknown ISO weight: {kind}")
    h_transition = (1 + s / w4) / (1 + s / (q4 * w4) + (s / w4).square())
    return torch.abs(h_high * h_low * h_transition)


class TorchManualFeatures:
    """Vectorized CPU/CUDA manual feature extractor.

    Input tensors have shape [users, items, time, sensors]. Intermediate
    quantities such as x**2, the derivative, and FFT are computed at most once
    and reused across requested statistics.
    """

    def __init__(
        self,
        manual_stats: OrderedDict[str, Sequence[str]],
        fs: float,
        standardize: bool = True,
        include_bias: bool = True,
    ) -> None:
        self.manual_stats = manual_stats
        self.channels = list(manual_stats)
        self.fs = float(fs)
        self.standardize = bool(standardize)
        self.include_bias = bool(include_bias)

        seen: set[tuple[str, str]] = set()
        self.pairs: list[tuple[int, str]] = []
        names: list[str] = []
        groups: list[str] = []
        for channel_idx, (channel, methods) in enumerate(manual_stats.items()):
            for method in methods:
                if method not in ALL_STATS:
                    raise KeyError(f"Unknown manual statistic: {method}")
                key = (channel, method)
                if key in seen:
                    continue
                seen.add(key)
                self.pairs.append((channel_idx, method))
                names.append(f"{channel}__{method}")
                groups.append(channel)

        bias = ["bias"] if include_bias else []
        self.feature_names = bias + names
        self.groups = bias + groups
        self.mu: torch.Tensor | None = None
        self.sd: torch.Tensor | None = None

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("x must have shape [users, items, time, sensors]")
        if x.shape[-1] != len(self.channels):
            raise ValueError("The tensor sensor dimension does not match manual_stats")

        needed = {method for _, method in self.pairs}
        method_channels = {
            method: sorted({channel for channel, name in self.pairs if name == method})
            for method in needed
        }
        x_fft_methods = {"wrms_z", "wrms_xy", "sigma_sd", "band_low", "band_mid"}
        derivative_fft_methods = {"wrms_rot", "wrms_z_d"}
        derivative_methods = {
            "rms_deriv",
            "p2p_deriv",
            "abs_peak_deriv",
            "vdv_deriv",
            "wrms_rot",
            "wrms_z_d",
        }
        derivative_channels = sorted(
            {
                channel
                for method in derivative_methods & needed
                for channel in method_channels[method]
            }
        )
        x_fft_channels = sorted(
            {channel for method in x_fft_methods & needed for channel in method_channels[method]}
        )
        derivative_fft_channels = sorted(
            {
                channel
                for method in derivative_fft_methods & needed
                for channel in method_channels[method]
            }
        )
        eps = torch.as_tensor(1e-12, device=x.device, dtype=x.dtype)
        values: dict[str, torch.Tensor] = {}

        x_abs: torch.Tensor | None = None
        x2: torch.Tensor | None = None
        mean: torch.Tensor | None = None
        derivative: torch.Tensor | None = None
        derivative2: torch.Tensor | None = None
        spectrum: torch.Tensor | None = None
        derivative_spectrum: torch.Tensor | None = None
        power: torch.Tensor | None = None
        frequencies: torch.Tensor | None = None
        iso_weights: dict[str, torch.Tensor] = {}

        def channel_index(channels: Sequence[int]) -> torch.Tensor:
            return torch.as_tensor(channels, device=x.device, dtype=torch.long)

        def expand_channels(method: str, selected: torch.Tensor) -> torch.Tensor:
            full = torch.zeros((*x.shape[:2], x.shape[-1]), device=x.device, dtype=x.dtype)
            return full.index_copy(-1, channel_index(method_channels[method]), selected)

        def get_abs() -> torch.Tensor:
            nonlocal x_abs
            if x_abs is None:
                x_abs = x.abs()
            return x_abs

        def get_x2() -> torch.Tensor:
            nonlocal x2
            if x2 is None:
                x2 = x.square()
            return x2

        def get_mean() -> torch.Tensor:
            nonlocal mean
            if mean is None:
                mean = x.mean(dim=2)
            return mean

        def get_derivative() -> torch.Tensor:
            nonlocal derivative
            if derivative is None:
                selected = _central_derivative(
                    x.index_select(-1, channel_index(derivative_channels)), self.fs
                )
                derivative = torch.zeros_like(x).index_copy(
                    -1, channel_index(derivative_channels), selected
                )
            return derivative

        def get_derivative2() -> torch.Tensor:
            nonlocal derivative2
            if derivative2 is None:
                derivative2 = get_derivative().square()
            return derivative2

        def get_frequencies() -> torch.Tensor:
            nonlocal frequencies
            if frequencies is None:
                frequencies = torch.fft.rfftfreq(
                    x.shape[2], d=1.0 / self.fs, device=x.device, dtype=x.dtype
                )
            return frequencies

        def get_spectrum(source: str = "x") -> torch.Tensor:
            nonlocal spectrum, derivative_spectrum
            if source == "x":
                if spectrum is None:
                    spectrum = torch.fft.rfft(
                        x.index_select(-1, channel_index(x_fft_channels)), dim=2
                    )
                return spectrum
            if source == "derivative":
                if derivative_spectrum is None:
                    derivative_spectrum = torch.fft.rfft(
                        get_derivative().index_select(
                            -1, channel_index(derivative_fft_channels)
                        ),
                        dim=2,
                    )
                return derivative_spectrum
            raise ValueError(source)

        def select_spectrum(source: str, method: str) -> torch.Tensor:
            available = x_fft_channels if source == "x" else derivative_fft_channels
            position = {channel: idx for idx, channel in enumerate(available)}
            selected_positions = [position[channel] for channel in method_channels[method]]
            return get_spectrum(source).index_select(-1, channel_index(selected_positions))

        def get_power() -> torch.Tensor:
            nonlocal power
            if power is None:
                power = get_spectrum().abs().square()
            return power

        def weighted_rms(source: str, weight: str, method: str) -> torch.Tensor:
            xf = select_spectrum(source, method)
            if weight not in iso_weights:
                iso_weights[weight] = _iso_weight(get_frequencies(), weight)
            w = iso_weights[weight].reshape(1, 1, -1, 1)
            filtered = torch.fft.irfft(xf * w, n=x.shape[2], dim=2)
            return expand_channels(method, filtered.square().mean(dim=2).sqrt())

        if "mean" in needed:
            values["mean"] = get_mean()
        if "std" in needed:
            values["std"] = x.std(dim=2, unbiased=False)
        if "abs_peak" in needed or "crest" in needed:
            values["abs_peak"] = get_abs().amax(dim=2)
        if "energy" in needed or "rms" in needed or "crest" in needed:
            values["energy"] = get_x2().mean(dim=2)
        if "rms" in needed:
            values["rms"] = values["energy"].sqrt()
        if "p95_abs" in needed:
            values["p95_abs"] = torch.quantile(get_abs(), 0.95, dim=2)
        if "p2p" in needed:
            values["p2p"] = x.amax(dim=2) - x.amin(dim=2)
        if "impulse_abs" in needed:
            values["impulse_abs"] = get_abs().sum(dim=2) / self.fs
        if "crest" in needed:
            values["crest"] = values["abs_peak"] / (values["energy"].sqrt() + eps)
        if "z_nod" in needed:
            if x.shape[2] < 3:
                values["z_nod"] = torch.zeros_like(get_mean())
            else:
                center = x[:, :, 1:-1, :]
                peak = (
                    (center > x[:, :, :-2, :])
                    & (center > x[:, :, 2:, :])
                    & (center > 0)
                )
                rank = peak.to(torch.int32).cumsum(dim=2)
                values["z_nod"] = torch.where(
                    peak & (rank == 3), center, torch.zeros_like(center)
                ).sum(dim=2)

        if "rms_deriv" in needed:
            values["rms_deriv"] = get_derivative2().mean(dim=2).sqrt()
        if "p2p_deriv" in needed:
            dx = get_derivative()
            values["p2p_deriv"] = dx.amax(dim=2) - dx.amin(dim=2)
        if "abs_peak_deriv" in needed:
            values["abs_peak_deriv"] = get_derivative().abs().amax(dim=2)
        if "vdv" in needed:
            values["vdv"] = (x.pow(4).sum(dim=2) / self.fs).pow(0.25)
        if "vdv_deriv" in needed:
            values["vdv_deriv"] = (
                get_derivative().pow(4).sum(dim=2) / self.fs
            ).pow(0.25)
        if "mtvv" in needed:
            window = int(round(self.fs))
            if window >= x.shape[2]:
                values["mtvv"] = get_x2().mean(dim=2).sqrt()
            else:
                cumulative = get_x2().cumsum(dim=2)
                running = (
                    cumulative[:, :, window:, :]
                    - cumulative[:, :, :-window, :]
                ) / window
                values["mtvv"] = running.amax(dim=2).sqrt()

        if "wrms_z" in needed:
            values["wrms_z"] = weighted_rms("x", "wk", "wrms_z")
        if "wrms_xy" in needed:
            values["wrms_xy"] = weighted_rms("x", "wd", "wrms_xy")
        if "wrms_rot" in needed:
            values["wrms_rot"] = weighted_rms("derivative", "we", "wrms_rot")
        if "wrms_z_d" in needed:
            values["wrms_z_d"] = weighted_rms("derivative", "wk", "wrms_z_d")

        if {"sigma_sd", "band_low", "band_mid"} & needed:
            f = get_frequencies()

            def band_sum(method: str, lo: float, hi: float) -> torch.Tensor:
                available = {channel: idx for idx, channel in enumerate(x_fft_channels)}
                positions = [available[channel] for channel in method_channels[method]]
                p = get_power().index_select(-1, channel_index(positions))
                mask = (f >= lo) & (f < hi)
                return expand_channels(method, p[:, :, mask, :].sum(dim=2))

            if "sigma_sd" in needed:
                values["sigma_sd"] = band_sum("sigma_sd", 0.5, 5.0).sqrt()
            if "band_low" in needed:
                available = {channel: idx for idx, channel in enumerate(x_fft_channels)}
                positions = [available[channel] for channel in method_channels["band_low"]]
                p = get_power().index_select(-1, channel_index(positions))
                total_power = p.sum(dim=2) + eps
                selected = p[:, :, (f >= 1.0) & (f < 2.0), :].sum(dim=2) / total_power
                values["band_low"] = expand_channels("band_low", selected)
            if "band_mid" in needed:
                available = {channel: idx for idx, channel in enumerate(x_fft_channels)}
                positions = [available[channel] for channel in method_channels["band_mid"]]
                p = get_power().index_select(-1, channel_index(positions))
                total_power = p.sum(dim=2) + eps
                selected = p[:, :, (f >= 4.0) & (f < 8.0), :].sum(dim=2) / total_power
                values["band_mid"] = expand_channels("band_mid", selected)

        if "rms_dfz" in needed:
            x_mean = get_mean().unsqueeze(2)
            normalized = (x - x_mean) / (x_mean + eps)
            values["rms_dfz"] = normalized.square().mean(dim=2).sqrt()

        columns = [values[method][..., channel] for channel, method in self.pairs]
        return torch.stack(columns, dim=-1)

    def fit_transform(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract(x)
        self.mu = features.mean(dim=(0, 1))
        self.sd = features.std(dim=(0, 1), unbiased=False) + 1e-12
        z = (features - self.mu) / self.sd if self.standardize else features
        if self.include_bias:
            ones = torch.ones((*z.shape[:-1], 1), device=z.device, dtype=z.dtype)
            z = torch.cat((ones, z), dim=-1)
        return z

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.mu is None or self.sd is None:
            raise RuntimeError("fit_transform must be called before transform")
        features = self.extract(x)
        z = (features - self.mu) / self.sd if self.standardize else features
        if self.include_bias:
            ones = torch.ones((*z.shape[:-1], 1), device=z.device, dtype=z.dtype)
            z = torch.cat((ones, z), dim=-1)
        return z


def pg_truncated(
    psi: torch.Tensor,
    terms: int,
    chunk_size: int,
) -> torch.Tensor:
    """K-term approximation to PG(1, psi), vectorized over every psi."""
    if terms < 1:
        raise ValueError("PG terms must be positive")
    if chunk_size < 1:
        raise ValueError("PG chunk size must be positive")
    result = torch.zeros_like(psi)
    scaled_psi2 = psi.square() / (4.0 * math.pi**2)
    prefix = (1,) * psi.ndim
    for start in range(0, terms, chunk_size):
        stop = min(start + chunk_size, terms)
        k = torch.arange(start + 1, stop + 1, device=psi.device, dtype=psi.dtype)
        denominator = (k - 0.5).square().reshape(*prefix, -1) + scaled_psi2.unsqueeze(-1)
        exponentials = torch.empty(
            (*psi.shape, stop - start), device=psi.device, dtype=psi.dtype
        ).exponential_(1.0)
        result.add_((exponentials / denominator).sum(dim=-1))
    return result / (2.0 * math.pi**2)


def pg_mean(psi: torch.Tensor) -> torch.Tensor:
    """Analytic E[PG(1, psi)]."""
    near_zero = psi.abs() < 1e-6
    safe = torch.where(near_zero, torch.ones_like(psi), psi)
    value = torch.tanh(safe / 2.0) / (2.0 * safe)
    return torch.where(near_zero, torch.full_like(value, 0.25), value)


def sample_inverse_wishart(scale: torch.Tensor, df: float) -> torch.Tensor:
    """Batched IW(df, scale) sample via Wishart Bartlett decomposition."""
    if scale.ndim != 3 or scale.shape[-1] != scale.shape[-2]:
        raise ValueError("scale must have shape [chains, d, d]")
    chains, d, _ = scale.shape
    if df <= d - 1:
        raise ValueError(f"Inverse-Wishart requires df > d-1; got df={df}, d={d}")

    # Sigma^{-1} ~ Wishart(df, scale^{-1}).
    wishart_scale = torch.linalg.inv(scale)
    wishart_chol = torch.linalg.cholesky(wishart_scale)

    bartlett = torch.tril(
        torch.randn((chains, d, d), device=scale.device, dtype=scale.dtype),
        diagonal=-1,
    )
    diag_df = df - torch.arange(d, device=scale.device, dtype=scale.dtype)
    concentration = (diag_df / 2.0).expand(chains, d)
    rate = torch.full_like(concentration, 0.5)
    chi_square = torch.distributions.Gamma(concentration, rate).sample()
    bartlett = bartlett + torch.diag_embed(chi_square.sqrt())

    factor = wishart_chol @ bartlett
    identity = torch.eye(d, device=scale.device, dtype=scale.dtype).expand(chains, d, d)
    factor_inv = torch.linalg.solve_triangular(factor, identity, upper=False)
    sigma = factor_inv.transpose(-1, -2) @ factor_inv
    return 0.5 * (sigma + sigma.transpose(-1, -2))


@dataclass(frozen=True)
class GibbsConfig:
    burnin: int
    samples: int
    thin: int
    chains: int
    pg_terms: int
    pg_chunk: int
    niw_kappa0: float = 1.0
    niw_nu0: float | None = None
    niw_lambda0_scale: float = 1.0
    eps_var: float | None = None
    jitter: float = 1e-5

    @property
    def total_steps(self) -> int:
        return self.burnin + self.samples * self.thin


@dataclass
class FitSummary:
    theta_mean: torch.Tensor
    mu_mean: torch.Tensor
    sigma_mean: torch.Tensor
    log_likelihood: float
    lpd_per_item: float
    brier: float
    accuracy: float
    theta_rmse: float


def fit_population(
    z: torch.Tensor,
    y: torch.Tensor,
    cfg: GibbsConfig,
    seed: int,
    theta_true: torch.Tensor | None = None,
) -> FitSummary:
    """Hierarchical PG-Gibbs + NIW using batched independent chains."""
    if z.ndim != 3 or y.shape != z.shape[:2]:
        raise ValueError("z must be [users, items, d] and y must be [users, items]")
    if cfg.samples < 1 or cfg.thin < 1 or cfg.chains < 1:
        raise ValueError("samples, thin, and chains must be positive")

    torch.manual_seed(seed)
    if z.is_cuda:
        torch.cuda.manual_seed_all(seed)

    users, _, d = z.shape
    chains = cfg.chains
    device, dtype = z.device, z.dtype
    kappa = y - 0.5
    z_transpose = z.transpose(-1, -2)
    m0 = torch.zeros(d, device=device, dtype=dtype)
    lambda0 = cfg.niw_lambda0_scale * torch.eye(d, device=device, dtype=dtype)
    nu0 = float(d + 2) if cfg.niw_nu0 is None else float(cfg.niw_nu0)
    identity = torch.eye(d, device=device, dtype=dtype)

    theta = 0.1 * torch.randn((chains, users, d), device=device, dtype=dtype)
    sigma = identity.expand(chains, d, d).clone()
    mu = torch.zeros((chains, d), device=device, dtype=dtype)
    epsilon = torch.zeros((chains, *y.shape), device=device, dtype=dtype)
    inv_eps = 0.0 if cfg.eps_var is None else 1.0 / cfg.eps_var

    theta_sum = torch.zeros_like(theta)
    mu_sum = torch.zeros_like(mu)
    sigma_sum = torch.zeros_like(sigma)
    retained = 0

    for iteration in range(cfg.total_steps):
        mean_eta = torch.einsum("und,cud->cun", z, theta)
        omega = pg_truncated(
            mean_eta + epsilon,
            terms=cfg.pg_terms,
            chunk_size=cfg.pg_chunk,
        )

        if cfg.eps_var is not None:
            precision = omega + inv_eps
            epsilon_mean = (kappa.unsqueeze(0) - omega * mean_eta) / precision
            epsilon = epsilon_mean + torch.randn_like(epsilon) / precision.sqrt()

        sigma_inv = torch.linalg.inv(sigma + cfg.jitter * identity)
        sigma_inv_mu = torch.einsum("cij,cj->ci", sigma_inv, mu)

        weighted_z = omega.unsqueeze(-1) * z.unsqueeze(0)
        precision_theta = (
            z_transpose.unsqueeze(0) @ weighted_z
            + sigma_inv.unsqueeze(1)
            + cfg.jitter * identity
        )
        residual = kappa.unsqueeze(0) - omega * epsilon
        linear_theta = (
            (z_transpose.unsqueeze(0) @ residual.unsqueeze(-1)).squeeze(-1)
            + sigma_inv_mu.unsqueeze(1)
        )

        chol = torch.linalg.cholesky(precision_theta)
        theta_mean = torch.cholesky_solve(linear_theta.unsqueeze(-1), chol).squeeze(-1)
        normal = torch.randn_like(theta_mean)
        theta_noise = torch.linalg.solve_triangular(
            chol.transpose(-1, -2), normal.unsqueeze(-1), upper=True
        ).squeeze(-1)
        theta = theta_mean + theta_noise

        theta_bar = theta.mean(dim=1)
        centered = theta - theta_bar.unsqueeze(1)
        kappa_n = cfg.niw_kappa0 + users
        nu_n = nu0 + users
        offset = theta_bar - m0
        lambda_n = (
            lambda0.unsqueeze(0)
            + centered.transpose(-1, -2) @ centered
            + (cfg.niw_kappa0 * users / kappa_n)
            * offset.unsqueeze(-1)
            * offset.unsqueeze(-2)
        )
        sigma = sample_inverse_wishart(lambda_n, nu_n)
        m_n = (cfg.niw_kappa0 * m0 + users * theta_bar) / kappa_n
        mu_chol = torch.linalg.cholesky(
            sigma / kappa_n + cfg.jitter * identity
        )
        mu = m_n + (mu_chol @ torch.randn_like(mu).unsqueeze(-1)).squeeze(-1)

        if iteration >= cfg.burnin and (iteration - cfg.burnin) % cfg.thin == 0:
            theta_sum += theta
            mu_sum += mu
            sigma_sum += sigma
            retained += 1

    theta_mean = (theta_sum / retained).mean(dim=0)
    mu_mean = (mu_sum / retained).mean(dim=0)
    sigma_mean = (sigma_sum / retained).mean(dim=0)
    eta = torch.einsum("und,ud->un", z, theta_mean)
    probability = torch.sigmoid(eta).clamp(1e-8, 1.0 - 1e-8)
    log_likelihood = (y * eta - torch.nn.functional.softplus(eta)).sum()
    lpd = (y * probability.log() + (1 - y) * torch.log1p(-probability)).mean()
    brier = (probability - y).square().mean()
    accuracy = ((probability >= 0.5) == (y >= 0.5)).to(dtype).mean()
    if theta_true is None:
        theta_rmse = torch.full((), float("nan"), device=device, dtype=dtype)
    else:
        theta_rmse = (theta_mean - theta_true.to(device=device, dtype=dtype)).square().mean().sqrt()

    return FitSummary(
        theta_mean=theta_mean.detach(),
        mu_mean=mu_mean.detach(),
        sigma_mean=sigma_mean.detach(),
        log_likelihood=float(log_likelihood.detach().cpu()),
        lpd_per_item=float(lpd.detach().cpu()),
        brier=float(brier.detach().cpu()),
        accuracy=float(accuracy.detach().cpu()),
        theta_rmse=float(theta_rmse.detach().cpu()),
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_call(device: torch.device, function):
    synchronize(device)
    start = time.perf_counter()
    result = function()
    synchronize(device)
    return result, time.perf_counter() - start


def median_timed_call(device: torch.device, repeats: int, function):
    if repeats < 1:
        raise ValueError("repeats must be positive")
    result = None
    durations = []
    for _ in range(repeats):
        result, duration = timed_call(device, function)
        durations.append(duration)
    return result, float(statistics.median(durations))


def generate_raw_trajectories(
    users: int,
    items: int,
    window: int,
    sensors: int,
    fs: float,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    """Generate heterogeneous, nontrivial multi-sensor trajectories on CPU."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    t = torch.arange(window, dtype=dtype).reshape(1, 1, window, 1) / fs
    shape = (users, items, 1, sensors)
    amplitude = torch.exp(0.30 * torch.randn(shape, generator=generator, dtype=dtype))
    frequency = 0.4 + 4.6 * torch.rand(shape, generator=generator, dtype=dtype)
    phase = 2.0 * math.pi * torch.rand(shape, generator=generator, dtype=dtype)
    offset = 0.25 * torch.randn(shape, generator=generator, dtype=dtype)
    user_offset = 0.20 * torch.randn(
        (users, 1, 1, sensors), generator=generator, dtype=dtype
    )
    modulation = 0.2 + 0.2 * torch.rand(shape, generator=generator, dtype=dtype)

    x = offset + user_offset
    x = x + amplitude * torch.sin(2.0 * math.pi * frequency * t + phase)
    x = x + modulation * amplitude * torch.sin(
        4.0 * math.pi * frequency * t + 0.5 * phase
    )
    x = x + 0.20 * torch.randn(
        (users, items, window, sensors), generator=generator, dtype=dtype
    )
    return x.contiguous()


def generate_feedback(
    z: torch.Tensor,
    seed: int,
    logit_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate hierarchical user coefficients and binary feedback."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    users, _, d = z.shape
    population_mean = 0.18 * torch.randn(d, generator=generator, dtype=z.dtype)
    if d > 0:
        population_mean[0] = 0.0
    theta_true = population_mean + 0.25 * torch.randn(
        (users, d), generator=generator, dtype=z.dtype
    )
    raw_eta = torch.einsum("und,ud->un", z, theta_true)
    scale = raw_eta.std(unbiased=False).clamp_min(1e-6) / logit_scale
    theta_true = theta_true / scale
    eta = torch.einsum("und,ud->un", z, theta_true)
    probability = torch.sigmoid(eta)
    y = torch.bernoulli(probability, generator=generator)
    for user in range(users):
        if y[user].sum() == 0:
            y[user, 0] = 1.0
        elif y[user].sum() == y.shape[1]:
            y[user, 0] = 0.0
    return y, theta_true


def tensor_bytes(shape: Sequence[int], dtype: torch.dtype) -> int:
    return math.prod(shape) * torch.empty((), dtype=dtype).element_size()


def benchmark_structure(
    *,
    users: int,
    items: int,
    window: int,
    sensors: int,
    pg_terms_values: Sequence[int],
    chain_values: Sequence[int],
    args: argparse.Namespace,
    dtype: torch.dtype,
) -> list[dict]:
    raw_gb = tensor_bytes((users, items, window, sensors), dtype) / 2**30
    if raw_gb > args.max_raw_gb:
        print(
            f"[skip] U={users} N={items} T={window} D={sensors}: "
            f"raw tensor {raw_gb:.2f} GiB > --max-raw-gb {args.max_raw_gb:.2f}"
        )
        return []

    structure_seed = args.seed + users * 1_000_003 + items * 101 + window * 17 + sensors
    print(
        f"\n[data] U={users} N/user={items} T={window} sensors={sensors} "
        f"raw={raw_gb:.3f} GiB"
    )
    raw_cpu = generate_raw_trajectories(
        users, items, window, sensors, args.fs, dtype, structure_seed
    )
    manual_stats = make_manual_stats(sensors, args.stats_preset, args.stats)

    cpu_device = torch.device("cpu")
    if args.warmup_features:
        warm_extractor = TorchManualFeatures(manual_stats, args.fs)
        warm_extractor.fit_transform(raw_cpu)
    z_cpu, cpu_feature_s = median_timed_call(
        cpu_device,
        args.repeats,
        lambda: TorchManualFeatures(manual_stats, args.fs).fit_transform(raw_cpu),
    )
    y_cpu, theta_true = generate_feedback(z_cpu, structure_seed + 1, args.logit_scale)

    cuda_available = torch.cuda.is_available() and not args.cpu_only
    gpu_transfer_s = float("nan")
    gpu_feature_s = float("nan")
    gpu_feature_diff = float("nan")
    raw_gpu = None
    y_gpu = None
    theta_true_gpu = None
    z_gpu = None
    gpu_peak_feature_mb = float("nan")

    if cuda_available:
        gpu_device = torch.device(args.cuda_device)
        torch.cuda.reset_peak_memory_stats(gpu_device)

        def transfer():
            return (
                raw_cpu.to(gpu_device),
                y_cpu.to(gpu_device),
                theta_true.to(gpu_device),
            )

        (raw_gpu, y_gpu, theta_true_gpu), gpu_transfer_s = timed_call(gpu_device, transfer)
        if args.warmup_features:
            TorchManualFeatures(manual_stats, args.fs).fit_transform(raw_gpu)
        z_gpu, gpu_feature_s = median_timed_call(
            gpu_device,
            args.repeats,
            lambda: TorchManualFeatures(manual_stats, args.fs).fit_transform(raw_gpu),
        )
        gpu_peak_feature_mb = torch.cuda.max_memory_allocated(gpu_device) / 2**20
        gpu_feature_diff = float(
            (z_cpu - z_gpu.detach().cpu()).abs().amax().cpu()
        )
    elif args.require_cuda:
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    rows: list[dict] = []
    for pg_terms, chains in itertools.product(pg_terms_values, chain_values):
        cfg = GibbsConfig(
            burnin=args.burnin,
            samples=args.samples,
            thin=args.thin,
            chains=chains,
            pg_terms=pg_terms,
            pg_chunk=min(args.pg_chunk, pg_terms),
            niw_kappa0=args.niw_kappa0,
            niw_nu0=None,
            niw_lambda0_scale=args.niw_lambda0_scale,
            eps_var=args.eps_var,
            jitter=args.jitter,
        )
        model_seed = structure_seed + 10_000 + pg_terms * 13 + chains
        print(
            f"  [model] K={pg_terms:<4} chains={chains:<2} "
            f"steps={cfg.total_steps:<4} d={z_cpu.shape[-1]}"
        )

        if args.warmup_gibbs:
            warm_cfg = GibbsConfig(**{**asdict(cfg), "burnin": 0, "samples": 1, "thin": 1})
            fit_population(z_cpu, y_cpu, warm_cfg, model_seed, theta_true)

        cpu_fit, cpu_gibbs_s = median_timed_call(
            cpu_device,
            args.repeats,
            lambda: fit_population(z_cpu, y_cpu, cfg, model_seed, theta_true),
        )

        gpu_gibbs_s = float("nan")
        gpu_fit = None
        gpu_peak_mb = float("nan")
        if cuda_available:
            gpu_device = torch.device(args.cuda_device)
            torch.cuda.reset_peak_memory_stats(gpu_device)
            if args.warmup_gibbs:
                warm_cfg = GibbsConfig(
                    **{**asdict(cfg), "burnin": 0, "samples": 1, "thin": 1}
                )
                fit_population(z_gpu, y_gpu, warm_cfg, model_seed, theta_true_gpu)
            gpu_fit, gpu_gibbs_s = median_timed_call(
                gpu_device,
                args.repeats,
                lambda: fit_population(z_gpu, y_gpu, cfg, model_seed, theta_true_gpu),
            )
            gpu_peak_mb = torch.cuda.max_memory_allocated(gpu_device) / 2**20

        cpu_total_s = cpu_feature_s + cpu_gibbs_s
        gpu_compute_s = gpu_feature_s + gpu_gibbs_s
        gpu_total_s = gpu_transfer_s + gpu_compute_s

        row = {
            "users": users,
            "items_per_user": items,
            "observations": users * items,
            "window_length": window,
            "sensors": sensors,
            "n_features": int(z_cpu.shape[-1]),
            "stats_preset": args.stats_preset if not args.stats else "custom",
            "dtype": str(dtype).replace("torch.", ""),
            "pg_terms": pg_terms,
            "pg_chunk": cfg.pg_chunk,
            "chains": chains,
            "burnin": cfg.burnin,
            "samples": cfg.samples,
            "thin": cfg.thin,
            "total_steps": cfg.total_steps,
            "eps_var": args.eps_var,
            "raw_gib": raw_gb,
            "cpu_feature_s": cpu_feature_s,
            "cpu_gibbs_s": cpu_gibbs_s,
            "cpu_total_s": cpu_total_s,
            "cpu_lpd": cpu_fit.lpd_per_item,
            "cpu_brier": cpu_fit.brier,
            "cpu_accuracy": cpu_fit.accuracy,
            "cpu_theta_rmse": cpu_fit.theta_rmse,
            "gpu_transfer_s": gpu_transfer_s,
            "gpu_feature_s": gpu_feature_s,
            "gpu_gibbs_s": gpu_gibbs_s,
            "gpu_compute_s": gpu_compute_s,
            "gpu_total_s": gpu_total_s,
            "gpu_lpd": float("nan") if gpu_fit is None else gpu_fit.lpd_per_item,
            "gpu_brier": float("nan") if gpu_fit is None else gpu_fit.brier,
            "gpu_accuracy": float("nan") if gpu_fit is None else gpu_fit.accuracy,
            "gpu_theta_rmse": float("nan") if gpu_fit is None else gpu_fit.theta_rmse,
            "feature_max_abs_diff": gpu_feature_diff,
            "feature_speedup": cpu_feature_s / gpu_feature_s,
            "gibbs_speedup": cpu_gibbs_s / gpu_gibbs_s,
            "compute_speedup": cpu_total_s / gpu_compute_s,
            "end_to_end_speedup": cpu_total_s / gpu_total_s,
            "gpu_peak_feature_mib": gpu_peak_feature_mb,
            "gpu_peak_model_mib": gpu_peak_mb,
        }
        rows.append(row)

        if gpu_fit is None:
            print(
                f"    CPU feature={cpu_feature_s:.3f}s Gibbs={cpu_gibbs_s:.3f}s "
                f"LPD={cpu_fit.lpd_per_item:+.4f}"
            )
        else:
            print(
                f"    CPU={cpu_total_s:.3f}s GPU={gpu_total_s:.3f}s "
                f"speedup={row['end_to_end_speedup']:.2f}x "
                f"(feature={row['feature_speedup']:.2f}x, "
                f"Gibbs={row['gibbs_speedup']:.2f}x)"
            )

    del raw_cpu, z_cpu, y_cpu, theta_true
    if cuda_available:
        del raw_gpu, z_gpu, y_gpu, theta_true_gpu
        torch.cuda.empty_cache()
    return rows


def write_csv(rows: Sequence[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def break_even_summary(rows: Sequence[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    keys = (
        "window_length",
        "sensors",
        "n_features",
        "stats_preset",
        "dtype",
        "pg_terms",
        "chains",
        "total_steps",
    )
    for row in rows:
        if not finite(row.get("end_to_end_speedup")):
            continue
        group = tuple(row[key] for key in keys)
        groups.setdefault(group, []).append(row)

    result = []
    for group, candidates in sorted(groups.items()):
        candidates.sort(key=lambda row: (row["observations"], row["users"]))
        winner = next(
            (row for row in candidates if row["end_to_end_speedup"] > 1.0),
            None,
        )
        record = {key: value for key, value in zip(keys, group)}
        if winner is None:
            record.update(
                {
                    "break_even_observations": None,
                    "break_even_users": None,
                    "break_even_items_per_user": None,
                    "speedup_at_break_even": None,
                    "largest_tested_speedup": max(
                        row["end_to_end_speedup"] for row in candidates
                    ),
                }
            )
        else:
            record.update(
                {
                    "break_even_observations": winner["observations"],
                    "break_even_users": winner["users"],
                    "break_even_items_per_user": winner["items_per_user"],
                    "speedup_at_break_even": winner["end_to_end_speedup"],
                    "largest_tested_speedup": max(
                        row["end_to_end_speedup"] for row in candidates
                    ),
                }
            )
        result.append(record)
    return result


def write_break_even(summary: Sequence[dict], output_dir: Path) -> None:
    with (output_dir / "break_even.json").open("w", encoding="utf-8") as handle:
        json.dump(list(summary), handle, indent=2, ensure_ascii=False)

    lines = [
        "CPU vs CUDA break-even summary (end-to-end speedup includes H2D transfer)",
        "=" * 80,
        "",
    ]
    if not summary:
        lines.append("No CUDA timing rows were available.")
    for row in summary:
        fixed = (
            f"T={row['window_length']}, sensors={row['sensors']}, "
            f"features={row['n_features']}, K={row['pg_terms']}, "
            f"chains={row['chains']}, steps={row['total_steps']}"
        )
        if row["break_even_observations"] is None:
            lines.append(
                f"{fixed}: no break-even in tested grid "
                f"(best={row['largest_tested_speedup']:.2f}x)"
            )
        else:
            lines.append(
                f"{fixed}: first observed break-even at "
                f"{row['break_even_observations']} observations "
                f"(U={row['break_even_users']}, N/U={row['break_even_items_per_user']}), "
                f"speedup={row['speedup_at_break_even']:.2f}x"
            )
    (output_dir / "break_even.txt").write_text("\n".join(lines), encoding="utf-8")


def plot_results(rows: Sequence[dict], output_dir: Path) -> None:
    valid = [row for row in rows if finite(row.get("end_to_end_speedup"))]
    if not valid:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib is unavailable; skipping benchmark plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    markers = {1: "o", 2: "s", 4: "^", 8: "D"}
    pg_values = sorted({row["pg_terms"] for row in valid})
    colors = plt.cm.viridis(torch.linspace(0.1, 0.9, len(pg_values)).numpy())

    for pg_terms, color in zip(pg_values, colors):
        for chains in sorted({row["chains"] for row in valid}):
            subset = [
                row
                for row in valid
                if row["pg_terms"] == pg_terms and row["chains"] == chains
            ]
            subset.sort(key=lambda row: row["observations"])
            if not subset:
                continue
            axes[0].scatter(
                [row["observations"] for row in subset],
                [row["feature_speedup"] for row in subset],
                marker=markers.get(chains, "o"),
                color=color,
                alpha=0.8,
                label=f"K={pg_terms}, C={chains}",
            )
            axes[1].scatter(
                [row["observations"] for row in subset],
                [row["gibbs_speedup"] for row in subset],
                marker=markers.get(chains, "o"),
                color=color,
                alpha=0.8,
                label=f"K={pg_terms}, C={chains}",
            )
            axes[2].scatter(
                [row["observations"] for row in subset],
                [row["end_to_end_speedup"] for row in subset],
                marker=markers.get(chains, "o"),
                color=color,
                alpha=0.8,
                label=f"K={pg_terms}, C={chains}",
            )

    for ax, title in zip(
        axes,
        ("Manual-feature speedup", "PG-Gibbs-only speedup", "End-to-end speedup"),
    ):
        ax.axhline(1.0, color="crimson", linestyle="--", linewidth=1.2)
        ax.set_xscale("log")
        ax.set_xlabel("total observations (users x items/user)")
        ax.set_ylabel("CPU time / CUDA time")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="upper center", ncol=min(4, len(unique)))
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(output_dir / "speedup.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def validate_pg(
    devices: Sequence[torch.device],
    terms_values: Sequence[int],
    samples: int,
    chunk_size: int,
    dtype: torch.dtype,
    seed: int,
) -> list[dict]:
    z_values = (0.0, 0.5, 1.0, 3.0, 5.0)
    rows = []
    for device in devices:
        for terms in terms_values:
            for z_value in z_values:
                torch.manual_seed(seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(seed)
                psi = torch.full((samples,), z_value, device=device, dtype=dtype)
                draw, elapsed = timed_call(
                    device,
                    lambda: pg_truncated(psi, terms, min(chunk_size, terms)),
                )
                empirical = float(draw.mean().cpu())
                analytic = float(pg_mean(psi[:1]).item())
                rows.append(
                    {
                        "device": str(device),
                        "pg_terms": terms,
                        "z": z_value,
                        "samples": samples,
                        "sample_mean": empirical,
                        "analytic_mean": analytic,
                        "relative_mean_error": abs(empirical - analytic) / analytic,
                        "seconds": elapsed,
                    }
                )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep CPU-vs-CUDA manual features and hierarchical PG-Gibbs"
    )
    parser.add_argument("--users", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--items-per-user", type=int, nargs="+", default=[100, 500, 2000])
    parser.add_argument("--window-lengths", type=int, nargs="+", default=[128])
    parser.add_argument("--sensor-counts", type=int, nargs="+", default=[5])
    parser.add_argument("--pg-terms", type=int, nargs="+", default=[25, 50, 100])
    parser.add_argument("--chains", type=int, nargs="+", default=[1])
    parser.add_argument("--pg-chunk", type=int, default=25)
    parser.add_argument("--burnin", type=int, default=500)
    parser.add_argument("--samples", type=int, default=1500)
    parser.add_argument("--thin", type=int, default=1)
    parser.add_argument("--eps-var", type=float, default=None)
    parser.add_argument("--niw-kappa0", type=float, default=1.0)
    parser.add_argument("--niw-lambda0-scale", type=float, default=1.0)
    parser.add_argument("--jitter", type=float, default=1e-5)
    parser.add_argument("--stats-preset", choices=("basic", "current", "full"), default="current")
    parser.add_argument(
        "--stats",
        nargs="+",
        choices=ALL_STATS,
        default=None,
        help="Custom list applied to every sensor; overrides --stats-preset",
    )
    parser.add_argument("--fs", type=float, default=50.0)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--logit-scale", type=float, default=1.25)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--cuda-device", default="cuda:0")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--max-raw-gb", type=float, default=4.0)
    parser.add_argument("--warmup-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup-gibbs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validate-pg", action="store_true")
    parser.add_argument("--pg-validation-samples", type=int, default=50_000)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/benchmark"))
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Override the grid with a very small smoke-test configuration",
    )
    parser.add_argument(
        "--fb-preset",
        action="store_true",
        help="run_fully_bayesian Config에 맞춘 하이퍼파라미터로 덮어씀 (population fit 타이밍)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fb_preset:
        args.users = [FB_USERS]
        args.items_per_user = [FB_ITEMS_PER_USER]
        args.window_lengths = [FB_WINDOW]
        args.sensor_counts = [FB_SENSORS]
        args.stats = list(FB_STATS)
        args.pg_terms = [FB_PG_TERMS]
        args.chains = [FB_CHAINS]
        args.burnin = FB_BURNIN
        args.samples = FB_SAMPLES
        args.thin = FB_THIN
        args.niw_kappa0 = FB_NIW_KAPPA0
        args.niw_lambda0_scale = FB_NIW_LAMBDA0_SCALE
        args.eps_var = FB_EPS_VAR
        args.dtype = FB_DTYPE
    if args.quick:
        args.users = [2]
        args.items_per_user = [24, 64]
        args.window_lengths = [64]
        args.sensor_counts = [3]
        args.pg_terms = [10, 25]
        args.chains = [1]
        args.burnin = 1
        args.samples = 2
        args.stats_preset = "basic"
        args.stats = None
        args.pg_validation_samples = min(args.pg_validation_samples, 5_000)

    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    args.output_dir.mkdir(parents=True, exist_ok=True)

    environment = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_device": (
            torch.cuda.get_device_name(torch.device(args.cuda_device))
            if torch.cuda.is_available() and not args.cpu_only
            else None
        ),
        "cpu_threads": torch.get_num_threads(),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(environment, indent=2, ensure_ascii=False))

    if args.require_cuda and (not torch.cuda.is_available() or args.cpu_only):
        raise RuntimeError("--require-cuda was set but CUDA is unavailable or disabled")

    if args.validate_pg:
        devices = [torch.device("cpu")]
        if torch.cuda.is_available() and not args.cpu_only:
            devices.append(torch.device(args.cuda_device))
        validation_rows = validate_pg(
            devices,
            args.pg_terms,
            args.pg_validation_samples,
            args.pg_chunk,
            dtype,
            args.seed,
        )
        write_csv(validation_rows, args.output_dir / "pg_validation.csv")
        print("\n[PG validation]")
        for row in validation_rows:
            print(
                f"  {row['device']:<8} K={row['pg_terms']:<4} z={row['z']:<3} "
                f"mean={row['sample_mean']:.6f} exact={row['analytic_mean']:.6f} "
                f"rel.err={row['relative_mean_error']:.2%}"
            )

    rows: list[dict] = []
    structural_grid = itertools.product(
        args.users,
        args.items_per_user,
        args.window_lengths,
        args.sensor_counts,
    )
    for users, items, window, sensors in structural_grid:
        rows.extend(
            benchmark_structure(
                users=users,
                items=items,
                window=window,
                sensors=sensors,
                pg_terms_values=args.pg_terms,
                chain_values=args.chains,
                args=args,
                dtype=dtype,
            )
        )
        write_csv(rows, args.output_dir / "benchmark.csv")

    summary = break_even_summary(rows)
    write_break_even(summary, args.output_dir)
    plot_results(rows, args.output_dir)
    print(f"\nSaved benchmark outputs to: {args.output_dir.resolve()}")
    print(
        "\n[NOTE] Timed workload is the population fit (train) only. New-user/test "
        "adaptation (newuser_n_iters; paper Alg. 2) and held-out evaluation are out of "
        "scope: it is a warm-started ~L-sweep step whose cost is negligible next to the "
        "population Gibbs, so it does not shift the CPU/GPU crossover. Reported "
        "lpd/brier/accuracy are in-sample training fit, not held-out."
    )


if __name__ == "__main__":
    main()
