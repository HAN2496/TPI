"""
Toy GPU Pólya-Gamma PG(1, z) sampler benchmark.

Purpose
-------
This script is intentionally separated from the main reward-model code.
It tests whether a CUDA/GPU approximation of PG(1,z) can match the Python
`polyagamma` package closely enough, and when GPU throughput begins to beat CPU.

Sampler used here
-----------------
Truncated infinite-sum representation:

    PG(1, z) ~= 1/(2*pi^2) * sum_{k=1}^K Exp(1) / ((k-1/2)^2 + z^2/(4*pi^2))

This is NOT Devroye's exact PG(1,z) sampler. It is a simple CUDA-friendly toy
sampler for feasibility, quality, and speed crossover tests.

Dependencies
------------
CPU reference:
    pip install polyagamma scipy numpy
GPU toy:
    pip install cupy-cuda12x   # or cupy-cuda11x matching your CUDA stack

Examples
--------
python cuda_pg_toy.py --sizes 1000 10000 100000 1000000 --Ks 50 100 200
python cuda_pg_toy.py --fixed-z 0.0 --sizes 10000 100000 1000000 --Ks 100 200 500
python cuda_pg_toy.py --quality-n 50000 --Ks 50 100 200 500 --skip-benchmark
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np

try:
    from polyagamma import random_polyagamma
except Exception:  # pragma: no cover
    random_polyagamma = None

try:
    from scipy.stats import ks_2samp
except Exception:  # pragma: no cover
    ks_2samp = None

try:
    import cupy as cp
except Exception:  # pragma: no cover
    cp = None


@dataclass
class BenchRow:
    N: int
    K: int
    cpu_sec: float | None
    gpu_sec: float
    speedup: float | None
    cpu_mean: float | None
    gpu_mean: float
    theory_mean: float
    gpu_abs_mean_err: float


def pg1_mean(z: np.ndarray) -> np.ndarray:
    """Analytic E[PG(1,z)] = tanh(z/2)/(2z), with limit 1/4 at z=0."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z, dtype=np.float64)
    small = np.abs(z) < 1e-10
    out[small] = 0.25
    out[~small] = np.tanh(z[~small] / 2.0) / (2.0 * z[~small])
    return out


def make_z(N: int, mode: str, fixed_z: float | None, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if fixed_z is not None:
        return np.full(N, fixed_z, dtype=np.float64)
    if mode == "normal":
        return rng.normal(loc=0.0, scale=3.0, size=N).astype(np.float64)
    if mode == "uniform":
        return rng.uniform(-8.0, 8.0, size=N).astype(np.float64)
    if mode == "grid":
        return np.linspace(-8.0, 8.0, N, dtype=np.float64)
    raise ValueError(f"Unknown z mode: {mode}")


def rpg1_polyagamma_cpu(z: np.ndarray, seed: int) -> np.ndarray:
    if random_polyagamma is None:
        raise RuntimeError("polyagamma is not installed. Try: pip install polyagamma")
    rng = np.random.default_rng(seed)
    return random_polyagamma(h=1.0, z=z, random_state=rng)


def rpg1_trunc_cupy(
    z: np.ndarray,
    K: int = 200,
    batch_size: int = 200_000,
    seed: int = 123,
    dtype: str = "float32",
) -> np.ndarray:
    """Approximate PG(1,z) using a GPU-truncated gamma/exponential series."""
    if cp is None:
        raise RuntimeError("CuPy is not installed. Try: pip install cupy-cuda12x")

    xp_dtype = cp.float32 if dtype == "float32" else cp.float64
    out = np.empty(z.shape[0], dtype=np.float32 if dtype == "float32" else np.float64)

    # RandomState is broadly compatible across CuPy versions.
    rng = cp.random.RandomState(seed)
    k = cp.arange(1, K + 1, dtype=xp_dtype)
    base = (k - xp_dtype(0.5)) ** 2
    two_pi_sq_inv = xp_dtype(1.0 / (2.0 * math.pi * math.pi))
    four_pi_sq_inv = xp_dtype(1.0 / (4.0 * math.pi * math.pi))

    for start in range(0, z.shape[0], batch_size):
        end = min(start + batch_size, z.shape[0])
        z_gpu = cp.asarray(z[start:end], dtype=xp_dtype)
        # denom: [B, K]
        denom = base[None, :] + (z_gpu[:, None] * z_gpu[:, None]) * four_pi_sq_inv
        # Gamma(1,1) == Exp(1). Generated on GPU.
        e = rng.exponential(scale=1.0, size=(end - start, K)).astype(xp_dtype, copy=False)
        vals = cp.sum(e / denom, axis=1) * two_pi_sq_inv
        out[start:end] = cp.asnumpy(vals)

    cp.cuda.Stream.null.synchronize()
    return out


def time_call(fn, *args, repeats: int = 1, sync_gpu: bool = False, **kwargs):
    times = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        if sync_gpu and cp is not None:
            cp.cuda.Stream.null.synchronize()
        times.append(time.perf_counter() - t0)
    return min(times), result


def benchmark(args) -> list[BenchRow]:
    rows: list[BenchRow] = []
    for N in args.sizes:
        z = make_z(N, args.z_mode, args.fixed_z, args.seed)
        theory_mean = float(pg1_mean(z).mean())

        cpu_sec = None
        cpu_samples = None
        cpu_mean = None
        if not args.no_cpu_ref and random_polyagamma is not None:
            cpu_sec, cpu_samples = time_call(rpg1_polyagamma_cpu, z, args.seed, repeats=args.repeats)
            cpu_mean = float(cpu_samples.mean())

        for K in args.Ks:
            # Warmup
            _ = rpg1_trunc_cupy(z[: min(N, 2048)], K=K, batch_size=args.batch_size,
                                seed=args.seed, dtype=args.dtype)
            gpu_sec, gpu_samples = time_call(
                rpg1_trunc_cupy,
                z,
                K=K,
                batch_size=args.batch_size,
                seed=args.seed,
                dtype=args.dtype,
                repeats=args.repeats,
                sync_gpu=True,
            )
            gpu_mean = float(gpu_samples.mean())
            rows.append(
                BenchRow(
                    N=N,
                    K=K,
                    cpu_sec=cpu_sec,
                    gpu_sec=gpu_sec,
                    speedup=(cpu_sec / gpu_sec if cpu_sec is not None else None),
                    cpu_mean=cpu_mean,
                    gpu_mean=gpu_mean,
                    theory_mean=theory_mean,
                    gpu_abs_mean_err=abs(gpu_mean - theory_mean),
                )
            )
    return rows


def quality_check(args) -> None:
    if random_polyagamma is None:
        print("[quality] skipped: polyagamma not installed")
        return
    if ks_2samp is None:
        print("[quality] scipy not installed; KS statistic skipped")

    print("\n[quality check: fixed z distributions]")
    header = "z       K      cpu_mean   gpu_mean   theory    abs_err   ks_stat   ks_p"
    print(header)
    print("-" * len(header))
    for z0 in args.quality_z:
        z = np.full(args.quality_n, z0, dtype=np.float64)
        cpu = rpg1_polyagamma_cpu(z, args.seed)
        theory = float(pg1_mean(z).mean())
        for K in args.Ks:
            gpu = rpg1_trunc_cupy(z, K=K, batch_size=args.batch_size, seed=args.seed, dtype=args.dtype)
            if ks_2samp is not None:
                ks = ks_2samp(cpu, gpu)
                ks_stat, ks_p = float(ks.statistic), float(ks.pvalue)
            else:
                ks_stat, ks_p = float("nan"), float("nan")
            print(
                f"{z0:>6.2f}  {K:>5d}  {cpu.mean():>9.5f}  {gpu.mean():>9.5f}  "
                f"{theory:>8.5f}  {abs(gpu.mean() - theory):>8.2e}  "
                f"{ks_stat:>7.4f}  {ks_p:>7.2g}"
            )


def print_benchmark(rows: Iterable[BenchRow]) -> None:
    print("\n[benchmark]")
    header = "N          K      CPU(s)     GPU(s)   speedup   cpu_mean  gpu_mean  theory   |gpu-theory|"
    print(header)
    print("-" * len(header))
    for r in rows:
        cpu_s = "   n/a" if r.cpu_sec is None else f"{r.cpu_sec:8.4f}"
        sp = "   n/a" if r.speedup is None else f"{r.speedup:7.2f}x"
        cm = "   n/a" if r.cpu_mean is None else f"{r.cpu_mean:8.5f}"
        print(
            f"{r.N:<10d} {r.K:<5d} {cpu_s}  {r.gpu_sec:8.4f}  {sp}  "
            f"{cm}  {r.gpu_mean:8.5f}  {r.theory_mean:7.5f}  {r.gpu_abs_mean_err:11.2e}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=[1_000, 10_000, 100_000, 1_000_000])
    p.add_argument("--Ks", type=int, nargs="+", default=[50, 100, 200])
    p.add_argument("--z-mode", choices=["grid", "normal", "uniform"], default="normal")
    p.add_argument("--fixed-z", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=200_000)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--quality-n", type=int, default=50_000)
    p.add_argument("--quality-z", type=float, nargs="+", default=[0.0, 1.0, 3.0, 7.0])
    p.add_argument("--skip-quality", action="store_true")
    p.add_argument("--skip-benchmark", action="store_true")
    p.add_argument("--no-cpu-ref", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if cp is None:
        raise SystemExit("CuPy is required for the GPU toy. Install e.g. `pip install cupy-cuda12x`.")
    print(f"CuPy version: {cp.__version__}")
    print(f"GPU device:   {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"dtype:        {args.dtype}")
    print(f"Ks:           {args.Ks}")

    if not args.skip_quality:
        quality_check(args)
    if not args.skip_benchmark:
        rows = benchmark(args)
        print_benchmark(rows)


if __name__ == "__main__":
    main()
