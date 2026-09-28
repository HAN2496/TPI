"""Measure long-horizon robustness and gait signals for saved PPO checkpoints."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR.parents[1]))

from pipeline.common import load_config
from pipeline.linear_experiment import evaluate


def _screen(payload):
    path, config, start, episodes = payload
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    policy = PPO.load(path, device="cpu")
    metrics = evaluate(policy, config, {}, range(start, start + episodes))
    return {"path": str(path), **metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("patterns", nargs="+", help="Glob patterns relative to preference_sim")
    parser.add_argument("--config", default="configs/walker2d_linear_release.yaml")
    parser.add_argument("--seed", type=int, default=691000)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    config, _ = load_config(args.config)
    paths = sorted({path.resolve() for pattern in args.patterns for path in THIS_DIR.glob(pattern)})
    payload = [(path, config, args.seed, args.episodes) for path in paths]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(_screen, payload))
    rows.sort(key=lambda row: (-row["completion_rate"], -row["speed"]))
    output = json.dumps(rows, indent=2)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
