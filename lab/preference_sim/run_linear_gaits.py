"""Run the linear-reward Walker2d experiment; legacy main.py is retained."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parents[1]))
sys.path.insert(0, str(THIS_DIR))

from pipeline.common import config_digest, load_config, resolve_run_dir, runtime_versions, write_json
from pipeline import linear_experiment as experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stages", nargs="*", default=["train", "select", "collect", "users", "validate", "videos"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    allowed = ("train", "select", "collect", "users", "validate", "videos")
    if set(args.stages) - set(allowed):
        parser.error(f"Stages must be in {allowed}")
    previous = THIS_DIR / "data" / "runs" / (args.run_id or "") / "config.yaml"
    config, source = load_config(args.config or (previous if previous.is_file() else "configs/walker2d_role_exchange.yaml"))
    # Validate all disjoint namespaces before expensive simulation.
    collection = config["collection"]
    seed_sets = [set(range(config["training"]["evaluation_seed"], config["training"]["evaluation_seed"] + config["training"]["evaluation_episodes"])),
                 set(range(config["selection"]["seed"], config["selection"]["seed"] + config["selection"]["episodes"]))]
    for split in ("calibration", "context", "test"):
        start = collection[f"{split}_seed"]
        seed_sets.append(set(range(start, start + collection[f"{split}_episodes"])) |
                         set(range(start + 10000, start + 10000 + collection["diagnostic_episodes"])))
    if any(left & right for i, left in enumerate(seed_sets) for right in seed_sets[i+1:]):
        raise ValueError("Screen, selection, calibration, context and test seeds must be disjoint")
    fresh = "train" in args.stages
    run_dir = resolve_run_dir(config, args.run_id, fresh=fresh)
    if fresh:
        shutil.copy2(source, run_dir / "config.yaml")
    experiment.torch.set_num_threads(1)
    manifest_path = run_dir / "manifest.json"
    manifest = experiment.read_json(manifest_path) if manifest_path.is_file() else {}
    manifest.update(status="running", runtime=runtime_versions(), stages=args.stages,
                    reward_model=f"linear_phi_v{config['environment'].get('feature_version', 1)}",
                    config_path=str(source), config_digest=config_digest(config))
    write_json(manifest_path, manifest)
    try:
        for stage in allowed:
            if stage in args.stages:
                print(f"[{stage}] {run_dir.name}", flush=True)
                result = getattr(experiment, stage)(config, run_dir)
                if isinstance(result, dict):
                    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
                manifest.setdefault("completed_stages", [])
                if stage not in manifest["completed_stages"]:
                    manifest["completed_stages"].append(stage)
                write_json(manifest_path, manifest)
        manifest["status"] = "complete"
    except Exception:
        manifest.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        write_json(manifest_path, manifest)
    print(f"Run complete: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
