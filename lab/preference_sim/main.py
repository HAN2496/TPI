from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
for path in (THIS_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pipeline.collect_rollouts import collect_policy_bank
from pipeline.build_blends import build_action_blend_bank
from pipeline.common import config_digest, load_config, resolve_run_dir, runtime_versions, write_json
from pipeline.export_fully_bayesian import export_fully_bayesian
from pipeline.plot_reports import plot_run_reports
from pipeline.record_policy_videos import record_policy_videos
from pipeline.select_policies import select_policy_checkpoints
from pipeline.simulate_users import simulate_users_and_feedback
from pipeline.train_policies import train_policy_bank
from pipeline.validate_bayesian import validate_bayesian_export

STAGES = ("train", "blend", "select", "collect", "users", "reports", "export", "validate", "videos")
DEFAULT_STAGES = ("train", "collect", "users", "reports", "export", "validate", "videos")


def main() -> None:
    parser = argparse.ArgumentParser(description="Walker2d preference-simulation pipeline.")
    parser.add_argument(
        "stages", nargs="*", default=list(DEFAULT_STAGES), metavar="stage",
        help=f"stages to run, in pipeline order (default: all). Choices: {', '.join(STAGES)}",
    )
    parser.add_argument("--config", default=None, help="Config YAML (default: the run's config.yaml, else configs/walker2d_style_smoke.yaml)")
    parser.add_argument("--run-id", default=None, help="Run directory name (default: config name, suffixed if taken)")
    parser.add_argument("--source-run", default=None, help="Copy policies from this run instead of training")
    parser.add_argument("--video-episodes", type=int, default=2)
    parser.add_argument("--all-checkpoints", action="store_true", help="Render videos for every checkpoint")
    args = parser.parse_args()

    unknown = set(args.stages) - set(STAGES)
    if unknown:
        parser.error(f"Unknown stages: {sorted(unknown)}")
    stages = [stage for stage in STAGES if stage in args.stages]
    if args.source_run and "train" in stages:
        stages.remove("train")
    fresh = "train" in stages or "blend" in stages or bool(args.source_run)

    config_path = args.config
    if config_path is None:
        candidate = THIS_DIR / "data" / "runs" / (args.run_id or "") / "config.yaml"
        config_path = candidate if args.run_id and candidate.exists() else "configs/walker2d_style_smoke.yaml"
    config, config_path = load_config(config_path)
    run_dir = resolve_run_dir(config, args.run_id, fresh)
    if fresh:
        shutil.copy2(config_path, run_dir / "config.yaml")
    if args.source_run:
        source = Path(args.source_run)
        if not source.exists():
            source = THIS_DIR / "data" / "runs" / args.source_run
        shutil.copytree(source.resolve() / "policies", run_dir / "policies", dirs_exist_ok=True)

    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    for key in ("started_at", "finished_at"):
        manifest.pop(key, None)
    manifest.update(
        {
            "status": "running", "stages": stages,
            "config_path": str(config_path), "config_digest": config_digest(config),
            "runtime": runtime_versions(),
        }
    )
    if args.source_run:
        manifest["source_policy_run"] = str(source.resolve())
    write_json(manifest_path, manifest)

    try:
        for stage in stages:
            print(f"[{stage}]")
            if stage == "train":
                manifest["n_checkpoints"] = len(train_policy_bank(config, run_dir))
            elif stage == "blend":
                manifest["n_checkpoints"] = len(build_action_blend_bank(config, run_dir))
            elif stage == "select":
                manifest["selection"] = select_policy_checkpoints(config, run_dir)
            elif stage == "collect":
                manifest["n_episodes"] = int(len(collect_policy_bank(config, run_dir)))
            elif stage == "users":
                users, feedback = simulate_users_and_feedback(config, run_dir)
                manifest["n_users"], manifest["n_feedback"] = int(len(users)), int(len(feedback))
            elif stage == "reports":
                plot_run_reports(run_dir)
            elif stage == "export":
                manifest["export"] = str(export_fully_bayesian(config, run_dir).relative_to(run_dir))
            elif stage == "validate":
                manifest["validation"] = validate_bayesian_export(config, run_dir)
            elif stage == "videos":
                record_policy_videos(config, run_dir, args.video_episodes, args.all_checkpoints)
        manifest["status"] = "complete"
    except Exception as error:
        manifest.update({"status": "failed", "error": repr(error), "traceback": traceback.format_exc()})
        raise
    finally:
        write_json(manifest_path, manifest)

    print(f"Run complete: {run_dir}")
    if "validation" in manifest:
        print(f"Validation: {manifest['validation']['test']}")


if __name__ == "__main__":
    main()
