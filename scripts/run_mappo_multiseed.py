"""Run reproducible MAPPO seeds sequentially and evaluate matched checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from uav_env.algorithms.mappo.config import load_mappo_config, validate_mappo_config
from uav_env.algorithms.mappo.runner import MAPPORunner


def _evaluate_checkpoints(runner: MAPPORunner, run_dir: Path, episodes: int, seed_start: int) -> dict[str, Any]:
    evaluations: dict[str, Any] = {}
    for label in ("initial", "last", "best"):
        checkpoint = run_dir / "checkpoints" / f"{label}.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        runner.resume(str(checkpoint), actor_only=True)
        evaluations[label] = runner.evaluate(episodes, seed_start, deterministic=True)
    return evaluations


def run_seed(config_path: str, seed: int, total_steps: int, evaluation_episodes: int, evaluation_seed_start: int, device: str, output_root: Path, resume_missing: bool) -> Path:
    config = load_mappo_config(config_path)
    config.update(seed=seed, total_env_steps=total_steps, evaluation_episodes=evaluation_episodes, device=device, run_id="run")
    validate_mappo_config(config)
    seed_root = output_root / f"seed_{seed}"
    run_dir = seed_root / "run"
    summary_path = run_dir / "seed_summary.yaml"
    if resume_missing and summary_path.exists():
        print(f"seed {seed}: completed, skipping")
        return summary_path
    runner = MAPPORunner(config, "", output_root=seed_root)
    last = run_dir / "checkpoints" / "last.pt"
    if resume_missing and last.exists():
        print(f"seed {seed}: resuming {last}")
        try:
            runner.resume(str(last))
        except (EOFError, OSError, RuntimeError) as error:
            marker=last.stat().st_mtime_ns
            preserved=last.with_name(f"last.corrupt.{marker}.pt")
            last.replace(preserved)
            metrics=run_dir/"metrics.csv"
            if metrics.exists(): metrics.replace(run_dir/f"metrics.interrupted.{marker}.csv")
            print(f"seed {seed}: preserved unreadable checkpoint as {preserved}; restarting seed ({error})")
    completed_dir = runner.run()
    evaluations = _evaluate_checkpoints(runner, completed_dir, evaluation_episodes, evaluation_seed_start)
    payload = {
        "seed": seed,
        "config": str(Path(config_path)),
        "total_env_steps": total_steps,
        "evaluation_episodes": evaluation_episodes,
        "evaluation_seed_start": evaluation_seed_start,
        "terminal_reward_profile": config["environment"].get("multi_terminal_reward_profile", "not_applicable"),
        "run_dir": str(completed_dir.resolve()),
        "checkpoints": {name: str((completed_dir / "checkpoints" / f"{name}.pt").resolve()) for name in ("initial", "last", "best")},
        "evaluations": evaluations,
    }
    summary_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(summary_path.resolve())
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--total-env-steps", type=int, required=True)
    parser.add_argument("--evaluation-episodes", type=int, required=True)
    parser.add_argument("--evaluation-seed-start", type=int, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--resume-missing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.workers != 1:
        print("Single-device safety: seeds are intentionally executed sequentially; --workers is reserved for CPU orchestration.")
    root = Path("outputs/mappo_multiseed") / args.output_name
    for seed in args.seeds:
        run_seed(args.config, seed, args.total_env_steps, args.evaluation_episodes, args.evaluation_seed_start, args.device, root, args.resume_missing)


if __name__ == "__main__":
    main()
