"""Run one bounded, one-factor reconstruction sensitivity profile."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml

from uav_combat.training.runner import PaperTrainingRunner


MAX_AUTOMATED_SENSITIVITY_STEPS = 200_000
GROUPS = ("weapon", "sensor", "controller", "scheduler")


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def apply_profile(
    environment: dict[str, Any],
    algorithm: dict[str, Any],
    candidates: dict[str, Any],
    group: str,
    profile: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if group not in GROUPS:
        raise ValueError(f"unknown sensitivity group: {group}")
    profiles = candidates.get(group, {})
    if profile not in profiles:
        raise ValueError(f"profile {profile!r} is not in group {group!r}")
    selected = profiles[profile]
    return (
        deep_merge(environment, selected.get("environment", {})),
        deep_merge(algorithm, selected.get("algorithm", {})),
    )


def validate_sampled_steps(sampled_steps: int) -> int:
    sampled_steps = int(sampled_steps)
    if sampled_steps <= 0:
        raise ValueError("sampled_steps must be positive")
    if sampled_steps > MAX_AUTOMATED_SENSITIVITY_STEPS:
        raise ValueError(
            "This is a long-run command and must be executed manually on the Ubuntu server."
        )
    return sampled_steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", required=True, choices=GROUPS)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--sampled-steps", type=int, default=24_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--output-dir", default="outputs/sensitivity")
    args = parser.parse_args()

    sampled_steps = validate_sampled_steps(args.sampled_steps)
    root = Path(__file__).resolve().parents[1]
    environment = yaml.safe_load((root / "configs/paper_environment.yaml").read_text(encoding="utf-8"))
    algorithm = yaml.safe_load((root / "configs/madsac.yaml").read_text(encoding="utf-8"))
    candidates = yaml.safe_load((root / "configs/sensitivity_candidates.yaml").read_text(encoding="utf-8"))
    environment, algorithm = apply_profile(
        environment, algorithm, candidates, args.group, args.profile
    )
    output_dir = Path(args.output_dir) / args.group / args.profile
    runner = PaperTrainingRunner(
        environment,
        algorithm,
        total_sampled_steps=sampled_steps,
        device=args.device,
        seed=args.seed,
        output_dir=output_dir,
    )
    print(json.dumps({"profile": args.profile, "startup": runner.startup_summary()}, indent=2))
    summary = runner.run()
    (runner.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
