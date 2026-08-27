"""Holdout evaluation for a formal MAPPO checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml

from algorithm.common.checkpoint import validate_checkpoint_environment
from algorithm.common.evaluator import evaluate
from algorithm.mappo.factory import build_mappo_trainer


def resolved(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=("mappo",), default="mappo")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env-config", required=True)
    parser.add_argument("--algorithm-config", required=True)
    parser.add_argument("--seed-base", type=int, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")

    env_config = yaml.safe_load(resolved(args.env_config).read_text(encoding="utf-8"))
    algorithm_config = yaml.safe_load(
        resolved(args.algorithm_config).read_text(encoding="utf-8")
    )
    checkpoint = resolved(args.checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint_environment(state, env_config)
    trainer = build_mappo_trainer(algorithm_config, args.device)
    trainer.load(checkpoint)
    seeds = range(args.seed_base, args.seed_base + args.episodes)
    result = evaluate(trainer, env_config, seeds)
    result.update({
        "algorithm": "MAPPO",
        "checkpoint": str(checkpoint),
        "holdout_seed_base": args.seed_base,
    })
    output = resolved(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
