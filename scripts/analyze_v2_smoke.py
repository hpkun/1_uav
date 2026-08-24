"""Summarize V2.2 training metrics and deterministic checkpoint evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch

from uav_combat.config import ENVIRONMENT_VERSION
from uav_combat.environment.observation import OBSERVATION_DIM
from uav_combat.madsac import MADSACTrainer
from uav_combat.training.evaluator import evaluate


def distribution(values) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()), "std": float(array.std()),
        "min": float(array.min()), "max": float(array.max()),
        "nonzero_rate": float(np.mean(array != 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output", default="outputs/v2_2_smoke_analysis.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    checkpoint = (root / args.checkpoint).resolve()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    version = state.get("extra", {}).get("environment_version")
    if version != ENVIRONMENT_VERSION:
        raise RuntimeError(f"checkpoint environment_version is {version!r}, expected 2.2")
    hidden = int(state["actor"]["backbone.0.weight"].shape[0])
    trainer = MADSACTrainer(observation_dim=OBSERVATION_DIM, hidden_dim=hidden)
    trainer.load(checkpoint)
    rows = [json.loads(line) for line in (root / args.metrics).read_text().splitlines()]
    channels = [
        "mean_step_reward", "mean_r1_reward", "mean_r2_reward",
        "mean_r3_reward", "mean_r4_reward", "red_fire_window_pairs",
        "blue_fire_window_pairs", "red_step_fire_attempts",
        "blue_step_fire_attempts", "red_step_weapon_hits", "blue_step_weapon_hits",
    ]
    report = {
        "environment_version": version,
        "sampled_steps": rows[-1]["sampled_steps"] if rows else 0,
        "training_channels": {
            key: distribution([row[key] for row in rows]) for key in channels
        },
        "evaluation": evaluate(
            trainer, root / "configs/combat_environment.yaml",
            seeds=range(10_000_000, 10_000_020),
        ),
    }
    output = (root / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
