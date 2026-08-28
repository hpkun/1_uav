"""Reusable, metadata-complete evaluation of MAPPO checkpoints."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch

from algorithm.common.checkpoint import validate_checkpoint_for_evaluation
from algorithm.common.evaluator import evaluate
from .factory import build_mappo_trainer


def evaluate_mappo_checkpoint(
    checkpoint_path: str | Path,
    algorithm_config: dict[str, Any],
    environment_config: dict[str, Any],
    device: str,
    evaluation_seeds: Iterable[int],
    allow_cross_variant: bool = False,
) -> dict[str, Any]:
    """Load one checkpoint and evaluate it under an explicit target protocol."""
    checkpoint = Path(checkpoint_path)
    seeds = [int(seed) for seed in evaluation_seeds]
    if not seeds:
        raise ValueError("evaluation_seeds must not be empty")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint_for_evaluation(
        state,
        environment_config,
        algorithm_config,
        allow_cross_variant=allow_cross_variant,
    )
    trainer = build_mappo_trainer(algorithm_config, device)
    trainer.load(checkpoint)
    result: dict[str, Any] = dict(
        evaluate(trainer, environment_config, seeds)
    )
    extra = state.get("extra", {})
    checkpoint_variant = str(extra.get("environment_variant", "direct_v2_3"))
    evaluation_variant = str(
        environment_config.get("environment_variant", "direct_v2_3")
    )
    network = algorithm_config["network"]
    result.update({
        "algorithm": "MAPPO",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_environment_version": extra.get("environment_version"),
        "checkpoint_environment_variant": checkpoint_variant,
        "evaluation_environment_version": str(
            environment_config["environment_version"]
        ),
        "evaluation_environment_variant": evaluation_variant,
        "cross_variant_evaluation": checkpoint_variant != evaluation_variant,
        "mappo_impl_version": state.get("mappo_impl_version"),
        "holdout_seed_base": seeds[0],
        "holdout_seed_end": seeds[-1],
        "evaluation_episodes": len(seeds),
        "device": str(device),
        "observation_dim": int(network["observation_dim"]),
        "action_dim": int(network["action_dim"]),
        "num_agents": int(network["num_agents"]),
    })
    return result


__all__ = ["evaluate_mappo_checkpoint"]
