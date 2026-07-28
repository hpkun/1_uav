"""Checkpoint persistence for independent-actor HAPPO."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from uav_env.algorithms.mappo.checkpoint import schema_metadata

HAPPO_CHECKPOINT_VERSION = 1


def save_happo_checkpoint(
    path: Path,
    actors,
    critic,
    actor_optimizers,
    critic_optimizer,
    normalizer,
    config: dict[str, Any],
    environment_steps: int,
    update_index: int,
    best_evaluation: dict[str, Any] | None,
    runner_state: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    """Atomically save all non-shared HAPPO training state."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "algorithm": "happo",
            "version": HAPPO_CHECKPOINT_VERSION,
            "schema_metadata": metadata,
            "actor_count": len(actors),
            "actors": [actor.state_dict() for actor in actors.actors],
            "actor_optimizers": [optimizer.state_dict() for optimizer in actor_optimizers],
            "critic": critic.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict(),
            "value_normalizer": normalizer.state_dict(),
            "config": config,
            "environment_steps": environment_steps,
            "update_index": update_index,
            "best_evaluation": best_evaluation,
            "runner_state": runner_state,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "torch_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        temporary,
    )
    temporary.replace(path)


def load_happo_checkpoint(
    path: str | Path,
    actors,
    critic=None,
    actor_optimizers=None,
    critic_optimizer=None,
    normalizer=None,
    actor_only: bool = False,
    map_location: str | torch.device = "cpu",
    expected_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load a HAPPO checkpoint; MAPPO checkpoints are intentionally rejected."""

    data = torch.load(path, map_location=map_location, weights_only=False)
    if data.get("algorithm") != "happo":
        raise ValueError("HAPPO runner can only load HAPPO checkpoints; MAPPO conversion is unsupported")
    if int(data.get("version", 0)) > HAPPO_CHECKPOINT_VERSION:
        raise ValueError("HAPPO checkpoint is newer than this implementation")
    if int(data.get("actor_count", -1)) != len(actors):
        raise ValueError(f"HAPPO actor count mismatch: checkpoint={data.get('actor_count')}, expected={len(actors)}")
    if expected_metadata is not None:
        actual = data.get("schema_metadata", {})
        for key in ("environment_schema_version", "observation_schema", "global_state_schema", "reward_profile", "scenario_profile", "obs_dim", "state_dim", "num_agents"):
            if actual.get(key) != expected_metadata.get(key):
                raise ValueError(f"HAPPO checkpoint schema mismatch for {key}: checkpoint={actual.get(key)!r}, expected={expected_metadata.get(key)!r}")
    for index, actor_state in enumerate(data["actors"]):
        try:
            actors[index].load_state_dict(actor_state)
        except RuntimeError as error:
            raise ValueError(f"HAPPO actor_{index} dimensions are incompatible: {error}") from error
    if actor_only:
        return data
    if critic is None or normalizer is None:
        raise ValueError("Full HAPPO resume requires critic and normalizer")
    try:
        critic.load_state_dict(data["critic"])
    except RuntimeError as error:
        raise ValueError(f"HAPPO critic dimensions are incompatible: {error}") from error
    normalizer.load_state_dict(data["value_normalizer"])
    if actor_optimizers is not None:
        for optimizer, state in zip(actor_optimizers, data["actor_optimizers"]):
            optimizer.load_state_dict(state)
    if critic_optimizer is not None:
        critic_optimizer.load_state_dict(data["critic_optimizer"])
    random.setstate(data["python_rng"])
    np.random.set_state(data["numpy_rng"])
    torch.set_rng_state(data["torch_rng"].cpu())
    if torch.cuda.is_available() and data.get("torch_cuda_rng"):
        torch.cuda.set_rng_state_all([state.cpu() for state in data["torch_cuda_rng"]])
    return data


__all__ = ["save_happo_checkpoint", "load_happo_checkpoint", "schema_metadata"]
