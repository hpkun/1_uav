"""Versioned exact MAPPO checkpoint persistence."""

from __future__ import annotations
import random
from pathlib import Path
from typing import Any
import numpy as np
import torch

CHECKPOINT_VERSION=3
INCOMPATIBLE_V2_MESSAGE = "v2 critic value semantics are incompatible with v3 physical-value critic"

def schema_metadata(config: dict[str,Any], obs_dim: int | None = None, state_dim: int | None = None, num_agents: int | None = None) -> dict[str,Any]:
    env = config.get("environment", {}) if isinstance(config.get("environment", {}), dict) else {}
    return {
        "environment_schema_version": env.get("environment_schema_version") or config.get("environment_schema_version") or "legacy",
        "observation_schema": env.get("observation_schema") or config.get("observation_schema") or "legacy",
        "global_state_schema": env.get("global_state_schema") or config.get("global_state_schema") or "legacy",
        "reward_profile": env.get("reward_profile") or config.get("reward_profile") or "legacy",
        "scenario_profile": env.get("scenario_profile") or config.get("scenario_profile") or "legacy",
        "obs_dim": obs_dim,
        "state_dim": state_dim,
        "num_agents": num_agents,
    }

def save_checkpoint(path: Path, actor, critic, actor_optimizer, critic_optimizer, normalizer, config: dict[str,Any], environment_steps: int, update_index: int, best_evaluation: dict[str,Any] | None, runner_state: dict[str, Any] | None = None, metadata: dict[str,Any] | None = None) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    torch.save({"version":CHECKPOINT_VERSION,"schema_metadata":metadata or schema_metadata(config),"actor":actor.state_dict(),"critic":critic.state_dict(),"actor_optimizer":actor_optimizer.state_dict(),"critic_optimizer":critic_optimizer.state_dict(),"value_normalizer":normalizer.state_dict(),"config":config,"environment_steps":environment_steps,"update_index":update_index,"best_evaluation":best_evaluation,"runner_state":runner_state,"python_rng":random.getstate(),"numpy_rng":np.random.get_state(),"torch_rng":torch.get_rng_state(),"torch_cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},temporary)
    temporary.replace(path)

def load_checkpoint(path: str|Path, actor, critic=None, actor_optimizer=None, critic_optimizer=None, normalizer=None, actor_only: bool=False, map_location="cpu", expected_metadata: dict[str,Any] | None = None) -> dict[str,Any]:
    data=torch.load(path,map_location=map_location,weights_only=False)
    version=int(data.get("version",1))
    if not actor_only and version < CHECKPOINT_VERSION:
        if version == 2: raise ValueError(INCOMPATIBLE_V2_MESSAGE)
        raise ValueError(f"checkpoint v{version} is incompatible with v3 physical-value critic")
    if version > CHECKPOINT_VERSION: raise ValueError(f"checkpoint v{version} is newer than supported v{CHECKPOINT_VERSION}")
    if not actor_only and expected_metadata is not None:
        if "schema_metadata" not in data:
            legacy_expected = all(expected_metadata.get(key) == "legacy" for key in ("environment_schema_version", "observation_schema", "global_state_schema"))
            if not legacy_expected:
                target = expected_metadata.get("environment_schema_version")
                raise ValueError(f"legacy checkpoint without schema metadata cannot resume into {target}")
        else:
            actual=data["schema_metadata"]
            for key in ("environment_schema_version","observation_schema","global_state_schema","reward_profile","scenario_profile","obs_dim","state_dim","num_agents"):
                if actual.get(key) != expected_metadata.get(key):
                    raise ValueError(f"checkpoint schema mismatch for {key}: checkpoint={actual.get(key)!r}, expected={expected_metadata.get(key)!r}")
    try: actor.load_state_dict(data["actor"])
    except RuntimeError as error: raise ValueError(f"Actor dimensions are incompatible: {error}") from error
    if not actor_only:
        if critic is None or normalizer is None: raise ValueError("Full resume requires critic and normalizer")
        try: critic.load_state_dict(data["critic"])
        except RuntimeError as error: raise ValueError(f"Critic dimensions are incompatible: {error}") from error
        normalizer.load_state_dict(data["value_normalizer"])
        if actor_optimizer is not None: actor_optimizer.load_state_dict(data["actor_optimizer"])
        if critic_optimizer is not None: critic_optimizer.load_state_dict(data["critic_optimizer"])
        random.setstate(data["python_rng"]); np.random.set_state(data["numpy_rng"]); torch.set_rng_state(data["torch_rng"].cpu())
        if torch.cuda.is_available() and data.get("torch_cuda_rng"): torch.cuda.set_rng_state_all([state.cpu() for state in data["torch_cuda_rng"]])
    return data
