"""MAPPO YAML configuration loading and validation."""

from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import yaml
from uav_env.utils.config import deep_merge, project_root


def load_mappo_config(path: str | Path) -> dict[str, Any]:
    selected=Path(path); selected=selected if selected.is_absolute() else project_root()/selected
    base=project_root()/"configs"/"mappo_base.yaml"
    with base.open(encoding="utf-8") as stream: base_data=yaml.safe_load(stream)
    with selected.open(encoding="utf-8") as stream: selected_data=yaml.safe_load(stream)
    config=deep_merge(base_data,selected_data); config.pop("inherits",None)
    required=("seed","num_envs","rollout_length","total_env_steps","gamma","gae_lambda","actor_lr","critic_lr")
    if any(key not in config for key in required): raise ValueError("Incomplete MAPPO configuration")
    validate_mappo_config(config)
    return config


def validate_mappo_config(config: dict[str, Any]) -> None:
    """Reject mathematically invalid PPO and network settings before a run starts."""

    for key in ("ppo_epochs", "num_mini_batches", "num_envs", "rollout_length"):
        if not isinstance(config.get(key), int) or int(config[key]) <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("validation_seed_start", "validation_episodes", "test_seed_start", "test_episodes"):
        if not isinstance(config.get(key), int) or int(config[key]) < 0 or (key.endswith("episodes") and int(config[key]) == 0):
            raise ValueError(f"{key} must be a valid nonnegative integer and episode counts must be positive")
    validation_range=set(range(int(config["validation_seed_start"]),int(config["validation_seed_start"])+int(config["validation_episodes"])))
    test_range=set(range(int(config["test_seed_start"]),int(config["test_seed_start"])+int(config["test_episodes"])))
    if validation_range & test_range:
        raise ValueError("validation and test seed ranges must not overlap")
    if config.get("checkpoint_selection") not in {"smoke","combat"}:
        raise ValueError("checkpoint_selection must be smoke or combat")
    agents = 1 if config.get("environment", {}).get("kind") == "1v1" else 2
    total_samples = int(config["num_envs"]) * int(config["rollout_length"]) * agents
    if int(config["num_mini_batches"]) > total_samples:
        raise ValueError(f"num_mini_batches cannot exceed rollout samples ({total_samples})")
    for key in ("clip_param", "value_clip_param"):
        if not np.isfinite(float(config[key])) or float(config[key]) < 0.0:
            raise ValueError(f"{key} must be finite and nonnegative")
    for key in ("gamma", "gae_lambda"):
        if not 0.0 <= float(config[key]) <= 1.0:
            raise ValueError(f"{key} must be in [0, 1]")
    for key in ("actor_lr", "critic_lr"):
        if not np.isfinite(float(config[key])) or float(config[key]) <= 0.0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("actor_hidden_sizes", "critic_hidden_sizes"):
        sizes = config.get(key)
        if not isinstance(sizes, list) or not sizes or any(not isinstance(size, int) or size <= 0 for size in sizes):
            raise ValueError(f"{key} must contain positive integers")
