"""HAPPO YAML configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from uav_env.utils.config import deep_merge, project_root


def load_happo_config(path: str | Path) -> dict[str, Any]:
    selected = Path(path)
    selected = selected if selected.is_absolute() else project_root() / selected
    base = project_root() / "configs" / "happo_base.yaml"
    with base.open(encoding="utf-8") as stream:
        base_data = yaml.safe_load(stream)
    with selected.open(encoding="utf-8") as stream:
        selected_data = yaml.safe_load(stream)
    config = deep_merge(base_data, selected_data)
    config.pop("inherits", None)
    validate_happo_config(config)
    return config


def validate_happo_config(config: dict[str, Any]) -> None:
    if config.get("algorithm") != "happo":
        raise ValueError("HAPPO config must set algorithm: happo")
    if config.get("reward_mode") != "joint_team":
        raise ValueError("This paper-aligned HAPPO baseline supports only reward_mode: joint_team")
    if bool(config.get("share_actor_parameters", False)):
        raise ValueError("HAPPO requires share_actor_parameters: false")
    for key in ("ppo_epochs", "actor_num_mini_batches", "critic_epochs", "critic_num_mini_batches", "num_envs", "rollout_length"):
        if not isinstance(config.get(key), int) or int(config[key]) <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("validation_seed_start", "validation_episodes", "test_seed_start", "test_episodes"):
        if not isinstance(config.get(key), int) or int(config[key]) < 0 or (key.endswith("episodes") and int(config[key]) == 0):
            raise ValueError(f"{key} must be a valid nonnegative integer and episode counts must be positive")
    validation_range = set(range(int(config["validation_seed_start"]), int(config["validation_seed_start"]) + int(config["validation_episodes"])))
    test_range = set(range(int(config["test_seed_start"]), int(config["test_seed_start"]) + int(config["test_episodes"])))
    if validation_range & test_range:
        raise ValueError("validation and test seed ranges must not overlap")
    if config.get("checkpoint_selection") not in {"smoke", "combat"}:
        raise ValueError("checkpoint_selection must be smoke or combat")
    if config.get("vector_env") not in {"sync", "parallel"}:
        raise ValueError("vector_env must be sync or parallel")
    kind = config.get("environment", {}).get("kind")
    if kind != "3v3":
        raise ValueError("This HAPPO baseline is scoped to fixed 3v3")
    if int(config["actor_num_mini_batches"]) > int(config["num_envs"]) * int(config["rollout_length"]):
        raise ValueError("actor_num_mini_batches cannot exceed rollout samples")
    if int(config["critic_num_mini_batches"]) > int(config["num_envs"]) * int(config["rollout_length"]):
        raise ValueError("critic_num_mini_batches cannot exceed rollout samples")
    for key in ("clip_param", "value_clip_param", "gamma", "gae_lambda", "actor_lr", "critic_lr"):
        if not np.isfinite(float(config[key])):
            raise ValueError(f"{key} must be finite")
    if not 0.0 <= float(config["gamma"]) <= 1.0 or not 0.0 <= float(config["gae_lambda"]) <= 1.0:
        raise ValueError("gamma and gae_lambda must be in [0, 1]")
    for key in ("actor_hidden_sizes", "critic_hidden_sizes"):
        sizes = config.get(key)
        if not isinstance(sizes, list) or not sizes or any(not isinstance(size, int) or size <= 0 for size in sizes):
            raise ValueError(f"{key} must contain positive integers")
