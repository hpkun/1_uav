"""MAPPO YAML configuration loading and validation."""

from __future__ import annotations
from pathlib import Path
from typing import Any
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
    return config
