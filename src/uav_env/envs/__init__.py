"""Importable environment class interfaces."""

from __future__ import annotations

from uav_env.envs.base_env import BaseUAVEnv
from uav_env.envs.combat_1v1_env import Combat1v1Env
from uav_env.envs.combat_multi_env import CombatMultiEnv
from uav_env.utils.config import load_experiment_config, load_multi_experiment_config

def make_1v1_env(
    scenario: str = "tail_chase",
    opponent: str = "straight",
    config_name: str = "paper_2024_homogeneous",
    seed: int | None = None,
) -> Combat1v1Env:
    """Load a flat experiment config and construct one homogeneous 1v1 env."""

    config = load_experiment_config(config_name, scenario)
    return Combat1v1Env(config, scenario, opponent, seed)


def make_2v2_env(
    scenario: str = "head_on_formation",
    opponent: str = "straight",
    config_name: str = "paper_2024_homogeneous",
    seed: int | None = None,
) -> CombatMultiEnv:
    """Construct the fixed homogeneous 2v2 experiment environment."""

    config = load_multi_experiment_config(config_name, scenario)
    return CombatMultiEnv(config, scenario, opponent, seed)


__all__ = ["BaseUAVEnv", "Combat1v1Env", "CombatMultiEnv", "make_1v1_env", "make_2v2_env"]
