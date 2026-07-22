import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "uav_env",
        "uav_env.actions.discrete_15",
        "uav_env.core.geometry",
        "uav_env.dynamics.point_mass_3d",
        "uav_env.dynamics.rk4",
        "uav_env.entities.uav",
        "uav_env.combat.attack_geometry",
        "uav_env.observations.single_observation",
        "uav_env.rewards.single_reward",
        "uav_env.opponents.predictive_rule",
        "uav_env.envs.combat_1v1_env",
        "uav_env.envs.combat_multi_env",
        "uav_env.utils.config",
    ],
)
def test_module_imports(module_name: str) -> None:
    assert importlib.import_module(module_name) is not None
