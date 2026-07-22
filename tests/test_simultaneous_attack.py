from __future__ import annotations

from math import pi

from conftest import make_state
from uav_env.core.enums import Team
from uav_env.envs.combat_1v1_env import Combat1v1Env
from uav_env.entities.type_profiles import profile_from_config


def test_simultaneous_attack_uses_pre_damage_health(experiment_config: dict[str, object]) -> None:
    config = dict(experiment_config)
    config["attack_angle_max"] = pi
    config["escape_angle_max"] = pi
    profile = profile_from_config(config)
    red = make_state(profile, x=0.0, health=20.0)
    blue = make_state(profile, x=500.0, heading=pi, team=Team.BLUE, health=20.0)
    env = Combat1v1Env(config, "tail_chase", "straight", seed=3)
    env.reset(seed=3, options={"red_state": red, "blue_state": blue})
    _, _, terminated, _, info = env.step(0)
    assert info["damage_to_red"].attempted
    assert info["damage_to_blue"].attempted
    assert info["damage_to_red"].health_before == 20.0
    assert info["damage_to_blue"].health_before == 20.0
    assert terminated == (not info["red_state"].alive or not info["blue_state"].alive)
