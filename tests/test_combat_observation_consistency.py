from __future__ import annotations

from dataclasses import replace
from math import pi

import numpy as np
import pytest

from uav_env.core.enums import Team
from uav_env.envs import make_3v3_env


def _feature_index(env, agent_id: str, feature: str) -> int:
    result = env._observations()
    return result.feature_names_by_agent[agent_id].index(feature)


def test_body_frame_lateral_and_longitudinal_signs_are_consistent() -> None:
    env = make_3v3_env("head_on_mirrored_jitter_v2", "greedy_combat", seed=1, multi_terminal_reward_profile="paper_2024_exact")
    env.reset(seed=1)
    own = env.red_aircraft[0]
    own.state = replace(own.state, x=0.0, y=0.0, z=1500.0, heading_angle=0.0)
    env.blue_aircraft[0].state = replace(env.blue_aircraft[0].state, x=500.0, y=100.0, z=1500.0)
    right = env._observations().raw[0, _feature_index(env, "red_0", "blue_0_body_relative_y")]
    env.blue_aircraft[0].state = replace(env.blue_aircraft[0].state, x=500.0, y=-100.0, z=1500.0)
    left = env._observations().raw[0, _feature_index(env, "red_0", "blue_0_body_relative_y")]
    assert right == pytest.approx(-left)

    env.blue_aircraft[0].state = replace(env.blue_aircraft[0].state, x=-500.0, y=0.0, z=1500.0)
    behind = env._observations().raw[0, _feature_index(env, "red_0", "blue_0_body_relative_x")]
    env.blue_aircraft[0].state = replace(env.blue_aircraft[0].state, x=500.0, y=0.0, z=1500.0)
    ahead = env._observations().raw[0, _feature_index(env, "red_0", "blue_0_body_relative_x")]
    assert ahead == pytest.approx(-behind)


def test_body_frame_rotation_preserves_ahead_relation() -> None:
    env = make_3v3_env("head_on_mirrored_jitter_v2", "greedy_combat", seed=2, multi_terminal_reward_profile="paper_2024_exact")
    env.reset(seed=2)
    env.red_aircraft[0].state = replace(env.red_aircraft[0].state, x=0.0, y=0.0, z=1500.0, heading_angle=0.0)
    env.blue_aircraft[0].state = replace(env.blue_aircraft[0].state, x=500.0, y=0.0, z=1500.0)
    x_index = _feature_index(env, "red_0", "blue_0_body_relative_x")
    y_index = _feature_index(env, "red_0", "blue_0_body_relative_y")
    baseline = env._observations().raw[0, [x_index, y_index]]
    env.red_aircraft[0].state = replace(env.red_aircraft[0].state, x=0.0, y=0.0, z=1500.0, heading_angle=pi / 2.0)
    env.blue_aircraft[0].state = replace(env.blue_aircraft[0].state, x=0.0, y=500.0, z=1500.0)
    rotated = env._observations().raw[0, [x_index, y_index]]
    assert rotated == pytest.approx(baseline)


def test_attack_geometry_features_improve_monotonically_in_observation() -> None:
    env = make_3v3_env("head_on_mirrored_jitter_v2", "greedy_combat", seed=3, multi_terminal_reward_profile="paper_2024_exact")
    env.reset(seed=3)
    env.red_aircraft[0].state = replace(env.red_aircraft[0].state, x=0.0, y=0.0, z=1500.0, heading_angle=0.0)
    env.blue_aircraft[0].state = replace(env.blue_aircraft[0].state, x=900.0, y=0.0, z=1500.0, heading_angle=0.0)
    distance_idx = _feature_index(env, "red_0", "blue_0_distance")
    attack_idx = _feature_index(env, "red_0", "blue_0_attack_angle")
    good = env._observations().raw[0, [distance_idx, attack_idx]]
    env.red_aircraft[0].state = replace(env.red_aircraft[0].state, heading_angle=pi / 2.0)
    env.blue_aircraft[0].state = replace(env.blue_aircraft[0].state, x=1500.0, y=0.0, z=1500.0, heading_angle=0.0)
    poor = env._observations().raw[0, [distance_idx, attack_idx]]
    assert good[0] < poor[0]
    assert good[1] < poor[1]


def test_fixed_id_slots_and_dead_masks_are_stable() -> None:
    env = make_3v3_env("head_on_mirrored_jitter_v2", "greedy_combat", seed=4, multi_terminal_reward_profile="paper_2024_exact")
    env.reset(seed=4)
    names = env._observations().feature_names_by_agent
    assert names["red_0"][8:16] == [f"red_1_{name}" for name in ("alive_flag", "body_relative_x", "body_relative_y", "relative_z", "body_relative_vx", "body_relative_vy", "relative_vz", "health_ratio")]
    assert names["red_0"][16:24] == [f"red_2_{name}" for name in ("alive_flag", "body_relative_x", "body_relative_y", "relative_z", "body_relative_vx", "body_relative_vy", "relative_vz", "health_ratio")]
    env.red_aircraft[1].state = replace(env.red_aircraft[1].state, health=0.0, alive=False, damaged=True)
    obs = env._observations()
    assert obs.ally_alive_masks[0].tolist() == [0, 1]
    assert obs.normalized[0, 8] == pytest.approx(-1.0)
    assert np.allclose(obs.normalized[0, 9:16], 0.0)
    masks = env.get_agent_masks()
    assert masks["agent_alive_mask"].tolist() == [1, 0, 1]
    assert masks["available_action_mask"][1].sum() == 1
    assert masks["available_action_mask"][1, 0]


def test_global_state_contains_blue_entities_and_time_feature() -> None:
    env = make_3v3_env("head_on_mirrored_jitter_v2", "greedy_combat", seed=5, multi_terminal_reward_profile="paper_2024_exact")
    _, info = env.reset(seed=5)
    names = info["global_state_feature_names"]
    assert len(names) == 61
    assert names[0] == "red_0_alive_flag"
    assert names[30] == "blue_0_alive_flag"
    assert names[-1] == "episode_progress"
    assert info["global_state"][60] == pytest.approx(-1.0)
