from __future__ import annotations

from dataclasses import replace
from math import pi

import numpy as np
import pytest

from conftest import make_state
from uav_env.actions.discrete_15 import DiscreteAction15, get_control
from uav_env.algorithms.happo.config import load_happo_config
from uav_env.algorithms.mappo.config import load_mappo_config
from uav_env.combat.attack_geometry import AttackZoneConfig, compute_combat_geometry
from uav_env.combat.multi_combat import assign_nearest_targets_independently, assign_targets
from uav_env.core.enums import Team
from uav_env.dynamics.propagation import propagate_state
from uav_env.envs import make_3v3_env
from uav_env.opponents.greedy_combat import GreedyCombatOpponent
from uav_env.opponents.random import RandomOpponent
from uav_env.opponents.straight import StraightOpponent
from uav_env.utils.config import load_multi_experiment_config


def make_policy(profile, **overrides) -> GreedyCombatOpponent:
    attack = AttackZoneConfig(40.0, 900.0, 0.6, 1.1, 0.8, 40.0, 1300.0, 1.1)
    config = {
        "offense_weight": 1.0,
        "defense_weight": 0.7,
        "angle_score_weight": 0.6,
        "distance_score_weight": 0.4,
        "attack_area_bonus": 0.5,
        "advantage_area_bonus": 0.25,
        "can_attack_bonus": 2.0,
        "incoming_attack_area_penalty": 0.75,
        "incoming_advantage_area_penalty": 0.25,
        "incoming_can_attack_penalty": 2.0,
        "minimum_safe_altitude": 300.0,
        "ceiling_margin": 300.0,
    }
    config.update(overrides.pop("config", {}))
    return GreedyCombatOpponent(profile, attack, config=config, **overrides)


def test_greedy_combat_is_deterministic_and_returns_valid_action(profile) -> None:
    policy = make_policy(profile)
    own = make_state(profile, x=0.0, z=1500.0, heading=0.0, team=Team.BLUE)
    target = replace(make_state(profile, x=600.0, z=1500.0, heading=0.0, team=Team.RED), last_action=int(DiscreteAction15.LEVEL_HOLD))
    first = policy.select_action(own, target, np.random.default_rng(1))
    second = policy.select_action(own, target, np.random.default_rng(999))
    assert first == second
    assert first in DiscreteAction15


def test_target_last_action_changes_prediction(profile) -> None:
    policy = make_policy(profile)
    target = make_state(profile, x=0.0, z=1500.0, heading=0.0, team=Team.RED)
    level = policy._predict_target(replace(target, last_action=int(DiscreteAction15.LEVEL_HOLD)))
    climb = policy._predict_target(replace(target, last_action=int(DiscreteAction15.CLIMB_HOLD)))
    invalid = policy._predict_target(replace(target, last_action=999))
    assert climb.z > level.z
    assert invalid.to_kinematic_vector() == pytest.approx(level.to_kinematic_vector())


def test_candidate_prediction_uses_full_physics_steps(profile) -> None:
    policy = make_policy(profile, physics_steps=7)
    state = make_state(profile, x=0.0, z=1500.0, heading=0.0, team=Team.BLUE)
    predicted = policy._predict_state(state, DiscreteAction15.LEFT_ACCELERATE)
    manual = state.copy()
    for _ in range(7):
        manual = propagate_state(manual, get_control(DiscreteAction15.LEFT_ACCELERATE), profile, 0.1, 9.81)
    assert predicted.to_kinematic_vector() == pytest.approx(manual.to_kinematic_vector())


def test_can_attack_bonus_increases_offensive_score(profile) -> None:
    policy = make_policy(profile)
    attacker = make_state(profile, x=0.0, z=1500.0, heading=0.0, team=Team.BLUE)
    attackable = make_state(profile, x=500.0, z=1500.0, heading=0.0, team=Team.RED)
    far = make_state(profile, x=1500.0, z=1500.0, heading=0.0, team=Team.RED)
    attackable_geometry = compute_combat_geometry(attacker, attackable, policy.attack_config)
    far_geometry = compute_combat_geometry(attacker, far, policy.attack_config)
    assert attackable_geometry.can_attack
    assert not far_geometry.can_attack
    assert policy._offensive_score(attackable_geometry) > policy._offensive_score(far_geometry)


def test_lower_incoming_threat_produces_higher_score(profile) -> None:
    policy = make_policy(profile)
    red = make_state(profile, x=0.0, z=1500.0, heading=0.0, team=Team.RED)
    threatened_blue = make_state(profile, x=500.0, z=1500.0, heading=pi, team=Team.BLUE)
    safer_blue = make_state(profile, x=-500.0, z=1500.0, heading=pi, team=Team.BLUE)
    high_threat = policy._incoming_threat(compute_combat_geometry(red, threatened_blue, policy.attack_config))
    low_threat = policy._incoming_threat(compute_combat_geometry(red, safer_blue, policy.attack_config))
    assert high_threat > low_threat
    same_offense = 1.0
    high_score = policy.greedy_config.offense_weight * same_offense - policy.greedy_config.defense_weight * high_threat
    low_score = policy.greedy_config.offense_weight * same_offense - policy.greedy_config.defense_weight * low_threat
    assert low_score > high_score


def test_unsafe_actions_are_excluded_near_ground_and_ceiling(profile) -> None:
    policy = make_policy(profile)
    target = make_state(profile, x=500.0, z=1500.0, heading=0.0, team=Team.RED)
    near_ground = make_state(profile, x=0.0, z=305.0, heading=0.0, team=Team.BLUE)
    ground_action = policy.select_action(near_ground, target)
    assert ground_action not in {DiscreteAction15.DIVE_HOLD, DiscreteAction15.DIVE_ACCELERATE, DiscreteAction15.DIVE_DECELERATE}
    near_ceiling = make_state(profile, x=0.0, z=4685.0, heading=0.0, team=Team.BLUE)
    ceiling_action = policy.select_action(near_ceiling, target)
    assert ceiling_action not in {DiscreteAction15.CLIMB_HOLD, DiscreteAction15.CLIMB_ACCELERATE, DiscreteAction15.CLIMB_DECELERATE}


def test_greedy_target_assignment_unique_reuses_and_tie_breaks() -> None:
    env = make_3v3_env("head_on_mirrored_jitter_v2", "greedy_combat", seed=2, multi_terminal_reward_profile="paper_2024_exact")
    env.reset(seed=2)
    for index, blue in enumerate(env.blue_aircraft):
        blue.state = replace(blue.state, x=0.0, y=float(index), z=1800.0)
    env.red_aircraft[0].state = replace(env.red_aircraft[0].state, x=10.0, y=0.0, z=1800.0)
    env.red_aircraft[1].state = replace(env.red_aircraft[1].state, x=-10.0, y=0.0, z=1800.0)
    env.red_aircraft[2].state = replace(env.red_aircraft[2].state, x=100.0, y=0.0, z=1800.0)
    greedy = assign_targets(env.blue_aircraft, env.red_aircraft)
    pursuit = assign_nearest_targets_independently(env.blue_aircraft, env.red_aircraft)
    assert [item.target_id for item in greedy] == ["red_0", "red_1", "red_2"]
    assert [item.target_id for item in pursuit] == ["red_0", "red_0", "red_0"]
    env.red_aircraft[1].state = replace(env.red_aircraft[1].state, health=0.0, alive=False, damaged=True)
    living = assign_targets(env.blue_aircraft, env.red_aircraft)
    assert set(item.target_id for item in living) <= {"red_0", "red_2"}
    assert len(living) == 3
    assert len({item.target_id for item in living}) == 2


def test_combat_multi_env_uses_greedy_assignments_but_keeps_pursuit_independent() -> None:
    greedy_env = make_3v3_env("head_on_mirrored_jitter_v2", "greedy_combat", seed=3, multi_terminal_reward_profile="paper_2024_exact")
    pursuit_env = make_3v3_env("head_on_mirrored_jitter_v2", "pursuit", seed=3, multi_terminal_reward_profile="paper_2024_exact")
    for env in (greedy_env, pursuit_env):
        env.reset(seed=3)
        for index, blue in enumerate(env.blue_aircraft):
            blue.state = replace(blue.state, x=0.0, y=float(index), z=1800.0)
        for red in env.red_aircraft:
            red.state = replace(red.state, x=10.0, y=0.0, z=1800.0)
    _, _, _, _, greedy_info = greedy_env.step(np.zeros(3, dtype=np.int64))
    _, _, _, _, pursuit_info = pursuit_env.step(np.zeros(3, dtype=np.int64))
    assert len({item.target_id for item in greedy_info["blue_target_assignments"]}) == 3
    assert len({item.target_id for item in pursuit_info["blue_target_assignments"]}) == 1


def test_straight_and_random_remain_unchanged(profile) -> None:
    own = make_state(profile, team=Team.BLUE)
    target = make_state(profile, x=1000.0, team=Team.RED)
    assert StraightOpponent().select_action(own, target) is DiscreteAction15.LEVEL_HOLD
    assert RandomOpponent().select_action(own, target, np.random.default_rng(4)) == RandomOpponent().select_action(own, target, np.random.default_rng(4))


def test_formal_configs_use_greedy_combat_and_learnability_stays_straight() -> None:
    assert load_mappo_config("configs/mappo_3v3_v2.yaml")["environment"]["opponent"] == "greedy_combat"
    assert load_happo_config("configs/happo_3v3_v2.yaml")["environment"]["opponent"] == "greedy_combat"
    assert load_mappo_config("configs/mappo_learnability_3v3.yaml")["environment"]["opponent"] == "straight"
    assert load_happo_config("configs/happo_learnability_3v3.yaml")["environment"]["opponent"] == "straight"


def test_greedy_config_validation_rejects_invalid_safety() -> None:
    config = load_multi_experiment_config("paper_2024_homogeneous", "head_on_mirrored_jitter_v2", team_size=3)
    config["greedy_combat"] = {**config["greedy_combat"], "ceiling_margin": float(config["max_altitude"])}
    with pytest.raises(ValueError, match="minimum safe altitude"):
        from uav_env.utils.config import validate_experiment_config

        validate_experiment_config(config)


def test_short_greedy_3v3_integration_has_finite_states_and_actions() -> None:
    env = make_3v3_env("head_on_mirrored_jitter_v2", "greedy_combat", seed=12, multi_terminal_reward_profile="paper_2024_exact")
    obs, info = env.reset(seed=12)
    assert np.isfinite(obs).all()
    assert np.isfinite(info["global_state"]).all()
    for _ in range(10):
        obs, _, terminated, truncated, info = env.step(np.zeros(3, dtype=np.int64))
        assert np.isfinite(obs).all()
        assert np.isfinite(info["global_state"]).all()
        assert all(0 <= int(action) < 15 for action in info["blue_actions"])
        if terminated or truncated:
            break
