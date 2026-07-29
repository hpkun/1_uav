from dataclasses import replace
from math import pi

import numpy as np
import pytest

from uav_env.actions.discrete_15 import DiscreteAction15, get_control
from uav_env.algorithms.mappo.adapter import CombatEnvDescription, ParallelCombatVectorEnv
from uav_env.algorithms.mappo.config import load_mappo_config
from uav_env.combat.multi_combat import MultiCombatStepResult
from uav_env.core.geometry import normalize_angle
from uav_env.dynamics.propagation import propagate_state
from uav_env.envs import make_3v3_env
from uav_env.opponents.pursuit import PursuitOpponent
from uav_env.opponents.team_controller import TeamRuleController
from uav_env.utils.config import load_multi_experiment_config


FORMAL_SCENARIO = "head_on_mirrored_jitter_v2"
LEARNABILITY_SCENARIO = "head_on_learnability_v1"
REACHABILITY_SEED = 1


def env_states(env):
    return np.asarray([u.state.to_kinematic_vector() for u in env.all_aircraft], dtype=np.float64)


def side_total(info, team: str, field: str) -> float:
    aircraft = info["statistics"]["aircraft"]
    return float(sum(aircraft[f"{team}_{index}"][field] for index in range(3)))


def test_learnability_config_is_isolated_from_formal_v2() -> None:
    formal = load_multi_experiment_config("paper_2024_homogeneous", FORMAL_SCENARIO, team_size=3)
    learn = load_multi_experiment_config("paper_2024_homogeneous", LEARNABILITY_SCENARIO, team_size=3)
    same_fields = [
        "physics_dt",
        "decision_dt",
        "physics_steps_per_action",
        "max_decision_steps",
        "max_episode_seconds",
        "gravity",
        "min_speed",
        "max_speed",
        "min_altitude",
        "max_altitude",
        "min_tangential_overload",
        "max_tangential_overload",
        "min_normal_overload",
        "max_normal_overload",
        "attack_distance_min",
        "attack_distance_max",
        "attack_angle_max",
        "escape_angle_max",
        "attack_area_angle_max",
        "damage_values",
        "damage_probability_thresholds",
        "initial_health",
        "environment_schema_version",
        "observation_schema",
        "global_state_schema",
        "reward_profile",
        "r_den0",
        "r_win0",
        "r_lose0",
        "timeout_reward",
        "multi_terminal_reward_profile",
        "project_assumptions",
    ]
    for field in same_fields:
        assert learn[field] == formal[field], field
    assert learn["scenario_name"] == LEARNABILITY_SCENARIO
    assert learn["scenario_profile"] == LEARNABILITY_SCENARIO
    assert formal["scenario_name"] == FORMAL_SCENARIO
    assert formal["scenario_profile"] == FORMAL_SCENARIO
    assert learn["initial_team_distance"] == pytest.approx(1200.0)
    assert learn["formation_lateral_spacing"] == pytest.approx(250.0)
    assert formal["initial_team_distance"] == pytest.approx(1800.0)
    assert formal["formation_lateral_spacing"] == pytest.approx(500.0)
    for field in ("longitudinal_jitter", "lateral_jitter", "altitude_jitter", "speed_jitter", "heading_jitter"):
        assert learn[field] == pytest.approx(0.25 * formal[field])
    base_mappo = load_mappo_config("configs/mappo_smoke_1v1.yaml")
    formal_mappo = load_mappo_config("configs/mappo_3v3_v2.yaml")
    learn_mappo = load_mappo_config("configs/mappo_learnability_3v3.yaml")
    assert base_mappo["run_symmetric_stress_test"] is False
    assert formal_mappo["environment"]["scenario"] == FORMAL_SCENARIO
    assert formal_mappo["environment"]["opponent"] == "greedy_combat"
    assert formal_mappo["total_env_steps"] == 300000
    assert formal_mappo["run_symmetric_stress_test"] is True
    assert learn_mappo["environment"]["scenario"] == LEARNABILITY_SCENARIO
    assert learn_mappo["environment"]["opponent"] == "straight"
    assert learn_mappo["seed"] == 1
    assert learn_mappo["total_env_steps"] == 50000
    assert learn_mappo["evaluation_interval"] == 10000
    assert learn_mappo["validation_episodes"] == 10
    assert learn_mappo["test_episodes"] == 10
    assert learn_mappo["run_symmetric_stress_test"] is False


def test_learnability_reset_determinism_geometry_shapes_and_straight_blue() -> None:
    a = make_3v3_env(LEARNABILITY_SCENARIO, "straight", seed=5, multi_terminal_reward_profile="paper_2024_exact")
    b = make_3v3_env(LEARNABILITY_SCENARIO, "straight", seed=5, multi_terminal_reward_profile="paper_2024_exact")
    c = make_3v3_env(LEARNABILITY_SCENARIO, "straight", seed=6, multi_terminal_reward_profile="paper_2024_exact")
    obs, info = a.reset(seed=5)
    b.reset(seed=5)
    c.reset(seed=6)
    assert np.array_equal(env_states(a), env_states(b))
    assert not np.array_equal(env_states(a), env_states(c))
    assert obs.shape == (3, 63)
    assert info["global_state"].shape == (61,)
    assert obs[:, 7].tolist() == [-1.0, -1.0, -1.0]
    assert info["global_state"][60] == pytest.approx(-1.0)
    assert np.isfinite(obs).all()
    assert np.isfinite(info["global_state"]).all()
    bases_y = [-250.0, 0.0, 250.0]
    for index, base_y in enumerate(bases_y):
        red = a.red_aircraft[index].state
        blue = a.blue_aircraft[index].state
        assert (red.x + blue.x) / 2.0 == pytest.approx(0.0)
        assert (red.x - blue.x) / 2.0 == pytest.approx(-600.0, abs=12.5)
        assert (red.y + blue.y) / 2.0 == pytest.approx(base_y)
        assert red.z + blue.z == pytest.approx(2.0 * a.config["initial_altitude"])
        assert red.speed == pytest.approx(blue.speed)
        assert abs(normalize_angle(red.heading_angle)) <= a.config["heading_jitter"]
        assert abs(normalize_angle(blue.heading_angle - pi)) <= a.config["heading_jitter"]
    selected = a._blue_actions([])
    assert selected == [DiscreteAction15.LEVEL_HOLD] * 3


def test_learnability_timeout_progress_parallel_workers_and_finite(monkeypatch) -> None:
    def no_attacks(aircraft, attack_config, damage_config, rng, sample_team_order=None):
        return MultiCombatStepResult({u.uav_id: u.state.copy() for u in aircraft}, [], [])

    monkeypatch.setattr("uav_env.envs.combat_multi_env.resolve_multi_attacks", no_attacks)
    monkeypatch.setattr("uav_env.envs.combat_multi_env.CombatMultiEnv._resolve_collisions", lambda self: ([], set()))
    env = make_3v3_env(LEARNABILITY_SCENARIO, "straight", seed=9, multi_terminal_reward_profile="paper_2024_exact")
    env.reset(seed=9)
    info = {}
    for _ in range(400):
        obs, _, terminated, truncated, info = env.step(np.zeros(3, dtype=np.int64))
        assert np.isfinite(obs).all()
        if terminated or truncated:
            break
    assert info["outcome"].termination_reason == "timeout"
    assert info["decision_step"] == 400
    assert info["local_observations"][:, 7].tolist() == [1.0, 1.0, 1.0]
    assert info["global_state"][60] == pytest.approx(1.0)

    vector = ParallelCombatVectorEnv(CombatEnvDescription("3v3", LEARNABILITY_SCENARIO, "straight", "paper_2024_exact"), 4, 20)
    try:
        reset = vector.reset()
        assert reset["local_obs"].shape == (4, 3, 63)
        assert reset["global_state"].shape == (4, 61)
        result = vector.step(np.zeros((4, 3), dtype=np.int64))
        assert result["next_local_obs"].shape == (4, 3, 63)
        assert result["next_global_state"].shape == (4, 61)
    finally:
        vector.close()
    assert not any(vector.workers_alive)


def test_learnability_rule_pursuit_reaches_combat_on_fixed_seed() -> None:
    env = make_3v3_env(LEARNABILITY_SCENARIO, "straight", seed=REACHABILITY_SEED, multi_terminal_reward_profile="paper_2024_exact")
    env.reset(seed=REACHABILITY_SEED)
    pursuit = {key: float(value) for key, value in env.config["pursuit"].items()}
    policy = PursuitOpponent(
        env.profile,
        env.attack_config,
        float(env.config["physics_dt"]),
        int(env.config["physics_steps_per_action"]),
        float(env.config["gravity"]),
        float(env.config["max_altitude"]),
        **pursuit,
    )
    controller = TeamRuleController("pursuit", policy, REACHABILITY_SEED + 1_000_003)
    info = {}
    terminated = truncated = False
    while not (terminated or truncated):
        actions, _ = controller.select_actions(env.red_aircraft, env.blue_aircraft)
        obs, _, terminated, truncated, info = env.step(np.asarray([int(action) for action in actions], dtype=np.int64))
        assert np.isfinite(obs).all()
    assert info["outcome"].decision_steps <= 400
    assert side_total(info, "red", "attack_attempts") > 0.0
    assert side_total(info, "red", "hits") > 0.0
    assert side_total(info, "red", "effective_damage") > 0.0


def test_learnability_basic_action_controllability_changes_actor_observation() -> None:
    env = make_3v3_env(LEARNABILITY_SCENARIO, "straight", seed=30, multi_terminal_reward_profile="paper_2024_exact")
    env.reset(seed=30)
    initial = replace(env.red_aircraft[1].state, flight_path_angle=0.0, heading_angle=0.0)

    def after(action: DiscreteAction15):
        state = initial.copy()
        for _ in range(5):
            state = propagate_state(state, get_control(action), env.profile, float(env.config["physics_dt"]), float(env.config["gravity"]))
        assert np.isfinite(state.to_kinematic_vector()).all()
        return state

    hold = after(DiscreteAction15.LEVEL_HOLD)
    left = after(DiscreteAction15.LEFT_HOLD)
    right = after(DiscreteAction15.RIGHT_HOLD)
    accel = after(DiscreteAction15.LEVEL_ACCELERATE)
    decel = after(DiscreteAction15.LEVEL_DECELERATE)
    climb = after(DiscreteAction15.CLIMB_HOLD)
    dive = after(DiscreteAction15.DIVE_HOLD)
    left_delta = normalize_angle(left.heading_angle - hold.heading_angle)
    right_delta = normalize_angle(right.heading_angle - hold.heading_angle)
    assert left_delta > 0.0
    assert right_delta < 0.0
    assert left_delta == pytest.approx(-right_delta, rel=1.0e-6)
    assert accel.speed > hold.speed
    assert decel.speed < hold.speed
    assert climb.z > hold.z or climb.flight_path_angle > hold.flight_path_angle
    assert dive.z < hold.z or dive.flight_path_angle < hold.flight_path_angle
    base_obs = env._observations().raw[1].copy()
    env.red_aircraft[1].state = left
    assert not np.array_equal(base_obs, env._observations().raw[1])
    assert np.isfinite(env._observations().raw).all()
