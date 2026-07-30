from __future__ import annotations

from dataclasses import replace
from math import pi

import numpy as np
import pytest

from conftest import make_state
from uav_env.actions.discrete_15 import DiscreteAction15, get_control
from uav_env.algorithms.mappo.adapter import CombatEnvDescription, ParallelCombatVectorEnv
from uav_env.algorithms.mappo.metrics import combat_outcome_rates, evaluation_key
from uav_env.combat.events import EpisodeOutcome
from uav_env.combat.multi_combat import ResolvedAttack
from uav_env.core.enums import CombatEventType, Team
from uav_env.dynamics.propagation import propagate_state
from uav_env.envs import make_3v3_env
from uav_env.opponents.greedy_combat import GreedyCombatOpponent
from uav_env.opponents.team_controller import TeamRuleController


def _side_total(info: dict, team: str, field: str) -> float:
    return float(sum(info["statistics"]["aircraft"][f"{team}_{index}"][field] for index in range(3)))


def test_environment_attack_chain_updates_stats_events_health_and_rewards() -> None:
    env = make_3v3_env("head_on_mirrored_jitter_v2", "straight", seed=7, multi_terminal_reward_profile="paper_2024_exact")
    env.reset(seed=7)
    for index in range(3):
        env.red_aircraft[index].state = replace(env.red_aircraft[index].state, x=0.0, y=float(index * 100), z=1500.0, heading_angle=0.0)
        env.blue_aircraft[index].state = replace(env.blue_aircraft[index].state, x=500.0, y=float(index * 100), z=1500.0, heading_angle=0.0)
    _, _, _, _, info = env.step(np.zeros(3, dtype=np.int64))
    assert _side_total(info, "red", "attack_attempts") > 0.0
    assert _side_total(info, "red", "hits") >= 0.0
    assert _side_total(info, "red", "effective_damage") >= 0.0
    assert all(attack.attacker_id.startswith(("red_", "blue_")) for attack in info["attack_attempts"])
    assert all(0.0 <= attack.random_value < 1.0 for attack in info["attack_attempts"])
    assert all(info["statistics"]["aircraft"][f"blue_{index}"]["effective_damage"] <= info["statistics"]["aircraft"][f"blue_{index}"]["nominal_damage"] for index in range(3))
    assert any(event.event_type in {CombatEventType.HIT, CombatEventType.MISS} for event in info["events"])
    if any(attack.attacker_id == "red_0" and attack.hit for attack in info["resolved_attacks"]):
        assert info["agent_reward_breakdowns"]["red_0"].hit_event_reward == pytest.approx(0.8)


def test_attack_target_assignment_is_maneuver_guidance_not_weapon_constraint() -> None:
    env = make_3v3_env("head_on_mirrored_jitter_v2", "greedy_combat", seed=8, multi_terminal_reward_profile="paper_2024_exact")
    env.reset(seed=8)
    env.blue_aircraft[0].state = replace(env.blue_aircraft[0].state, x=0.0, y=0.0, z=1500.0, heading_angle=0.0)
    env.blue_aircraft[1].state = replace(env.blue_aircraft[1].state, x=10.0, y=0.0, z=1500.0, heading_angle=0.0)
    env.blue_aircraft[2].state = replace(env.blue_aircraft[2].state, x=2000.0, y=0.0, z=1500.0, heading_angle=pi)
    env.red_aircraft[0].state = replace(env.red_aircraft[0].state, x=100.0, y=0.0, z=1500.0, heading_angle=0.0)
    env.red_aircraft[1].state = replace(env.red_aircraft[1].state, x=500.0, y=0.0, z=1500.0, heading_angle=0.0)
    env.red_aircraft[2].state = replace(env.red_aircraft[2].state, x=2400.0, y=0.0, z=1500.0, heading_angle=pi)
    _, _, _, _, info = env.step(np.zeros(3, dtype=np.int64))
    assignment = {item.attacker_id: item.target_id for item in info["blue_target_assignments"]}
    blue_1_targets = {attempt.target_id for attempt in info["attack_attempts"] if attempt.attacker_id == "blue_1"}
    assert assignment["blue_1"] == "red_1"
    assert "red_0" in blue_1_targets


def test_combat_event_reward_attribution_is_per_red_agent() -> None:
    env = make_3v3_env("head_on_mirrored_jitter_v2", "straight", seed=9, multi_terminal_reward_profile="paper_2024_exact")
    env.reset(seed=9)
    resolved = [
        ResolvedAttack("red_0", "blue_0", 100.0, 0.0, 51.0, 30.0, 21.0, True, True),
        ResolvedAttack("blue_0", "red_1", 100.0, 0.0, 51.0, 20.0, 31.0, True, False),
        ResolvedAttack("blue_1", "red_1", 100.0, 0.0, 51.0, 10.0, 41.0, True, True),
    ]
    red0_event, red0_contribution, red0_components = env._combat_event_reward(env.red_aircraft[0], resolved, {}, set())
    red1_event, _, red1_components = env._combat_event_reward(env.red_aircraft[1], resolved, {}, set())
    assert red0_event == pytest.approx(2.3)
    assert red0_contribution == pytest.approx(7.0)
    assert red0_components["hit"] == pytest.approx(0.8)
    assert red0_components["destroy"] == pytest.approx(1.5)
    assert red1_event == pytest.approx(-3.4)
    assert red1_components["attacked"] == pytest.approx(-1.8)
    assert red1_components["destroyed"] == pytest.approx(-1.6)


def test_timeout_survivor_win_is_not_elimination_win() -> None:
    outcomes = [
        EpisodeOutcome("red", True, True, "timeout", 400, 200.0, 2, 1),
        EpisodeOutcome("red", True, False, "blue_eliminated", 100, 50.0, 1, 0),
        EpisodeOutcome("draw", True, True, "timeout", 400, 200.0, 2, 2),
    ]
    rates = combat_outcome_rates(outcomes)
    assert rates["overall_red_win_rate"] == pytest.approx(2 / 3)
    assert rates["timeout_survival_win_rate"] == pytest.approx(1 / 3)
    assert rates["elimination_win_rate"] == pytest.approx(1 / 3)
    assert evaluation_key({"overall_red_win_rate": 1.0, "red_crash_rate": 0.0, "blue_crash_rate": 0.0, "mean_episode_return": 0.0}, "smoke")


def test_reset_cleans_health_alive_damage_and_last_action_state() -> None:
    env = make_3v3_env("head_on_mirrored_jitter_v2", "greedy_combat", seed=10, multi_terminal_reward_profile="paper_2024_exact")
    env.reset(seed=10)
    env.red_aircraft[0].state = replace(env.red_aircraft[0].state, health=0.0, alive=False, damaged=True, last_action=7)
    _, info = env.reset(seed=10)
    assert all(aircraft.is_alive for aircraft in env.all_aircraft)
    assert all(aircraft.state.health == pytest.approx(env.config["initial_health"]) for aircraft in env.all_aircraft)
    assert all(not aircraft.state.damaged for aircraft in env.all_aircraft)
    assert all(aircraft.state.last_action == int(DiscreteAction15.LEVEL_HOLD) for aircraft in env.all_aircraft)
    assert info["red_agent_alive_mask"].tolist() == [1, 1, 1]


def test_action_controls_have_expected_directional_effects(profile) -> None:
    env = make_3v3_env("head_on_mirrored_jitter_v2", "straight", seed=11, multi_terminal_reward_profile="paper_2024_exact")
    env.reset(seed=11)
    initial = replace(env.red_aircraft[0].state, x=0.0, y=0.0, z=1500.0, speed=100.0, flight_path_angle=0.0, heading_angle=0.0)

    def after(action: DiscreteAction15):
        current = initial.copy()
        for _ in range(10):
            current = propagate_state(current, get_control(action), env.profile, float(env.config["physics_dt"]), float(env.config["gravity"]))
        return current

    hold = after(DiscreteAction15.LEVEL_HOLD)
    left = after(DiscreteAction15.LEFT_HOLD)
    right = after(DiscreteAction15.RIGHT_HOLD)
    climb = after(DiscreteAction15.CLIMB_HOLD)
    dive = after(DiscreteAction15.DIVE_HOLD)
    accel = after(DiscreteAction15.LEVEL_ACCELERATE)
    decel = after(DiscreteAction15.LEVEL_DECELERATE)
    assert abs(hold.flight_path_angle) < 1.0e-9
    assert left.heading_angle > hold.heading_angle
    assert right.heading_angle > pi
    assert climb.z > hold.z
    assert dive.z < hold.z
    assert accel.speed > hold.speed
    assert decel.speed < hold.speed


def test_short_red_greedy_probe_can_generate_attacks_in_reachable_micro_scenario_without_training() -> None:
    env = make_3v3_env("head_on_mirrored_jitter_v2", "straight", seed=12, multi_terminal_reward_profile="paper_2024_exact")
    env.reset(seed=12)
    for index in range(3):
        y = float((index - 1) * 200.0)
        env.red_aircraft[index].state = replace(env.red_aircraft[index].state, x=0.0, y=y, z=1500.0, heading_angle=0.0)
        env.blue_aircraft[index].state = replace(env.blue_aircraft[index].state, x=500.0, y=y, z=1500.0, heading_angle=0.0)
    policy = GreedyCombatOpponent(
        env.profile,
        env.attack_config,
        float(env.config["physics_dt"]),
        int(env.config["physics_steps_per_action"]),
        float(env.config["gravity"]),
        float(env.config["min_altitude"]),
        float(env.config["max_altitude"]),
        env.config["greedy_combat"],
    )
    controller = TeamRuleController("greedy_combat", policy, 12)
    info = None
    for _ in range(3):
        actions, _ = controller.select_actions(env.red_aircraft, env.blue_aircraft)
        _, _, terminated, truncated, info = env.step(np.asarray([int(action) for action in actions], dtype=np.int64))
        if terminated or truncated:
            break
    assert info is not None
    assert _side_total(info, "red", "attack_attempts") > 0.0


def test_parallel_worker_seeds_are_distinct_and_finite() -> None:
    description = CombatEnvDescription("3v3", "head_on_mirrored_jitter_v2", "greedy_combat", "paper_2024_exact")
    vector = ParallelCombatVectorEnv(description, 2, 123)
    try:
        reset = vector.reset()
        assert reset["local_obs"].shape == (2, 3, 63)
        assert np.isfinite(reset["local_obs"]).all()
        assert not np.array_equal(reset["global_state"][0], reset["global_state"][1])
        result = vector.step(np.zeros((2, 3), dtype=np.int64))
        assert result["next_local_obs"].shape == (2, 3, 63)
        assert np.isfinite(result["next_global_state"]).all()
    finally:
        vector.close()
    assert not any(vector.workers_alive)
