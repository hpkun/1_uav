from dataclasses import replace
from math import pi

import numpy as np
import pytest

from uav_env.actions.discrete_15 import DiscreteAction15
from uav_env.algorithms.mappo.adapter import CombatEnvDescription, ParallelCombatVectorEnv, make_adapter_from_description
from uav_env.combat.events import EpisodeOutcome
from uav_env.combat.multi_combat import MultiCombatStepResult, ResolvedAttack
from uav_env.envs import make_3v3_env
from uav_env.observations.global_state import global_state_feature_names_v2
from uav_env.observations.multi_observation import multi_observation_feature_names_v2, multi_observation_feature_names_v2_for_agent
from uav_env.rewards.multi_reward import multi_terminal_reward_allocations


def make_v2(seed: int = 1):
    env = make_3v3_env("head_on_mirrored_jitter_v2", "pursuit", seed=seed, multi_terminal_reward_profile="paper_2024_exact")
    env.reset(seed=seed)
    return env


def test_v2_shapes_feature_names_and_mappo_adapter() -> None:
    env = make_v2(1)
    obs, info = env.reset(seed=1)
    assert env.environment_schema_version == "homogeneous_3v3_v2"
    assert obs.shape == (3, 62)
    assert info["global_state"].shape == (60,)
    assert info["local_observation_feature_names"] == multi_observation_feature_names_v2()
    assert info["local_observation_feature_names_by_agent"]["red_0"] == multi_observation_feature_names_v2_for_agent("red_0")
    assert "red_1_alive_flag" in info["local_observation_feature_names_by_agent"]["red_0"]
    assert "red_2_alive_flag" in info["local_observation_feature_names_by_agent"]["red_0"]
    assert "blue_0_alive_flag" in info["local_observation_feature_names_by_agent"]["red_0"]
    assert info["global_state_feature_names"] == global_state_feature_names_v2()
    adapter = make_adapter_from_description(CombatEnvDescription("3v3", "head_on_mirrored_jitter_v2", "pursuit", "paper_2024_exact"))
    step = adapter.reset(3)
    assert step.local_obs.shape == (3, 62)
    assert step.global_state.shape == (60,)


def test_v2_body_frame_bearing_health_and_heading_are_observable() -> None:
    env = make_v2(2)
    raw = env._observations().raw[0]
    env.red_aircraft[0].state = replace(env.red_aircraft[0].state, heading_angle=pi / 2)
    assert not np.array_equal(raw, env._observations().raw[0])
    env = make_v2(2)
    raw = env._observations().raw[0]
    env.red_aircraft[0].state = replace(env.red_aircraft[0].state, health=1.0)
    assert not np.array_equal(raw, env._observations().raw[0])
    env = make_v2(2)
    raw = env._observations().raw[0]
    env.blue_aircraft[0].state = replace(env.blue_aircraft[0].state, health=1.0)
    assert not np.array_equal(raw, env._observations().raw[0])
    env = make_v2(2)
    own = env.red_aircraft[1]
    env.blue_aircraft[0].state = replace(env.blue_aircraft[0].state, x=own.state.x + 100.0, y=own.state.y + 100.0)
    left = env._observations().raw[1][7 + 16 + 8]
    env.blue_aircraft[0].state = replace(env.blue_aircraft[0].state, y=own.state.y - 100.0)
    right = env._observations().raw[1][7 + 16 + 8]
    assert left > 0.0 and right < 0.0


def test_v2_fixed_id_slots_do_not_swap_and_dead_slots_zero() -> None:
    env = make_3v3_env("symmetric_stress_test_v2", seed=3, multi_terminal_reward_profile="paper_2024_exact")
    env.reset(seed=3)
    own = env.red_aircraft[0]
    prefixes = []
    for offset in (2.0, 0.1, -0.1, -2.0):
        env.blue_aircraft[0].state = replace(env.blue_aircraft[0].state, x=own.state.x + 100.0, y=own.state.y + offset)
        env.blue_aircraft[1].state = replace(env.blue_aircraft[1].state, x=own.state.x + 100.0, y=own.state.y - offset)
        raw = env._observations().raw[0]
        prefixes.append((raw[25], raw[38]))
    assert prefixes[0][0] == pytest.approx(2.0)
    assert prefixes[-1][0] == pytest.approx(-2.0)
    assert prefixes[0][1] == pytest.approx(-2.0)
    env.blue_aircraft[0].state = replace(env.blue_aircraft[0].state, alive=False, damaged=True, health=0.0, x=9999.0)
    block = env._observations().raw[0][23:36]
    assert block[0] == -1.0
    assert np.allclose(block[1:], 0.0)


def test_v2_global_state_markov_distinctions_and_dead_normalization() -> None:
    env = make_v2(4)
    base = env._global_state().raw
    for mutation in (
        lambda e: [setattr(u.state, "z", 1.0) for u in e.all_aircraft],
        lambda e: setattr(e.red_aircraft[0], "state", replace(e.red_aircraft[0].state, speed=150.0)),
        lambda e: setattr(e.red_aircraft[0], "state", replace(e.red_aircraft[0].state, health=1.0)),
        lambda e: setattr(e.blue_aircraft[0], "state", replace(e.blue_aircraft[0].state, health=1.0)),
        lambda e: setattr(e.blue_aircraft[0], "state", replace(e.blue_aircraft[0].state, last_action=int(DiscreteAction15.RIGHT_HOLD))),
        lambda e: setattr(e.blue_aircraft[0], "state", replace(e.blue_aircraft[0].state, heading_angle=1.0)),
        lambda e: setattr(e.blue_aircraft[0], "state", replace(e.blue_aircraft[0].state, flight_path_angle=0.1)),
    ):
        env = make_v2(4)
        mutation(env)
        assert not np.array_equal(base, env._global_state().raw)
    env = make_v2(4)
    env.blue_aircraft[0].state = replace(env.blue_aircraft[0].state, alive=False, damaged=True, health=0.0, x=9999.0)
    block = env._global_state().raw[30:40]
    assert block[0] == -1.0
    assert np.allclose(block[1:9], 0.0)
    assert block[9] == int(DiscreteAction15.LEVEL_HOLD)


def test_v2_jitter_reproducible_and_symmetric_stress_exact() -> None:
    a = make_3v3_env("head_on_mirrored_jitter_v2", seed=5)
    b = make_3v3_env("head_on_mirrored_jitter_v2", seed=5)
    c = make_3v3_env("head_on_mirrored_jitter_v2", seed=6)
    a.reset(seed=5); b.reset(seed=5); c.reset(seed=6)
    assert [u.state.to_kinematic_vector().tolist() for u in a.all_aircraft] == [u.state.to_kinematic_vector().tolist() for u in b.all_aircraft]
    assert [u.state.to_kinematic_vector().tolist() for u in a.all_aircraft] != [u.state.to_kinematic_vector().tolist() for u in c.all_aircraft]
    stress = make_3v3_env("symmetric_stress_test_v2", seed=5)
    stress.reset(seed=5)
    assert [u.state.x for u in stress.red_aircraft] == [-900.0, -900.0, -900.0]
    assert [u.state.x for u in stress.blue_aircraft] == [900.0, 900.0, 900.0]
    assert [u.state.y for u in stress.red_aircraft] == [-500.0, 0.0, 500.0]
    assert [u.state.heading_angle for u in stress.red_aircraft] == [0.0, 0.0, 0.0]


def test_v2_reward_split_damage_penalty_and_timeout_terminal(monkeypatch) -> None:
    env = make_v2(7)
    env.red_aircraft[0].state = replace(env.red_aircraft[0].state, z=float(env.config["min_altitude"]), health=300.0, alive=True, damaged=False, crashed=False)
    _, _, _, _, info = env.step(np.zeros(3, dtype=np.int64))
    breakdown = info["agent_reward_breakdowns"]["red_0"]
    assert breakdown.combat_event == pytest.approx(-0.5)
    assert breakdown.assigned_shape <= -0.03
    assert breakdown.dense_reward == pytest.approx(breakdown.assigned_shape - 0.5)
    assert breakdown.dense_reward <= -0.53

    env = make_v2(8)

    def fake_resolve(aircraft, attack_config, damage_config, rng, sample_team_order=None):
        states = {u.uav_id: u.state.copy() for u in aircraft}
        states["red_0"] = replace(states["red_0"], health=0.0, alive=False, damaged=True, ever_hit=True)
        return MultiCombatStepResult(states, [], [ResolvedAttack("blue_0", "red_0", 100.0, 0.0, 300.0, 300.0, 0.0, True, True)])

    monkeypatch.setattr("uav_env.envs.combat_multi_env.resolve_multi_attacks", fake_resolve)
    _, _, _, _, info = env.step(np.zeros(3, dtype=np.int64))
    breakdown = info["agent_reward_breakdowns"]["red_0"]
    assert breakdown.combat_event == pytest.approx(-2.5)
    assert breakdown.dense_reward <= -2.53

    outcome = EpisodeOutcome("red", True, True, "timeout", 400, 200.0, 3, 2)
    terminal = multi_terminal_reward_allocations(outcome, env.red_aircraft, {}, env.config)
    assert {item.reward for item in terminal.values()} == {-4.0}
    win = EpisodeOutcome("red", True, False, "blue_eliminated", 10, 5.0, 3, 0)
    assert all(item.reward > 0.0 for item in multi_terminal_reward_allocations(win, env.red_aircraft, {}, env.config).values())


def test_v2_parallel_worker_shapes() -> None:
    vector = ParallelCombatVectorEnv(CombatEnvDescription("3v3", "head_on_mirrored_jitter_v2", "pursuit", "paper_2024_exact"), 4, 31)
    try:
        reset = vector.reset()
        assert reset["local_obs"].shape == (4, 3, 62)
        assert reset["global_state"].shape == (4, 60)
        result = vector.step(np.zeros((4, 3), dtype=np.int64))
        assert result["next_local_obs"].shape == (4, 3, 62)
        assert result["next_global_state"].shape == (4, 60)
    finally:
        vector.close()
    assert not any(vector.workers_alive)
