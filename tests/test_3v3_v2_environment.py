from dataclasses import replace
from math import pi

import numpy as np
import pytest

from uav_env.actions.discrete_15 import DiscreteAction15
from uav_env.algorithms.mappo.adapter import CombatEnvDescription, ParallelCombatVectorEnv, SyncCombatVectorEnv, make_adapter_from_description
from uav_env.algorithms.mappo.config import load_mappo_config
from uav_env.algorithms.mappo.runner import MAPPORunner
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
    assert env.environment_schema_version == "homogeneous_3v3_v2_timeaware"
    assert obs.shape == (3, 63)
    assert info["global_state"].shape == (61,)
    assert len(info["local_observation_feature_names"]) == 63
    assert len(info["global_state_feature_names"]) == 61
    assert info["local_observation_feature_names"] == multi_observation_feature_names_v2()
    assert info["local_observation_feature_names_by_agent"]["red_0"] == multi_observation_feature_names_v2_for_agent("red_0")
    assert "red_1_alive_flag" in info["local_observation_feature_names_by_agent"]["red_0"]
    assert "red_2_alive_flag" in info["local_observation_feature_names_by_agent"]["red_0"]
    assert "blue_0_alive_flag" in info["local_observation_feature_names_by_agent"]["red_0"]
    assert info["global_state_feature_names"] == global_state_feature_names_v2()
    adapter = make_adapter_from_description(CombatEnvDescription("3v3", "head_on_mirrored_jitter_v2", "pursuit", "paper_2024_exact"))
    step = adapter.reset(3)
    assert step.local_obs.shape == (3, 63)
    assert step.global_state.shape == (61,)
    assert info["local_observations_raw"][0, 7] == pytest.approx(0.0)
    assert obs[0, 7] == pytest.approx(-1.0)
    assert info["global_state_raw"][60] == pytest.approx(0.0)
    assert info["global_state"][60] == pytest.approx(-1.0)


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
    left = env._observations().raw[1][8 + 16 + 8]
    env.blue_aircraft[0].state = replace(env.blue_aircraft[0].state, y=own.state.y - 100.0)
    right = env._observations().raw[1][8 + 16 + 8]
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
        prefixes.append((raw[26], raw[39]))
    assert prefixes[0][0] == pytest.approx(2.0)
    assert prefixes[-1][0] == pytest.approx(-2.0)
    assert prefixes[0][1] == pytest.approx(-2.0)
    env.blue_aircraft[0].state = replace(env.blue_aircraft[0].state, alive=False, damaged=True, health=0.0, x=9999.0)
    result = env._observations()
    raw_block = result.raw[0][24:37]
    normalized_block = result.normalized[0][24:37]
    assert raw_block[0] == -1.0
    assert np.allclose(raw_block[1:], 0.0)
    assert np.allclose(normalized_block, [-1.0, *([0.0] * 12)])
    assert not np.any(result.saturated_feature_masks[0][24:37])


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
    state = env._global_state()
    block = state.raw[30:40]
    assert block[0] == -1.0
    assert np.allclose(block[1:9], 0.0)
    assert block[9] == int(DiscreteAction15.LEVEL_HOLD)
    assert np.allclose(state.normalized[30:40], [-1.0, *([0.0] * 9)])
    assert not np.any(state.saturated_feature_mask[30:40])
    assert state.normalized[60] == pytest.approx(-1.0)


def test_v2_episode_progress_values_and_time_markov_state() -> None:
    env = make_v2(40)
    env.decision_step = 0
    start_obs = env._observations()
    start_state = env._global_state()
    env.decision_step = 399
    late_obs = env._observations()
    late_state = env._global_state()
    assert not np.array_equal(start_obs.raw[0], late_obs.raw[0])
    assert not np.array_equal(start_state.raw, late_state.raw)
    assert np.flatnonzero(np.abs(start_obs.raw[0] - late_obs.raw[0]) > 1.0e-12).tolist() == [7]
    assert np.flatnonzero(np.abs(start_state.raw - late_state.raw) > 1.0e-12).tolist() == [60]
    assert start_obs.raw[0, 7] == pytest.approx(0.0)
    assert start_obs.normalized[0, 7] == pytest.approx(-1.0)
    env.decision_step = 200
    assert env._observations().raw[0, 7] == pytest.approx(0.5)
    assert env._observations().normalized[0, 7] == pytest.approx(0.0)
    assert env._global_state().raw[60] == pytest.approx(0.5)
    assert env._global_state().normalized[60] == pytest.approx(0.0)
    env.decision_step = 400
    assert env._observations().raw[0, 7] == pytest.approx(1.0)
    assert env._observations().normalized[0, 7] == pytest.approx(1.0)
    assert env._global_state().raw[60] == pytest.approx(1.0)
    assert env._global_state().normalized[60] == pytest.approx(1.0)


def test_v2_real_400th_step_timeout_terminal_and_vector_autoreset(monkeypatch) -> None:
    def no_attacks(aircraft, attack_config, damage_config, rng, sample_team_order=None):
        return MultiCombatStepResult({u.uav_id: u.state.copy() for u in aircraft}, [], [])

    monkeypatch.setattr("uav_env.envs.combat_multi_env.resolve_multi_attacks", no_attacks)
    monkeypatch.setattr("uav_env.envs.combat_multi_env.CombatMultiEnv._resolve_collisions", lambda self: ([], set()))
    monkeypatch.setattr("uav_env.envs.combat_multi_env.CombatMultiEnv._blue_actions", lambda self, assignments: [DiscreteAction15.LEVEL_HOLD] * 3)
    vector = SyncCombatVectorEnv([lambda: make_adapter_from_description(CombatEnvDescription("3v3", "symmetric_stress_test_v2", "pursuit", "paper_2024_exact"))], 400)
    try:
        reset = vector.reset()
        assert reset["local_obs"][0, :, 7].tolist() == [-1.0, -1.0, -1.0]
        assert reset["global_state"][0, 60] == pytest.approx(-1.0)
        result = None
        for _ in range(399):
            result = vector.step(np.zeros((1, 3), dtype=np.int64))
            assert not result["terminated"][0]
            assert not result["truncated"][0]
        assert result is not None
        assert result["terminal_steps"][0].info["decision_step"] == 399
        assert result["terminal_steps"][0].global_state[60] == pytest.approx(2.0 * 399.0 / 400.0 - 1.0)
        result = vector.step(np.zeros((1, 3), dtype=np.int64))
        terminal = result["terminal_steps"][0]
        reset_step = result["reset_steps"][0]
        assert not terminal.terminated and terminal.truncated
        assert terminal.info["outcome"].termination_reason == "timeout"
        assert terminal.info["decision_step"] == 400
        assert terminal.info["episode_progress"] == pytest.approx(1.0)
        assert terminal.local_obs[:, 7].tolist() == [1.0, 1.0, 1.0]
        assert terminal.global_state[60] == pytest.approx(1.0)
        assert {b.terminal for b in terminal.info["agent_reward_breakdowns"].values()} == {-4.0}
        assert {b.terminal_profile for b in terminal.info["agent_reward_breakdowns"].values()} == {"project_3v3_v2_timeout"}
        assert reset_step is not None
        assert reset_step.info["decision_step"] == 0
        assert reset_step.local_obs[:, 7].tolist() == [-1.0, -1.0, -1.0]
        assert reset_step.global_state[60] == pytest.approx(-1.0)
        assert result["next_global_state"][0, 60] == pytest.approx(-1.0)
        assert terminal.global_state[60] != result["next_global_state"][0, 60]
    finally:
        vector.close()


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
    assert {item.profile for item in terminal.values()} == {"project_3v3_v2_timeout"}
    simultaneous = EpisodeOutcome("draw", False, False, "simultaneous_elimination", 10, 5.0, 0, 0)
    sim_terminal = multi_terminal_reward_allocations(simultaneous, env.red_aircraft, {}, env.config)
    assert {item.reward for item in sim_terminal.values()} == {0.0}
    assert {item.profile for item in sim_terminal.values()} == {"project_3v3_v2_simultaneous_elimination"}
    win = EpisodeOutcome("red", True, False, "blue_eliminated", 10, 5.0, 3, 0)
    win_terminal = multi_terminal_reward_allocations(win, env.red_aircraft, {}, env.config)
    assert all(item.reward > 0.0 for item in win_terminal.values())
    assert {item.profile for item in win_terminal.values()} == {"paper_2024_exact"}


def test_v2_parallel_worker_shapes() -> None:
    vector = ParallelCombatVectorEnv(CombatEnvDescription("3v3", "head_on_mirrored_jitter_v2", "pursuit", "paper_2024_exact"), 4, 31)
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


def test_old_development_v2_runtime_schema_is_rejected() -> None:
    from uav_env.envs.combat_multi_env import CombatMultiEnv
    from uav_env.utils.config import load_yaml, deep_merge, project_root

    root = project_root()
    config = deep_merge(load_yaml(root / "configs" / "base.yaml"), load_yaml(root / "configs" / "paper_2024_homogeneous.yaml"), load_yaml(root / "configs" / "scenario_3v3_v2.yaml"))
    config["environment_schema_version"] = "homogeneous_3v3_v2"
    config["observation_schema"] = "fixed_id_body_62d"
    config["global_state_schema"] = "full_entity_60d"
    config["red_count"] = config["blue_count"] = 3
    with pytest.raises(ValueError, match="development-only 62D/60D schema"):
        CombatMultiEnv(config, "head_on_mirrored_jitter_v2", "pursuit")


def test_v2_rollout_reward_components_include_nonterminal_steps(tmp_path) -> None:
    config = load_mappo_config("configs/mappo_smoke_3v3_v2.yaml")
    config.update(num_envs=1, vector_env="sync", rollout_length=3, total_env_steps=3, device="cpu", run_id="component_rollout")
    runner = MAPPORunner(config, "component_rollout_test", tmp_path)
    try:
        env = runner.vector.envs[0].env
        own = env.red_aircraft[0]
        env.blue_aircraft[0].state = replace(env.blue_aircraft[0].state, x=own.state.x + 500.0, y=own.state.y, z=own.state.z, heading_angle=0.0)
        obs, info = env._build_step_output_info(reset=True)
        runner.current = runner.vector._stack([runner.vector.envs[0]._pack(obs, info, np.zeros(3), False, False, 0.0)])
        _, diagnostics = runner.collect()
    finally:
        runner.close()
    assert diagnostics["rollout_episode_count"] == 0.0
    assert diagnostics["geometry_event_reward_per_episode"] == 0.0
    assert diagnostics["geometry_event_reward_per_agent_episode"] == 0.0
    assert diagnostics["geometry_event_reward_per_decision_step"] != 0.0
    assert np.isfinite(list(diagnostics.values())).all()
    for key in ("rollout_red_attack_attempts_mean", "rollout_blue_attack_attempts_mean", "rollout_red_collisions_mean", "rollout_blue_collisions_mean"):
        assert key in diagnostics
