from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from uav_env.algorithms.common.reward_diagnostics import allows_truncation_bootstrap, restore_reward_component_accumulators
from uav_env.algorithms.happo.runner import HAPPORunner, REWARD_COMPONENT_NAMES as HAPPO_COMPONENTS
from uav_env.algorithms.mappo.adapter import CombatEnvDescription, ParallelCombatVectorEnv
from uav_env.algorithms.mappo.rollout_buffer import RolloutBuffer
from uav_env.algorithms.mappo.runner import MAPPORunner, REWARD_COMPONENT_NAMES as MAPPO_COMPONENTS
from uav_env.combat.events import EpisodeOutcome
from uav_env.combat.multi_combat import AttackAttempt, MultiCombatStepResult, ResolvedAttack
from uav_env.core.enums import Team
from uav_env.envs import make_3v3_env
from uav_env.rewards.multi_reward import MultiAgentRewardBreakdown


def _functional_env(mode: str = "heterogeneous_relay"):
    roles = ["combat", "combat", "combat"] if mode == "homogeneous_control" else ["combat", "combat", "support"]
    return make_3v3_env(
        "head_on_functional_heterogeneous_v1",
        "greedy_combat",
        seed=33,
        multi_terminal_reward_profile="paper_2024_exact",
        functional_mode=mode,
        red_roles=roles,
        relay_enabled=(mode == "heterogeneous_relay"),
    )


def _freeze_physics(env) -> None:
    env._propagate_all = lambda action_map: ([], {}, 0)


def _states(aircraft) -> dict[str, object]:
    return {u.uav_id: u.state.copy() for u in aircraft}


def _closure(breakdown: MultiAgentRewardBreakdown) -> None:
    assert breakdown.total == pytest.approx(
        breakdown.assigned_shape
        + breakdown.combat_event
        + breakdown.terminal_base_reward
        + breakdown.mission_success_bonus
    )


def test_functional_timeout_no_bootstrap_and_buffer_mask() -> None:
    step = SimpleNamespace(
        truncated=True,
        info={"outcome": EpisodeOutcome("draw", True, True, "timeout", 1, 0.5, 3, 3)},
    )
    runner_like = SimpleNamespace(schema_metadata={"environment_schema_version": "functional_heterogeneous_3v3_v1"})
    assert MAPPORunner._allows_truncation_bootstrap(runner_like, step) is False

    mask = np.asarray([float(MAPPORunner._allows_truncation_bootstrap(runner_like, step))], dtype=np.float32)
    buffer = RolloutBuffer(1, 1, 3, 69, 64)
    buffer.set_initial(np.zeros((1, 3, 69), np.float32), np.zeros((1, 64), np.float32), np.ones((1, 3, 15), bool))
    buffer.insert(
        np.zeros((1, 3), np.int64),
        np.zeros((1, 3), np.float32),
        np.ones((1, 3), np.float32),
        np.zeros((1, 3), np.float32),
        np.asarray([False]),
        np.asarray([True]),
        np.ones((1, 3), np.float32),
        np.ones((1, 3), np.float32),
        np.zeros((1, 3, 69), np.float32),
        np.zeros((1, 64), np.float32),
        np.ones((1, 3, 15), bool),
        np.full((1, 3), 99.0, np.float32),
        mask,
    )
    buffer.finish(np.full((1, 3), 77.0, np.float32), 0.99, 1.0)
    assert np.all(buffer.truncation_bootstrap_masks == 0.0)
    assert np.all(buffer.next_values == 0.0)


def test_terminal_timeout_schemas_and_legacy_bootstrap_rules() -> None:
    timeout = SimpleNamespace(truncated=True, info={"outcome": EpisodeOutcome("red", True, True, "timeout", 1, 0.5, 3, 2)})
    bad = SimpleNamespace(truncated=True, info={"outcome": EpisodeOutcome(None, True, True, "external_interrupt", 1, 0.5, 3, 3)})
    v2 = SimpleNamespace(schema_metadata={"environment_schema_version": "homogeneous_3v3_v2_timeaware"})
    functional = SimpleNamespace(schema_metadata={"environment_schema_version": "functional_heterogeneous_3v3_v1"})
    legacy = SimpleNamespace(schema_metadata={"environment_schema_version": "legacy"})
    assert MAPPORunner._allows_truncation_bootstrap(v2, timeout) is False
    assert MAPPORunner._allows_truncation_bootstrap(functional, timeout) is False
    assert MAPPORunner._allows_truncation_bootstrap(legacy, timeout) is True
    with pytest.raises(RuntimeError, match="truncated step must be timeout"):
        MAPPORunner._allows_truncation_bootstrap(functional, bad)
    assert allows_truncation_bootstrap("legacy", True, "external_interrupt") is True


def test_support_dead_before_step_gets_no_later_team_event(monkeypatch: pytest.MonkeyPatch) -> None:
    env = _functional_env("heterogeneous_relay")
    env.reset(seed=40)
    env.red_aircraft[2].state = replace(env.red_aircraft[2].state, health=0.0, alive=False, damaged=True)
    _freeze_physics(env)

    def fake_resolve(aircraft, attack_config, damage_config, rng, sample_team_order=None, armed_ids=None):
        return MultiCombatStepResult(
            _states(aircraft),
            [AttackAttempt("red_0", "blue_0", 100.0, 0.1, 300.0)],
            [ResolvedAttack("red_0", "blue_0", 100.0, 0.1, 300.0, 300.0, 0.0, True, True)],
        )

    monkeypatch.setattr("uav_env.envs.combat_multi_env.resolve_multi_attacks", fake_resolve)
    _, reward, terminated, truncated, info = env.step(np.zeros(3, dtype=np.int64))
    support = info["agent_reward_breakdowns"]["red_2"]
    assert not terminated and not truncated
    assert support.support_team_event == 0.0
    assert support.combat_event == 0.0
    assert support.total == 0.0
    assert reward == pytest.approx(sum(info["agent_rewards"].values()) / 3.0)
    env.close()


def test_support_same_step_destroy_keeps_team_event_then_loses_it_next_step(monkeypatch: pytest.MonkeyPatch) -> None:
    env = _functional_env("heterogeneous_relay")
    env.reset(seed=41)
    _freeze_physics(env)
    calls = {"count": 0}

    def fake_resolve(aircraft, attack_config, damage_config, rng, sample_team_order=None, armed_ids=None):
        calls["count"] += 1
        states = _states(aircraft)
        if calls["count"] == 1:
            states["red_2"] = replace(states["red_2"], health=0.0, alive=False, damaged=True, ever_hit=True)
            return MultiCombatStepResult(
                states,
                [
                    AttackAttempt("red_0", "blue_0", 100.0, 0.1, 300.0),
                    AttackAttempt("blue_0", "red_2", 100.0, 0.1, 300.0),
                ],
                [
                    ResolvedAttack("red_0", "blue_0", 100.0, 0.1, 300.0, 300.0, 0.0, True, True),
                    ResolvedAttack("blue_0", "red_2", 100.0, 0.1, 300.0, 300.0, 0.0, True, True),
                ],
            )
        return MultiCombatStepResult(
            states,
            [AttackAttempt("red_0", "blue_1", 100.0, 0.1, 300.0)],
            [ResolvedAttack("red_0", "blue_1", 100.0, 0.1, 300.0, 300.0, 0.0, True, True)],
        )

    monkeypatch.setattr("uav_env.envs.combat_multi_env.resolve_multi_attacks", fake_resolve)
    _, _, _, _, info = env.step(np.zeros(3, dtype=np.int64))
    support = info["agent_reward_breakdowns"]["red_2"]
    assert support.support_team_event == pytest.approx(0.575)
    assert support.attacked_event_penalty == pytest.approx(-0.9)
    assert support.destroyed_event_penalty == pytest.approx(-2.4)
    _closure(support)
    _, _, _, _, next_info = env.step(np.zeros(3, dtype=np.int64))
    dead_support = next_info["agent_reward_breakdowns"]["red_2"]
    assert dead_support.support_team_event == 0.0
    assert dead_support.total == 0.0
    env.close()


def test_mission_bonus_zero_when_support_dead_or_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    env = _functional_env("heterogeneous_relay")
    env.reset(seed=42)
    env.red_aircraft[2].state = replace(env.red_aircraft[2].state, health=0.0, alive=False, damaged=True)
    _freeze_physics(env)

    def blue_eliminated(aircraft, attack_config, damage_config, rng, sample_team_order=None, armed_ids=None):
        states = _states(aircraft)
        for blue_id in ("blue_0", "blue_1", "blue_2"):
            states[blue_id] = replace(states[blue_id], health=0.0, alive=False, damaged=True, ever_hit=True)
        return MultiCombatStepResult(states, [], [])

    monkeypatch.setattr("uav_env.envs.combat_multi_env.resolve_multi_attacks", blue_eliminated)
    _, _, terminated, _, info = env.step(np.zeros(3, dtype=np.int64))
    assert terminated
    assert info["functional_metrics"]["mission_success"] == 0.0
    assert all(item.mission_success_bonus == 0.0 for item in info["agent_reward_breakdowns"].values())
    env.close()

    timeout_env = _functional_env("heterogeneous_relay")
    timeout_env.reset(seed=43)
    timeout_env.config["max_decision_steps"] = 1
    _freeze_physics(timeout_env)
    monkeypatch.setattr(
        "uav_env.envs.combat_multi_env.resolve_multi_attacks",
        lambda aircraft, attack_config, damage_config, rng, sample_team_order=None, armed_ids=None: MultiCombatStepResult(_states(aircraft), [], []),
    )
    _, _, terminated, truncated, timeout_info = timeout_env.step(np.zeros(3, dtype=np.int64))
    assert not terminated and truncated
    assert timeout_info["outcome"].termination_reason == "timeout"
    assert all(item.mission_success_bonus == 0.0 for item in timeout_info["agent_reward_breakdowns"].values())
    timeout_env.close()


def test_support_boundary_penalty_is_multiplied_and_closed() -> None:
    env = _functional_env("heterogeneous_relay")
    env.reset(seed=44)
    env.red_aircraft[2].state = replace(env.red_aircraft[2].state, z=float(env.config["min_altitude"]), health=300.0, alive=True, damaged=False)
    _, _, _, _, info = env.step(np.zeros(3, dtype=np.int64))
    support = info["agent_reward_breakdowns"]["red_2"]
    assert support.boundary_collision_penalty == pytest.approx(-0.75)
    assert support.support_loss_adjustment == pytest.approx(-0.25)
    assert support.combat_event == pytest.approx(-0.75)
    _closure(support)
    env.close()


class _FakeActor:
    def __call__(self, obs, available=None):
        return torch.zeros((obs.shape[0], 15), dtype=torch.float32)


class _FakeAdapter:
    def __init__(self) -> None:
        self.calls = 0
        self.env = SimpleNamespace(close=lambda: None)

    def reset(self, seed: int):
        return SimpleNamespace(
            local_obs=np.zeros((3, 69), dtype=np.float32),
            available_action_mask=np.ones((3, 15), dtype=bool),
            agent_alive_mask=np.ones(3, dtype=bool),
        )

    def step(self, action):
        self.calls += 1
        aircraft_stats = {
            f"{team}_{index}": {
                "ground_crashes": 0,
                "ceiling_violations": 0,
                "collisions": 0,
                "attack_attempts": 0,
                "hits": 0,
                "effective_damage": 0.0,
                "nominal_damage": 0.0,
                "overkill_damage": 0.0,
                "attack_area_steps": 0,
            }
            for team in ("red", "blue")
            for index in range(3)
        }
        breakdowns = {
            f"red_{index}": MultiAgentRewardBreakdown(
                situation=0.0,
                event=0.0,
                raw_dense=0.0,
                assigned_dense=0.0,
                terminal=2.0,
                terminal_base_reward=2.0,
                mission_success_bonus=1.0,
                total=4.0,
                contribution_score=0.0,
            )
            for index in range(3)
        }
        return SimpleNamespace(
            local_obs=np.zeros((3, 69), dtype=np.float32),
            available_action_mask=np.ones((3, 15), dtype=bool),
            agent_alive_mask=np.ones(3, dtype=bool),
            team_reward=4.0,
            agent_reward_sum=12.0,
            terminated=True,
            truncated=False,
            info={
                "agent_reward_breakdowns": breakdowns,
                "outcome": EpisodeOutcome("red", True, False, "blue_eliminated", 1, 0.5, 3, 0),
                "statistics": {"aircraft": aircraft_stats, "collisions": 0},
            },
        )


def test_terminal_reward_proportion_includes_mission_bonus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("uav_env.algorithms.mappo.runner.make_adapter_from_description", lambda description: _FakeAdapter())
    runner = object.__new__(MAPPORunner)
    runner.config = {"validation_episodes": 1, "environment": {"kind": "3v3", "scenario": "x", "opponent": "x"}, "deterministic_evaluation": True}
    runner.num_agents = 3
    runner.device = torch.device("cpu")
    runner.actor = _FakeActor()
    result = MAPPORunner.evaluate(runner, episodes=1, seed_start=1)
    assert result["terminal_reward_proportion"] == pytest.approx(0.75)

    monkeypatch.setattr("uav_env.algorithms.happo.runner.make_adapter_from_description", lambda description: _FakeAdapter())
    happo = object.__new__(HAPPORunner)
    happo.config = {"validation_episodes": 1, "deterministic_evaluation": True}
    happo.description = CombatEnvDescription("3v3", "x", "x")
    happo.num_agents = 3
    happo.device = torch.device("cpu")
    happo.actors = [_FakeActor(), _FakeActor(), _FakeActor()]
    result_happo = HAPPORunner.evaluate(happo, episodes=1, seed_start=1)
    assert result_happo["terminal_reward_proportion"] == pytest.approx(0.75)


def test_reward_accumulator_restore_backfills_new_diagnostic_fields() -> None:
    old_names = tuple(name for name in MAPPO_COMPONENTS if name not in {"terminal_base_reward", "mission_success_bonus", "support_position_raw", "support_coverage_raw", "support_safety_raw", "support_team_event_reward", "support_loss_adjustment"})
    state = {name: np.asarray([float(index)], dtype=np.float64) for index, name in enumerate(old_names)}
    state["deprecated_extra"] = np.asarray([123.0])
    restored = restore_reward_component_accumulators(state, MAPPO_COMPONENTS, (1,), error_prefix="reward_component_episode_accumulators")
    assert restored["situation_reward"][0] == pytest.approx(0.0)
    assert restored["terminal_base_reward"][0] == 0.0
    assert restored["support_position_raw"][0] == 0.0
    assert set(restored) == set(MAPPO_COMPONENTS)

    with pytest.raises(ValueError, match="shape mismatch"):
        restore_reward_component_accumulators({"situation_reward": np.zeros(2)}, MAPPO_COMPONENTS, (1,), error_prefix="reward_component_episode_accumulators")
    with pytest.raises(ValueError, match="must be a mapping"):
        restore_reward_component_accumulators([], MAPPO_COMPONENTS, (1,), error_prefix="reward_component_episode_accumulators")
    restored_happo = restore_reward_component_accumulators(state, HAPPO_COMPONENTS, (1,), error_prefix="HAPPO reward_component_episode_accumulators")
    assert set(restored_happo) == set(HAPPO_COMPONENTS)


def test_functional_parallel_vector_short_step_shapes() -> None:
    description = CombatEnvDescription(
        "3v3",
        "head_on_functional_heterogeneous_v1",
        "greedy_combat",
        "paper_2024_exact",
        "heterogeneous_relay",
        ("combat", "combat", "support"),
        True,
    )
    vector = ParallelCombatVectorEnv(description, 1, 45)
    try:
        current = vector.reset()
        assert current["local_obs"].shape == (1, 3, 69)
        assert current["global_state"].shape == (1, 64)
        result = vector.step(np.zeros((1, 3), dtype=np.int64))
        assert result["next_local_obs"].shape == (1, 3, 69)
        assert result["next_global_state"].shape == (1, 64)
        assert np.all(np.isfinite(result["next_local_obs"]))
        assert np.all(np.isfinite(result["next_global_state"]))
    finally:
        vector.close()
