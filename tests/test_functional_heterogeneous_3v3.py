from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from uav_env.actions.discrete_15 import DiscreteAction15, get_control
from uav_env.algorithms.mappo.config import load_mappo_config
from uav_env.combat.multi_combat import AttackAttempt, MultiCombatStepResult, ResolvedAttack, resolve_multi_attacks
from uav_env.core.enums import Team
from uav_env.entities.uav import UAV
from uav_env.dynamics.propagation import propagate_state
from uav_env.envs import make_3v3_env


CONFIGS = {
    "homogeneous_control": "configs/mappo_functional_homogeneous_3v3.yaml",
    "heterogeneous_no_relay": "configs/mappo_functional_heterogeneous_no_relay_3v3.yaml",
    "heterogeneous_relay": "configs/mappo_functional_heterogeneous_relay_3v3.yaml",
}


def _env(mode: str, seed: int = 1):
    cfg = load_mappo_config(CONFIGS[mode])["environment"]
    return make_3v3_env(
        cfg["scenario"], cfg["opponent"], seed=seed,
        multi_terminal_reward_profile=cfg["multi_terminal_reward_profile"],
        functional_mode=cfg["functional_mode"], red_roles=cfg["red_roles"], relay_enabled=cfg["relay_enabled"],
    )


def _freeze_physics(env) -> None:
    env._propagate_all = lambda action_map: ([], {}, 0)


def _states(aircraft) -> dict[str, object]:
    return {u.uav_id: u.state.copy() for u in aircraft}


def _assert_reward_closure(breakdown) -> None:
    assert breakdown.dense_reward == pytest.approx(breakdown.assigned_shape + breakdown.combat_event)
    assert breakdown.assigned_dense == pytest.approx(breakdown.assigned_shape)
    assert breakdown.terminal == pytest.approx(breakdown.terminal_base_reward)
    assert breakdown.total == pytest.approx(
        breakdown.assigned_shape
        + breakdown.combat_event
        + breakdown.terminal_base_reward
        + breakdown.mission_success_bonus
    )


def test_functional_configs_resolve_and_keep_common_training_fields() -> None:
    loaded = {key: load_mappo_config(path) for key, path in CONFIGS.items()}
    common_keys = (
        "seed", "num_envs", "vector_env", "rollout_length", "total_env_steps", "evaluation_interval",
        "validation_seed_start", "validation_episodes", "test_seed_start", "test_episodes",
        "checkpoint_interval", "checkpoint_selection", "run_symmetric_stress_test",
        "gamma", "gae_lambda", "clip_param", "value_clip_param", "ppo_epochs", "num_mini_batches",
        "actor_lr", "critic_lr", "entropy_coef", "value_loss_coef", "max_grad_norm",
        "actor_hidden_sizes", "critic_hidden_sizes", "deterministic_evaluation",
    )
    baseline = loaded["homogeneous_control"]
    for config in loaded.values():
        for key in common_keys:
            assert config[key] == baseline[key]
        assert config["environment"]["scenario"] == "head_on_functional_heterogeneous_v1"
        assert config["environment"]["opponent"] == "greedy_combat"


def test_functional_reset_shapes_and_old_v2_unchanged() -> None:
    for mode in CONFIGS:
        env = _env(mode)
        obs, info = env.reset(seed=10)
        assert obs.shape == (3, 69)
        assert info["global_state"].shape == (64,)
        assert len(info["local_observation_feature_names"]) == 69
        assert len(info["global_state_feature_names"]) == 64
        assert np.all(np.isfinite(obs))
        assert np.all(np.abs(obs) <= 1.0 + 1e-12)
        env.close()
    old = make_3v3_env("head_on_mirrored_jitter_v2", "greedy_combat", seed=10, multi_terminal_reward_profile="paper_2024_exact")
    obs, info = old.reset(seed=10)
    assert obs.shape == (3, 63)
    assert info["global_state"].shape == (61,)
    old.close()


def test_initial_motion_states_are_identical_across_functional_modes() -> None:
    states = []
    for mode in CONFIGS:
        env = _env(mode, seed=123)
        env.reset(seed=123)
        states.append([(u.state.x, u.state.y, u.state.z, u.state.speed, u.state.heading_angle) for u in env.all_aircraft])
        env.close()
    assert states[0] == pytest.approx(states[1])
    assert states[0] == pytest.approx(states[2])
    red0, red1, red2 = states[2][:3]
    assert red2[0] < red0[0]
    assert red0[1] < red2[1] < red1[1]


def test_roles_do_not_change_profile_action_or_dynamics() -> None:
    env = _env("heterogeneous_relay")
    env.reset(seed=3)
    assert env.red_aircraft[0].profile is env.red_aircraft[2].profile
    control = get_control(DiscreteAction15.CLIMB_ACCELERATE)
    combat_next = propagate_state(env.red_aircraft[0].state.copy(), control, env.profile, 0.1, 9.81)
    support_state = env.red_aircraft[0].state.copy()
    support_next = propagate_state(support_state, control, env.profile, 0.1, 9.81)
    assert combat_next.to_kinematic_vector() == pytest.approx(support_next.to_kinematic_vector())
    env.close()


def test_visibility_no_relay_vs_relay_and_support_death() -> None:
    relay = _env("heterogeneous_relay")
    no_relay = _env("heterogeneous_no_relay")
    for env in (relay, no_relay):
        red_states, blue_states = env._initialize_scenario("head_on_functional_heterogeneous_v1")
        red_states[0].x, red_states[0].y = -2000.0, 0.0
        red_states[1].x, red_states[1].y = -2000.0, 500.0
        red_states[2].x, red_states[2].y = 0.0, 0.0
        blue_states[0].x, blue_states[0].y = 1800.0, 0.0
        blue_states[1].alive = False
        blue_states[2].alive = False
        _, info = env.reset(seed=1, options={"red_states": red_states, "blue_states": blue_states})
        assert info["enemy_local_visible_masks"][2, 0] == 1
    assert no_relay.reset(seed=1, options={"red_states": red_states, "blue_states": blue_states})[1]["enemy_visible_masks"][0, 0] == 0
    relay_info = relay.reset(seed=1, options={"red_states": red_states, "blue_states": blue_states})[1]
    assert relay_info["enemy_relay_visible_masks"][0, 0] == 1
    assert relay_info["enemy_visible_masks"][0, 0] == 1
    relay.red_aircraft[2].state.alive = False
    assert relay._visibility_masks()["final"][0, 0] == 0
    relay.close()
    no_relay.close()


def test_support_unarmed_but_blue_can_attack_support() -> None:
    env = _env("heterogeneous_relay")
    support = UAV("red_2", int(Team.RED), env._state(Team.RED, 0.0, 0.0, 1800.0, 110.0, 0.0), env.profile)
    blue = UAV("blue_0", int(Team.BLUE), env._state(Team.BLUE, -500.0, 0.0, 1800.0, 110.0, 0.0), env.profile)
    result = resolve_multi_attacks([support, blue], env.attack_config, env.damage_config, np.random.default_rng(1), armed_ids={"blue_0"})
    attempts = result.attack_attempts
    assert all(attempt.attacker_id != "red_2" for attempt in attempts)
    assert any(attempt.attacker_id == "blue_0" and attempt.target_id == "red_2" for attempt in attempts)
    env.close()


def test_homogeneous_control_red2_is_armed() -> None:
    env = _env("homogeneous_control")
    red_states, blue_states = env._initialize_scenario("head_on_functional_heterogeneous_v1")
    for state in red_states + blue_states:
        state.alive = False
    red_states[2] = env._state(Team.RED, 0.0, 0.0, 1800.0, 110.0, 0.0)
    red_states[2].alive = True
    blue_states[0] = env._state(Team.BLUE, 500.0, 0.0, 1800.0, 110.0, 0.0)
    blue_states[0].alive = True
    env.reset(seed=2, options={"red_states": red_states, "blue_states": blue_states})
    _, _, _, _, info = env.step(np.asarray([0, 0, 0]))
    assert any(attempt.attacker_id == "red_2" for attempt in info["attack_attempts"])
    env.close()


def test_invalid_functional_role_configs_are_rejected() -> None:
    with pytest.raises(ValueError):
        make_3v3_env("head_on_functional_heterogeneous_v1", "greedy_combat", red_roles=["combat", "support", "support"], functional_mode="heterogeneous_relay", relay_enabled=True)
    with pytest.raises(ValueError):
        make_3v3_env("head_on_functional_heterogeneous_v1", "greedy_combat", red_roles=["combat", "combat", "support"], functional_mode="heterogeneous_no_relay", relay_enabled=True)
    env = _env("heterogeneous_relay")
    bad = dict(env.config)
    bad["support_detection_range"] = bad["combat_detection_range"]
    from uav_env.envs.combat_multi_env import CombatMultiEnv
    with pytest.raises(ValueError):
        CombatMultiEnv(bad, "head_on_functional_heterogeneous_v1", "greedy_combat")
    env.close()


def test_functional_short_step_metrics_and_reward_consistency() -> None:
    env = _env("heterogeneous_relay")
    obs, info = env.reset(seed=4)
    obs, reward, terminated, truncated, info = env.step(np.asarray([0, 0, 0]))
    assert obs.shape == (3, 69)
    assert np.isfinite(reward)
    assert "functional_metrics" in info
    assert "support_detection_coverage_mean" in info["functional_metrics"]
    assert "combat_attack_attempts_total" in info["functional_metrics"]
    assert "combat_hits_total" in info["functional_metrics"]
    assert "combat_effective_damage_total" in info["functional_metrics"]
    rewards = info["agent_rewards"]
    assert reward == pytest.approx(sum(rewards.values()) / 3.0)
    env.close()


def test_functional_combat_events_enter_final_reward(monkeypatch) -> None:
    env = _env("homogeneous_control")
    env.reset(seed=20)
    _freeze_physics(env)

    def fake_resolve(aircraft, attack_config, damage_config, rng, sample_team_order=None, armed_ids=None):
        return MultiCombatStepResult(
            _states(aircraft),
            [AttackAttempt("red_0", "blue_0", 100.0, 0.1, 50.0)],
            [ResolvedAttack("red_0", "blue_0", 100.0, 0.1, 50.0, 50.0, 0.0, True, False)],
        )

    monkeypatch.setattr("uav_env.envs.combat_multi_env.resolve_multi_attacks", fake_resolve)
    _, _, _, _, info = env.step(np.asarray([0, 0, 0]))
    breakdown = info["agent_reward_breakdowns"]["red_0"]
    assert breakdown.hit_event_reward == pytest.approx(0.8)
    assert breakdown.combat_event == pytest.approx(0.8)
    _assert_reward_closure(breakdown)
    assert info["agent_rewards"]["red_0"] == pytest.approx(breakdown.total)
    env.close()


def test_support_team_event_is_positive_event_share_and_capped(monkeypatch) -> None:
    env = _env("heterogeneous_relay")
    env.reset(seed=21)
    _freeze_physics(env)

    def fake_resolve(aircraft, attack_config, damage_config, rng, sample_team_order=None, armed_ids=None):
        return MultiCombatStepResult(
            _states(aircraft),
            [
                AttackAttempt("red_0", "blue_0", 100.0, 0.1, 300.0),
                AttackAttempt("red_1", "blue_1", 100.0, 0.1, 300.0),
            ],
            [
                ResolvedAttack("red_0", "blue_0", 100.0, 0.1, 300.0, 300.0, 0.0, True, True),
                ResolvedAttack("red_1", "blue_1", 100.0, 0.1, 300.0, 300.0, 0.0, True, True),
            ],
        )

    monkeypatch.setattr("uav_env.envs.combat_multi_env.resolve_multi_attacks", fake_resolve)
    _, _, _, _, info = env.step(np.asarray([0, 0, 0]))
    support = info["agent_reward_breakdowns"]["red_2"]
    assert support.support_team_event == pytest.approx(1.0)
    assert support.combat_event == pytest.approx(1.0)
    _assert_reward_closure(support)
    env.close()


def test_support_team_event_excludes_negative_events(monkeypatch) -> None:
    env = _env("heterogeneous_relay")
    env.reset(seed=22)
    _freeze_physics(env)

    def fake_resolve(aircraft, attack_config, damage_config, rng, sample_team_order=None, armed_ids=None):
        return MultiCombatStepResult(
            _states(aircraft),
            [AttackAttempt("blue_0", "red_0", 100.0, 0.1, 50.0)],
            [ResolvedAttack("blue_0", "red_0", 100.0, 0.1, 50.0, 50.0, 0.0, True, False)],
        )

    monkeypatch.setattr("uav_env.envs.combat_multi_env.resolve_multi_attacks", fake_resolve)
    _, _, _, _, info = env.step(np.asarray([0, 0, 0]))
    assert info["agent_reward_breakdowns"]["red_0"].attacked_event_penalty == pytest.approx(-0.9)
    assert info["agent_reward_breakdowns"]["red_2"].support_team_event == pytest.approx(0.0)
    env.close()


def test_support_loss_multiplier_is_explicit_breakdown(monkeypatch) -> None:
    env = _env("heterogeneous_relay")
    env.reset(seed=23)
    _freeze_physics(env)

    def fake_resolve(aircraft, attack_config, damage_config, rng, sample_team_order=None, armed_ids=None):
        states = _states(aircraft)
        states["red_2"] = replace(states["red_2"], health=0.0, alive=False, damaged=True, ever_hit=True)
        return MultiCombatStepResult(
            states,
            [AttackAttempt("blue_0", "red_2", 100.0, 0.1, 300.0)],
            [ResolvedAttack("blue_0", "red_2", 100.0, 0.1, 300.0, 300.0, 0.0, True, True)],
        )

    monkeypatch.setattr("uav_env.envs.combat_multi_env.resolve_multi_attacks", fake_resolve)
    _, _, _, _, info = env.step(np.asarray([0, 0, 0]))
    support = info["agent_reward_breakdowns"]["red_2"]
    assert support.attacked_event_penalty == pytest.approx(-0.9)
    assert support.destroyed_event_penalty == pytest.approx(-2.4)
    assert support.support_loss_adjustment == pytest.approx(-0.8)
    assert support.combat_event == pytest.approx(-3.3)
    _assert_reward_closure(support)
    env.close()


def test_heterogeneous_mission_success_bonus_is_terminal_only(monkeypatch) -> None:
    env = _env("heterogeneous_relay")
    env.reset(seed=24)
    _freeze_physics(env)

    def fake_resolve(aircraft, attack_config, damage_config, rng, sample_team_order=None, armed_ids=None):
        states = _states(aircraft)
        for blue_id in ("blue_0", "blue_1", "blue_2"):
            states[blue_id] = replace(states[blue_id], health=0.0, alive=False, damaged=True, ever_hit=True)
        return MultiCombatStepResult(
            states,
            [
                AttackAttempt("red_0", "blue_0", 100.0, 0.1, 300.0),
                AttackAttempt("red_1", "blue_1", 100.0, 0.1, 300.0),
                AttackAttempt("red_0", "blue_2", 100.0, 0.1, 300.0),
            ],
            [
                ResolvedAttack("red_0", "blue_0", 100.0, 0.1, 300.0, 300.0, 0.0, True, True),
                ResolvedAttack("red_1", "blue_1", 100.0, 0.1, 300.0, 300.0, 0.0, True, True),
                ResolvedAttack("red_0", "blue_2", 100.0, 0.1, 300.0, 300.0, 0.0, True, True),
            ],
        )

    monkeypatch.setattr("uav_env.envs.combat_multi_env.resolve_multi_attacks", fake_resolve)
    _, _, terminated, truncated, info = env.step(np.asarray([0, 0, 0]))
    assert terminated
    assert not truncated
    assert info["outcome"].termination_reason == "blue_eliminated"
    assert info["functional_metrics"]["mission_success"] == pytest.approx(1.0)
    for breakdown in info["agent_reward_breakdowns"].values():
        assert breakdown.mission_success_bonus == pytest.approx(1.0)
        _assert_reward_closure(breakdown)
    env.close()


def test_homogeneous_control_has_no_support_metrics_or_mission_bonus(monkeypatch) -> None:
    env = _env("homogeneous_control")
    env.reset(seed=25)
    _freeze_physics(env)

    def fake_resolve(aircraft, attack_config, damage_config, rng, sample_team_order=None, armed_ids=None):
        states = _states(aircraft)
        for blue_id in ("blue_0", "blue_1", "blue_2"):
            states[blue_id] = replace(states[blue_id], health=0.0, alive=False, damaged=True, ever_hit=True)
        return MultiCombatStepResult(
            states,
            [],
            [ResolvedAttack("red_2", "blue_2", 100.0, 0.1, 300.0, 300.0, 0.0, True, True)],
        )

    monkeypatch.setattr("uav_env.envs.combat_multi_env.resolve_multi_attacks", fake_resolve)
    _, _, terminated, _, info = env.step(np.asarray([0, 0, 0]))
    assert terminated
    metrics = info["functional_metrics"]
    assert metrics["has_support_agent"] == pytest.approx(0.0)
    assert metrics["support_metrics_applicable"] == pytest.approx(0.0)
    assert metrics["mission_success"] == pytest.approx(0.0)
    assert all(b.mission_success_bonus == 0.0 for b in info["agent_reward_breakdowns"].values())
    env.close()
