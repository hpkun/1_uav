from __future__ import annotations

import numpy as np
import pytest

from uav_env.actions.discrete_15 import DiscreteAction15, get_control
from uav_env.algorithms.mappo.config import load_mappo_config
from uav_env.combat.multi_combat import resolve_multi_attacks
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
    rewards = info["agent_rewards"]
    assert reward == pytest.approx(sum(rewards.values()) / 3.0)
    env.close()
