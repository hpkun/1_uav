"""Minimal persistent-wave semantics without changing the V2.3 contract."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from uav_combat.config import load_config
from uav_combat.environment.env import MultiUAVCombatEnv
from uav_combat.environment.factory import make_combat_environment
from uav_combat.environment.persistent_env import PersistentWaveCombatEnv


ROOT = Path(__file__).resolve().parents[1]


def persistent_config() -> dict:
    return load_config(ROOT / "configs/persistent_wave_environment.yaml")


def clear_blue(env: MultiUAVCombatEnv) -> None:
    for state in env.blue:
        state.alive = False


def test_factory_selects_variant_without_changing_observation_or_weapon_contract():
    env = make_combat_environment(persistent_config())
    assert isinstance(env, PersistentWaveCombatEnv)
    observation, info = env.reset(71)
    assert observation.shape == (4, 52)
    assert len(env.red_fire_states) == len(env.blue_fire_states) == 4
    assert all(hasattr(state, "armed") for state in env.red_fire_states)
    assert not any(hasattr(state, "ammo_remaining") for state in env.red_fire_states)
    assert info["wave_index"] == 1
    assert info["total_waves"] == 3


def test_intermediate_clear_preserves_red_and_spawns_nearest_policy_blue_wave():
    env = PersistentWaveCombatEnv(persistent_config())
    env.reset(2025)
    env.red[1].alive = False
    env.red[0].x = 1234.0
    clear_blue(env)

    observation, _, terminated, truncated, info = env.step(
        np.zeros((4, 3), dtype=np.float32)
    )

    assert not terminated and not truncated
    assert env.wave_index == 2 and env.waves_cleared == 1
    assert info["wave_cleared_this_step"]
    assert info["spawned_next_wave"]
    assert info["termination_reason"] == "ongoing"
    assert info["blue_survivors"] == 4
    assert info["blue_losses"] == 4
    assert observation.shape == (4, 52)
    assert not env.red[1].alive
    assert 0.0 < abs(env.red[0].x - 1234.0) < 30.0
    assert all(state.alive for state in env.blue)
    assert env.fixed_policy.__class__.__name__ == "NearestTargetPursuitPolicy"
    assert not any(
        env._in_fire_window(red, blue) or env._in_fire_window(blue, red)
        for red in env.red if red.alive for blue in env.blue
    )


def test_wave_refresh_does_not_change_the_clearing_step_reward():
    cfg = persistent_config()
    direct = MultiUAVCombatEnv(cfg)
    persistent = PersistentWaveCombatEnv(cfg)
    direct.reset(9001)
    persistent.reset(9001)
    clear_blue(direct)
    clear_blue(persistent)
    actions = np.zeros((4, 3), dtype=np.float32)

    _, direct_reward, _, _, _ = direct.step(actions)
    _, persistent_reward, _, _, _ = persistent.step(actions)

    assert np.array_equal(persistent_reward, direct_reward)


def test_final_wave_uses_original_success_termination():
    cfg = copy.deepcopy(persistent_config())
    cfg["persistent_waves"]["total_waves"] = 1
    env = PersistentWaveCombatEnv(cfg)
    env.reset(5)
    clear_blue(env)

    _, _, terminated, truncated, info = env.step(
        np.zeros((4, 3), dtype=np.float32)
    )

    assert terminated and not truncated
    assert info["red_success"] and info["red_win"]
    assert info["wave_cleared_this_step"]
    assert not info["spawned_next_wave"]
    assert info["waves_cleared"] == 1
    assert info["blue_losses"] == 4


def test_time_limit_prevents_spawning_another_wave():
    env = PersistentWaveCombatEnv(persistent_config())
    env.reset(6)
    env.steps = env.max_steps - 1
    clear_blue(env)

    _, _, terminated, truncated, info = env.step(
        np.zeros((4, 3), dtype=np.float32)
    )

    assert not terminated and truncated
    assert info["termination_reason"] == "red_failure_timeout"
    assert info["wave_index"] == 1
    assert not info["spawned_next_wave"]


def test_next_wave_generation_is_seed_deterministic():
    first = PersistentWaveCombatEnv(persistent_config())
    second = PersistentWaveCombatEnv(persistent_config())
    first.reset(12345)
    second.reset(12345)
    clear_blue(first)
    clear_blue(second)
    actions = np.zeros((4, 3), dtype=np.float32)

    first.step(actions)
    second.step(actions)

    assert np.array_equal(
        np.stack([state.as_array() for state in first.blue]),
        np.stack([state.as_array() for state in second.blue]),
    )
