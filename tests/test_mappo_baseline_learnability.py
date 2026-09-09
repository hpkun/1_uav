"""Focused, non-performance tests for the baseline learnability protocol."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from algorithm.modules.actor_lr_decay import ActorLRDecayModule
from env.factory import make_combat_environment
from env.fixed_policy import GroundAwareNearestTargetPursuitPolicy
from env.models import AircraftState
from tools.analyze_mappo_baseline_learnability import enriched, select_endpoint
from tools.preflight_mappo_baseline_learnability import (
    ENV_PATHS, freshness_scan, load_yaml, normalized_env, validate_configs,
)

ROOT = Path(__file__).resolve().parents[1]


def state(x=0.0, alive=True):
    return AircraftState(x=x, y=0.0, z=-3000.0, v=225.0, psi=0.0, theta=0.0, alive=alive)


def test_protocol_action_environment_and_modules_are_frozen():
    result = validate_configs()
    algorithm = result["algorithm"]
    assert algorithm["network"]["observation_dim"] == 52
    assert algorithm["network"]["action_dim"] == 3
    assert [k for k, v in algorithm["modules"].items() if v.get("enabled", False)] == ["actor_lr_decay"]
    assert hashlib.sha256((ROOT / "env/fixed_policy.py").read_bytes()).hexdigest() == result["manifest"]["frozen_contract"]["blue_policy_source_sha256"]


def test_ladder_diff_and_first_wave_are_exact_matched():
    envs = {k: load_yaml(v) for k, v in ENV_PATHS.items()}
    assert [(envs[k]["persistent_waves"]["total_waves"], envs[k]["simulation"]["max_steps"]) for k in ("L1", "L2", "L3")] == [(1, 1000), (2, 2000), (3, 3000)]
    assert normalized_env(envs["L1"]) == normalized_env(envs["L2"]) == normalized_env(envs["L3"])
    instances = [make_combat_environment(envs[k]) for k in ("L1", "L2", "L3")]
    observations = [env.reset(7_650_321) for env in instances]
    assert np.array_equal(observations[0][0], observations[1][0])
    assert np.array_equal(observations[0][0], observations[2][0])
    for attr in ("red", "blue"):
        arrays = [np.stack([s.as_array() for s in getattr(env, attr)]) for env in instances]
        assert np.array_equal(arrays[0], arrays[1]) and np.array_equal(arrays[0], arrays[2])
    assert envs["L1"]["weapon"] == envs["L2"]["weapon"] == envs["L3"]["weapon"]
    assert envs["L1"]["reward"] == envs["L2"]["reward"] == envs["L3"]["reward"]
    assert envs["L1"]["blue_policy"] == envs["L2"]["blue_policy"] == envs["L3"]["blue_policy"]


def test_blue_reselects_nearest_alive_without_stale_target():
    cfg = load_yaml(ENV_PATHS["L3"])
    policy = GroundAwareNearestTargetPursuitPolicy(cfg["blue_policy"], cfg["action"], cfg["aircraft"])
    own, targets = state(), [state(100.0), state(200.0)]
    assert policy.nearest_target_index(own, targets) == 0
    targets[0].alive = False
    assert policy.nearest_target_index(own, targets) == 1
    targets[1].alive = False
    assert policy.nearest_target_index(own, targets) is None


def test_lr_matches_900k_protocol_and_holds_afterward():
    cfg = validate_configs()["algorithm"]
    decay = ActorLRDecayModule(cfg["modules"]["actor_lr_decay"])
    assert decay.learning_rate(600_000, 3e-4) == pytest.approx(3e-4)
    assert decay.learning_rate(750_000, 3e-4) == pytest.approx(2e-4)
    assert decay.learning_rate(900_000, 3e-4) == pytest.approx(1e-4)
    assert decay.learning_rate(3_000_000, 3e-4) == pytest.approx(1e-4)
    assert cfg["training"]["critic_learning_rate"] == pytest.approx(3e-4)


def test_l2_artificial_wave_respawn_uses_same_policy():
    env = make_combat_environment(load_yaml(ENV_PATHS["L2"])); env.reset(7_650_322)
    policy_id = id(env.fixed_policy)
    for aircraft in env.blue: aircraft.alive = False
    _, _, terminated, truncated, info = env.step(np.zeros((4, 3), np.float32))
    assert not terminated and not truncated and info["spawned_next_wave"]
    assert info["wave_index"] == 2 and id(env.fixed_policy) == policy_id
    assert all(aircraft.alive for aircraft in env.blue)


@pytest.mark.parametrize("condition,missing", [("L1", ("w2", "w3")), ("L2", ("w3",))])
def test_analyzer_preserves_structurally_missing_wave_fields(condition, missing):
    run = {"condition": condition, "training_seed": 1}
    row = {"sampled_steps": 900000, "clear_wave_1_probability": .5,
           "average_waves_cleared": .5, "average_return": 0,
           "average_red_loss": 1, "average_blue_loss": 2,
           "average_red_ground_losses": 0, "average_red_boundary_exits": 0,
           "average_episode_length": 100, "timeout_rate": 0}
    if condition == "L2": row["clear_wave_2_probability"] = .2
    result = enriched(run, row)
    assert all(result[key] is None for key in missing)
    assert result["w1"] == .5


def test_candidate_seed_freshness_and_reserved_range_breach_are_detected():
    scan = freshness_scan(checkpoints=False)["hits"]
    assert scan["training"] == []
    assert scan["evaluation"] == []
    # Historical smoke metadata proves the requested "33M untouched" premise
    # is false; keeping this assertion prevents the launcher from hiding it.
    assert any(row["value"] == 33_000_000 for row in scan["reserved_33m"])


def test_analyzer_uses_nearest_real_evaluation_without_interpolation():
    rows = [{"sampled_steps": 811008}, {"sampled_steps": 909312}, {"sampled_steps": 1007616}]
    assert select_endpoint(rows, 900_000)["sampled_steps"] == 909312
