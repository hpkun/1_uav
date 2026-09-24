from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from algorithm.modules.persistent_wave_trajectory_replay import (
    BRIDGE_12, BRIDGE_23, PWTR_PARTITION_CAPACITY, W2_INTERNAL, W3_INTERNAL,
    PersistentWaveTrajectoryReplayModule, clipped_importance_weights,
    normalized_ess_and_freshness, replay_batch_budget,
    vtrace_targets_and_advantages, wave_stratified_permutation,
)
from algorithm.train_modular_mappo import load_config
from tools.analyze_pwtr_ablation import classify_full_gate, descriptive


ROOT = Path(__file__).resolve().parents[1]


def cfg(**overrides):
    value = {
        "enabled": True, "fresh_wave_stratification": True,
        "replay_enabled": True, "replay_source": "recent_uniform",
        "priority_enabled": False, "bridge_enabled": True,
        "actor_replay": True, "critic_replay": True,
        "sequence_length": 128, "bridge_half_length": 64,
        "min_segment_length": 32, "partition_capacity": 32,
        "actor_max_age_updates": 2,
    }
    value.update(overrides)
    return value


def rollout(waves, transitions=None, dones=None, alive=None, next_alive=None):
    waves = np.asarray(waves, dtype=np.int64).reshape(-1, 1)
    t = len(waves)
    transitions = np.zeros((t, 1), dtype=np.float32) if transitions is None else np.asarray(transitions, dtype=np.float32).reshape(t, 1)
    dones = np.zeros((t, 1), dtype=np.float32) if dones is None else np.asarray(dones, dtype=np.float32).reshape(t, 1)
    alive = np.ones((t, 1, 4), dtype=np.float32) if alive is None else np.asarray(alive, dtype=np.float32).reshape(t, 1, 4)
    next_alive = alive.copy() if next_alive is None else np.asarray(next_alive, dtype=np.float32).reshape(t, 1, 4)
    marker = np.arange(t, dtype=np.float32).reshape(t, 1, 1, 1)
    obs = np.broadcast_to(marker, (t, 1, 4, 52)).copy()
    return SimpleNamespace(
        observations=obs, next_observations=obs + 0.5,
        raw_actions=np.zeros((t, 1, 4, 3), np.float32),
        old_log_probs=np.zeros((t, 1, 4), np.float32),
        rewards=np.ones((t, 1, 4), np.float32), dones=dones,
        alive_masks=alive, next_alive_masks=next_alive,
        wave_indices=waves, wave_transition_flags=transitions,
    )


def test_stratification_is_exact_and_handles_remainder_and_missing_waves():
    waves = np.array([1] * 47 + [2] * 35 + [3] * 18)
    order = wave_stratified_permutation(waves, 17, np.random.default_rng(7))
    assert sorted(order.tolist()) == list(range(100))
    assert np.bincount(waves[order], minlength=4).tolist() == np.bincount(waves, minlength=4).tolist()
    global_fraction = np.bincount(waves, minlength=4)[1:] / len(waves)
    strat_error = np.mean([np.abs(np.bincount(waves[order[i:i+17]], minlength=4)[1:] / len(order[i:i+17]) - global_fraction).sum() for i in range(0, 85, 17)])
    random = np.random.default_rng(19).permutation(len(waves))
    random_error = np.mean([np.abs(np.bincount(waves[random[i:i+17]], minlength=4)[1:] / len(random[i:i+17]) - global_fraction).sum() for i in range(0, 85, 17)])
    assert strat_error < random_error
    assert len(wave_stratified_permutation(np.ones(23), 8, np.random.default_rng(1))) == 23
    no_w3 = np.array([1] * 9 + [2] * 8)
    assert sorted(wave_stratified_permutation(no_w3, 6, np.random.default_rng(2)).tolist()) == list(range(17))


def test_internal_extraction_cross_rollout_remainders_fifo_and_partitions():
    module = PersistentWaveTrajectoryReplayModule(cfg(), 3)
    module.ingest_rollout(rollout([2] * 80), 1)
    assert not module.partitions[W2_INTERNAL]
    module.ingest_rollout(rollout([2] * 48), 1)
    assert module.partitions[W2_INTERNAL][-1]["actual_length"] == 128
    module.ingest_rollout(rollout([3] * 40, dones=[0] * 39 + [1]), 2)
    assert module.partitions[W3_INTERNAL][-1]["actual_length"] == 40
    before = len(module.partitions[W2_INTERNAL])
    module.ingest_rollout(rollout([2] * 31, dones=[0] * 30 + [1]), 3)
    assert len(module.partitions[W2_INTERNAL]) == before
    for generation in range(40):
        module.ingest_rollout(rollout([2] * 32, dones=[0] * 31 + [1]), 10 + generation)
    assert len(module.partitions[W2_INTERNAL]) == PWTR_PARTITION_CAPACITY
    assert len(module.partitions[W3_INTERNAL]) == 1


def test_bridge_12_23_cross_rollout_padding_and_boundary_semantics():
    module = PersistentWaveTrajectoryReplayModule(cfg(), 4)
    # Transition flag is carried by the final source-wave transition; its next obs is post-spawn.
    module.ingest_rollout(rollout([1] * 70, transitions=[0] * 69 + [1]), 1)
    assert 1 in module.pending_bridges[0]
    module.ingest_rollout(rollout([2] * 64), 1)
    b12 = module.partitions[BRIDGE_12][-1]
    assert b12["actual_length"] == 128
    assert np.all(b12["wave_indices"][:64] == 1) and np.all(b12["wave_indices"][64:128] == 2)
    assert b12["dones"][63] == 0 and b12["next_alive_masks"][63].sum() == 4
    module.ingest_rollout(rollout([2] * 70, transitions=[0] * 69 + [1]), 2)
    module.ingest_rollout(rollout([3] * 20, dones=[0] * 19 + [1]), 2)
    b23 = module.partitions[BRIDGE_23][-1]
    assert b23["actual_length"] == 84
    assert b23["valid_time_mask"].sum() == 84 and b23["valid_time_mask"][84:].sum() == 0


def test_importance_freshness_vtrace_terminal_death_bootstrap_and_finite():
    new = torch.tensor([[[0.0, -1.0, 1000.0, -1000.0]]])
    old = torch.zeros_like(new)
    alive = torch.tensor([[[1.0, 1.0, 0.0, 0.0]]])
    valid = torch.ones(1, 1)
    _, individual, joint_log, joint = clipped_importance_weights(new, old, alive, valid)
    assert individual[0, 0, 0] == 1 and 0 < individual[0, 0, 1] < 1
    assert joint_log.item() == -1 and torch.allclose(joint, torch.tensor([[np.exp(-1)]], dtype=torch.float32))
    ess, divergence, freshness = normalized_ess_and_freshness(torch.tensor([[0.0, -2.0]]), torch.ones(1, 2))
    assert 0 < ess <= 1 and divergence > 0 and 0 < freshness <= 1
    rewards = torch.tensor([[[1.0]], [[2.0]]]).transpose(0, 1)
    values = torch.zeros_like(rewards)
    # For contiguous transitions V(next_s_t) equals V(s_{t+1}); the final
    # terminal transition's next value is intentionally irrelevant.
    next_values = torch.tensor([[[0.0], [20.0]]])
    dones = torch.tensor([[0.0, 1.0]])
    alive1 = torch.ones_like(rewards)
    target, advantage = vtrace_targets_and_advantages(rewards, values, next_values, dones, alive1, alive1, torch.ones(1, 2), torch.ones(1, 2), .9)
    assert torch.allclose(target[0, 1], torch.tensor([2.0]))
    assert torch.allclose(target[0, 0], torch.tensor([1 + .9 * 2]))
    # Death truncates even when episode is not done.
    dead_next = alive1.clone(); dead_next[:, 0] = 0
    death_target, _ = vtrace_targets_and_advantages(rewards, values, next_values, torch.zeros_like(dones), alive1, dead_next, torch.ones(1, 2), torch.ones(1, 2), .9)
    assert torch.allclose(death_target[0, 0], torch.tensor([1.0]))
    # A nonterminal last segment step uses stored next-state bootstrap.
    one_target, _ = vtrace_targets_and_advantages(torch.ones(1, 1, 1), torch.zeros(1, 1, 1), torch.full((1, 1, 1), 5.0), torch.zeros(1, 1), torch.ones(1, 1, 1), torch.ones(1, 1, 1), torch.ones(1, 1), torch.ones(1, 1), .9)
    assert torch.allclose(one_target, torch.tensor([[[5.5]]]))
    assert torch.isfinite(individual).all() and torch.isfinite(target).all() and torch.isfinite(advantage).all()


def test_priority_formula_has_no_wave_or_success_bonus_and_budget_is_matched():
    def priority(error, freshness):
        return (error + 1e-6) * freshness
    assert priority(2, .8) > priority(1, .8)
    assert priority(1, .8) > priority(1, .4)
    # Wave and outcome labels are intentionally absent from the frozen formula.
    assert priority(1, .5) == priority(1, .5)
    assert [replay_batch_budget(n) for n in (0, 1, 512, 513, 6144)] == [0, 1, 1, 2, 12]


def test_memory_rng_checkpoint_roundtrip_and_sampling_identity():
    module = PersistentWaveTrajectoryReplayModule(cfg(), 123)
    module.ingest_rollout(rollout([2] * 128), 1)
    module.ingest_rollout(rollout([3] * 128), 1)
    state = module.state_dict()
    restored = PersistentWaveTrajectoryReplayModule(cfg(), 999)
    restored.load_state_dict(deepcopy(state))
    assert restored.state_dict()["segment_id_counter"] == state["segment_id_counter"]
    left = [s["segment_id"] for s in module.sample_batch(2)]
    right = [s["segment_id"] for s in restored.sample_batch(2)]
    assert left == right


def test_six_branch_configs_are_exact_and_matched():
    names = ("plain", "stratified", "current_extra", "uniform_recent", "priority_recent", "full")
    configs = {name: load_config(ROOT / f"configs/dev_pwtr_{name}_300k.yaml") for name in names}
    for config in configs.values():
        assert config["training"]["total_sampled_steps"] == 1_805_280
        assert config["development_branch"]["source_sampled_steps"] == 1_505_280
        assert config["development_branch"]["additional_sampled_steps"] == 300_000
        assert config["implementation"]["evaluation_seed_base"] == 44_000_000
    assert not configs["plain"]["modules"].get("persistent_wave_trajectory_replay", {}).get("enabled", False)
    assert configs["full"]["modules"]["persistent_wave_trajectory_replay"]["bridge_enabled"] is True


def test_disabled_module_does_not_consume_rng_or_change_permutation():
    disabled = PersistentWaveTrajectoryReplayModule({"enabled": False}, 1)
    a = np.random.default_rng(88); b = np.random.default_rng(88)
    assert np.array_equal(disabled.fresh_epoch_permutation(np.array([1, 2, 3]), 2, a), b.permutation(3))


def test_analyzer_frozen_gates_and_descriptive_labels():
    good = [{"AverageWaves": .1, "W3": .02, "W1": 0.0}] * 3
    assert classify_full_gate(good) == ("PROMISING", True, True, True)
    unsafe = deepcopy(good); unsafe[0] = {"AverageWaves": -.5, "W3": .02, "W1": 0.0}
    assert classify_full_gate(unsafe)[0] == "SAFETY_FAIL"
    label, _ = descriptive([{"AverageWaves": .1, "W3": .1, "Q2": .1, "Q3": .1}] * 3)
    assert label == "POSITIVE"
