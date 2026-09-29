from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from algorithm.modular_mappo.networks import ModularMAPPOActor
from env.factory import make_combat_environment
from tools import audit_fixed10_deployment_modes as audit

ROOT = Path(__file__).resolve().parents[1]


def _episode(waves: int, seed: int = 88_330_000, role: str = "Peak", stream=0):
    return {"checkpoint_role": role, "environment_seed": seed, "policy_rng_stream_id": stream,
            "waves_cleared": waves, "team_episode_return": float(waves), "red_losses": 3 - waves,
            "red_boundary_exits": 0, "red_ground_losses": 0, "episode_length": 100 + waves}


def test_checkpoint_selection_is_exact_peak_and_final():
    assert audit.CHECKPOINTS["Peak"]["path"].name == "best_eval.pt"
    assert audit.CHECKPOINTS["Peak"]["step"] == 1_701_888
    assert audit.CHECKPOINTS["Final"]["path"].name == "final.pt"
    assert audit.CHECKPOINTS["Final"]["step"] == 1_805_280


def test_45m_rejected_and_88m_accepted():
    assert audit.validate_environment_seeds(audit.ENVIRONMENT_SEEDS) == audit.ENVIRONMENT_SEEDS
    with pytest.raises(RuntimeError, match="45M"):
        audit.validate_environment_seeds([45_000_000])


def test_historical_deterministic_exactly_matches_two_checkpoints_and_16_seeds():
    rows = audit.read_csv(audit.HISTORICAL_DETERMINISTIC)
    selected = audit.validate_historical_deterministic(rows)
    assert len(selected) == 32
    for role in ("Peak", "Final"):
        group = [row for row in selected if row["checkpoint_role"] == role]
        assert {row["environment_seed"] for row in group} == set(audit.ENVIRONMENT_SEEDS)
        assert all(row["data_source"] == "HISTORICAL_REUSED" for row in group)


def test_historical_duplicate_missing_and_mismatch_fail_closed():
    source = audit.read_csv(audit.HISTORICAL_DETERMINISTIC)
    selected_missing = deepcopy(source)
    selected_missing.pop(next(i for i, row in enumerate(selected_missing) if row["policy_id"] == audit.CHECKPOINTS["Final"]["id"]))
    with pytest.raises(RuntimeError):
        audit.validate_historical_deterministic(selected_missing)
    duplicate = deepcopy(source)
    final_index = next(i for i, row in enumerate(duplicate) if row["policy_id"] == audit.CHECKPOINTS["Final"]["id"])
    duplicate[final_index + 1] = dict(duplicate[final_index])
    with pytest.raises(RuntimeError):
        audit.validate_historical_deterministic(duplicate)
    mismatch = deepcopy(source); mismatch[0]["checkpoint_step"] = "1"
    with pytest.raises(RuntimeError):
        audit.validate_historical_deterministic(mismatch)


def test_wave_metrics_aw_q2_q3_are_correct():
    rows = [_episode(wave, seed=88_330_000 + i) for i, wave in enumerate((0, 1, 2, 3))]
    result = audit.aggregate(rows)
    assert result["W1"] == .75 and result["W2"] == .5 and result["W3"] == .25
    assert result["AverageWaves"] == 1.5
    assert result["Q2"] == pytest.approx(2 / 3) and result["Q3"] == .5


def test_conditional_ratios_zero_denominator_are_null_with_reason():
    result = audit.aggregate([_episode(0)])
    assert result["Q2"] is None and result["Q2_undefined_reason"] == "W1_ZERO"
    assert result["Q3"] is None and result["Q3_undefined_reason"] == "W2_ZERO"


def test_peak_final_pairing_key_includes_environment_and_stream():
    rows = [_episode(1, role=role, stream=stream) for stream in (0, 1) for role in ("Peak", "Final")]
    pairs = audit.paired_rows([], rows)
    assert len(pairs) == 2
    assert {(row["environment_seed"], row["policy_rng_stream_id"]) for row in pairs} == {(88_330_000, 0), (88_330_000, 1)}


def test_policy_rng_same_stream_reproducible_and_different_stream_distinct():
    seed0 = audit.policy_rng_seed(88_330_000, 0)
    seed1 = audit.policy_rng_seed(88_330_000, 1)
    assert seed0 != seed1
    with audit.isolated_policy_rng(seed0, "cpu"):
        first = torch.randn(12)
    with audit.isolated_policy_rng(seed0, "cpu"):
        second = torch.randn(12)
    with audit.isolated_policy_rng(seed1, "cpu"):
        third = torch.randn(12)
    assert torch.equal(first, second)
    assert not torch.equal(first, third)


def test_policy_rng_fork_restores_global_torch_state():
    torch.manual_seed(123); before = torch.random.get_rng_state().clone()
    with audit.isolated_policy_rng(999, "cpu"):
        _ = torch.randn(20)
    assert torch.equal(before, torch.random.get_rng_state())


def test_environment_numpy_rng_is_isolated_from_torch_policy_rng():
    config = yaml.safe_load((ROOT / "configs/persistent_wave_v2_environment.yaml").read_text(encoding="utf-8"))
    results = []
    for policy_seed in (101, 202):
        with audit.isolated_policy_rng(policy_seed, "cpu"):
            _ = torch.randn(30)
            env = make_combat_environment(deepcopy(config)); obs, _ = env.reset(88_330_000)
            results.append(env.step(np.zeros((4, 3), np.float32))[:2])
    assert np.array_equal(results[0][0], results[1][0])
    assert np.array_equal(results[0][1], results[1][1])


def test_full_requires_exactly_96_unique_stochastic_episodes():
    rows = []
    for role in audit.CHECKPOINTS:
        for seed in audit.ENVIRONMENT_SEEDS:
            for stream in audit.POLICY_STREAM_IDS:
                rows.append(_episode(0, seed, role, stream))
    audit.validate_stochastic_count(rows, full=True)
    with pytest.raises(RuntimeError):
        audit.validate_stochastic_count(rows[:-1], full=True)


def test_smoke_cannot_be_labeled_full():
    rows = [_episode(0, role="Peak"), _episode(0, role="Final")]
    audit.validate_stochastic_count(rows, full=False)
    summary = {role: audit.aggregate([row]) for role, row in zip(("Peak", "Final"), rows)}
    assert audit.descriptive_label(summary, summary, full=False) == "INSUFFICIENT_EVIDENCE"


def test_actor_eval_no_grad_sampling_and_parameters_unchanged():
    actor = ModularMAPPOActor(52, 3, 32).eval()
    before = {key: value.clone() for key, value in actor.state_dict().items()}
    with torch.no_grad(), audit.isolated_policy_rng(77, "cpu"):
        distribution, _ = actor.distribution_step(torch.zeros(1, 4, 52), alive_mask=torch.ones(1, 4))
        action = torch.tanh(distribution.rsample())
    assert not actor.training and not action.requires_grad
    assert all(torch.equal(value, before[key]) for key, value in actor.state_dict().items())


def test_actual_checkpoint_protocol_and_sha_are_valid():
    _, _, metadata = audit.validate_protocol()
    assert metadata["Peak"]["sampled_steps"] == 1_701_888
    assert metadata["Final"]["sampled_steps"] == 1_805_280
    assert metadata["Peak"]["checkpoint_sha256"] != metadata["Final"]["checkpoint_sha256"]


def test_tool_has_no_training_backward_or_optimizer_path():
    source = (ROOT / "tools/audit_fixed10_deployment_modes.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    attrs = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
    assert "backward" not in attrs
    assert "optimizer.step" not in source
    assert "resume(" not in source
