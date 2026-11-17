"""Independent 20v20 event-driven nearest assignment and full episode contracts."""
import json

import numpy as np
import pytest

from algorithm.ideal_20v20_nearest_guidance import run_episode, TRACE_VERSION
from env.ideal_interception.assignment import nearest_distance_assignment as original_nearest
from env.ideal_interception.config import PROTOCOL as OLD_PROTOCOL, DEFENDER_SLOTS
from env.ideal_interception.large_scale_20v20_env import (TwentyConfig,
    TwentyVTwentyEnv, PROTOCOL_20V20)
from tools import ideal_20v20_nearest_benchmark as benchmark_module


def _neutral():
    return np.full((20, 3), .5)


def _approach_target(env, defender_ids, offsets):
    target = env.intruders[0, :3].copy()
    velocity = env.intruders[0, 3:].copy()
    heading = velocity/np.linalg.norm(velocity)
    actions = _neutral()
    for i, offset in zip(defender_ids, offsets):
        env.defenders[i, :3] = target+offset*heading
        actions[i] = .5-.5*heading
    return actions


def test_20v20_level2_initialization_and_event_driven_assignment():
    env = TwentyVTwentyEnv()
    _, info = env.reset(17)
    repeated = TwentyVTwentyEnv()
    _, again = repeated.reset(17)
    assert PROTOCOL_20V20 != OLD_PROTOCOL and DEFENDER_SLOTS == 15
    assert env.config.assignment_policy == "initial_nearest_target_resolution_fallback"
    assert info["defenders"].shape == info["intruders"].shape == (20, 6)
    assert info["defender_mask"].all() and info["intruder_mask"].all()
    assert info["initial_unique_target_count"] == 1
    assert info["initial_max_defenders_per_target"] == 20
    assert set(info["assignment"].values()) == {0}
    assert len(info["intruder_paths"]) == 20
    assert len(set(path["polyline_sha256"] for path in info["intruder_paths"])) == 20
    assert all(plan["policy"] == "level2_maneuvering" for plan in info["intruder_plans"])
    assert all(np.max(np.abs(info["intruders"][j, :3]-env.config.intruder_starts[j])) <= 5.
               for j in range(20))
    np.testing.assert_array_equal(info["defenders"], again["defenders"])
    np.testing.assert_array_equal(info["intruders"], again["intruders"])
    assert info["intruder_paths"] == again["intruder_paths"]
    original = info["assignment"].copy()
    _, _, terminated, _, after = env.step(_neutral())
    assert not terminated and not after["events"]
    assert after["assignment"] == original
    assert after["assignment_update_count"] == 0
    assert after["assignment_updates"] == []


def test_earliest_contact_retires_only_one_pursuer_and_reassigns_others():
    env = TwentyVTwentyEnv()
    env.reset(17)
    actions = _approach_target(env, (0, 1), (.5, 1.))
    _, _, terminated, _, info = env.step(actions)
    contacts = [event for event in info["events"] if event["kind"] == "interception_contact"]
    assert not terminated
    assert len(contacts) == 1 and contacts[0]["members"] == [0, 20]
    assert contacts[0]["time"] == pytest.approx((.5-.1)/40., abs=1e-5)
    assert not info["defender_mask"][0] and info["defender_mask"][1]
    assert not info["intruder_mask"][0]
    assert info["defender_mask"].sum() == 19
    assert set(info["assignment"]) == set(range(1, 20))
    assert 0 not in info["assignment"].values()
    assert info["assignment_update_count"] == 1
    assert info["assignment_updates"][0]["time"] == pytest.approx(contacts[0]["time"])
    assert info["assignment_updates"][0]["assignment"] == info["assignment"]
    assert info["assignment_updates"][0]["reason"] == "target_resolution_nearest_reassignment"
    assert info["assignment_updates"][0]["resolved_target"] == 0
    assert info["assignment_updates"][0]["affected_defenders"] == list(range(1, 20))
    assert info["intercepted_count"] == 1 and info["unassigned_contact_count"] == 0


def test_active_target_is_sticky_even_when_another_target_becomes_closer():
    env = TwentyVTwentyEnv()
    env.reset(17)
    env.defenders[0, :3] = env.intruders[1, :3]+np.array((0., 5., 0.))
    assert np.linalg.norm(env.defenders[0, :3]-env.intruders[1, :3]) < \
        np.linalg.norm(env.defenders[0, :3]-env.intruders[0, :3])
    _, _, _, _, info = env.step(_neutral())
    assert not info["events"]
    assert info["assignment"][0] == 0
    assert info["assignment_update_count"] == 0


def test_target_resolution_reselects_only_affected_active_pursuers(monkeypatch):
    env = TwentyVTwentyEnv()
    env.reset(17)
    env.assignment[19] = 1  # A valid assigned target other than the resolving I0.
    calls = []
    import env.ideal_interception.large_scale_20v20_env as module
    def nearest(positions, targets, defender_mask, intruder_mask):
        calls.append(np.asarray(defender_mask).copy())
        return original_nearest(positions, targets, defender_mask, intruder_mask)
    monkeypatch.setattr(module, "nearest_distance_assignment", nearest)
    _, _, _, _, info = env.step(_approach_target(env, (0,), (.5,)))
    assert len(calls) == 1
    assert calls[0].sum() == 18 and not calls[0][0] and not calls[0][19]
    assert info["assignment"][19] == 1
    update = info["assignment_updates"][0]
    assert update["reason"] == "target_resolution_nearest_reassignment"
    assert update["resolved_target"] == 0
    assert update["affected_defenders"] == list(range(1, 19))
    assert update["old_assignment"][19] == update["new_assignment"][19] == 1
    assert 0 not in update["new_assignment"]
    assert 0 not in update["new_assignment"].values()


@pytest.mark.parametrize("status", ["breached", "path_complete", "boundary_exit",
                                    "intruder_contact_exit"])
def test_each_target_resolution_status_uses_the_same_local_fallback(status):
    env = TwentyVTwentyEnv()
    env.reset(17)
    env.assignment[19] = 1
    before_defenders = env.defender_mask.copy()
    before_intruders = env.intruder_mask.copy()
    env.intruder_mask[0] = False
    env.intruder_status[0] = status
    env.intruders[0, 3:] = 0.
    env.update_assignments_after_member_event(before_defenders, before_intruders)
    assert env.assignment[19] == 1
    assert 0 not in env.assignment.values()
    assert env.assignment_history[-1]["affected_defenders"] == list(range(19))
    assert env.assignment_history[-1]["reason"] == \
        "target_resolution_nearest_reassignment"


def test_defender_only_exit_removes_its_assignment_without_reselecting(monkeypatch):
    env = TwentyVTwentyEnv()
    env.reset(17)
    env.defenders[0, :3] = (299.5, 200., 60.)
    import env.ideal_interception.large_scale_20v20_env as module
    def forbidden(*args):
        raise AssertionError("defender-only exit must not rerun nearest assignment")
    monkeypatch.setattr(module, "nearest_distance_assignment", forbidden)
    _, _, _, _, info = env.step(_neutral())
    assert not info["defender_mask"][0]
    assert set(info["assignment"]) == set(range(1, 20))
    assert set(info["assignment"].values()) == {0}
    assert info["assignment_updates"][0]["reason"] == "defender_exit_assignment_removed"
    assert info["target_resolution_reassignment_count"] == 0


def test_unassigned_physical_contact_is_not_assigned_interception():
    env = TwentyVTwentyEnv()
    env.reset(17)
    env.assignment[0] = 1
    _, _, _, _, info = env.step(_approach_target(env, (0,), (.5,)))
    contacts = [event for event in info["events"] if event["kind"] == "unassigned_contact"]
    assert len(contacts) == 1 and contacts[0]["members"] == [0, 20]
    assert info["intercepted_count"] == 0 and info["interception_ratio"] == 0.
    assert info["unassigned_contact_count"] == 1
    assert not info["defender_mask"][0] and not info["intruder_mask"][0]
    assert info["intruder_status"][0] == "unassigned_contact_exit"
    assert 0 not in info["assignment"] and 0 not in info["assignment"].values()


def test_full_episode_trace_preserves_prediction_and_assignment_history():
    summary, trace = run_episode(17)
    assert summary["avoidance_mode"] == "none"
    assert trace["trace_version"] == TRACE_VERSION
    assert trace["environment_protocol"] == PROTOCOL_20V20
    assert len(trace["references"]) == 20
    assert all(row["specification"]["mode"] == "bezier" and
               row["specification"]["speed"] == 20. for row in trace["references"])
    assert summary["initial_unique_target_count"] == 1
    assert summary["initial_max_defenders_per_target"] == 20
    assert summary["target_resolution_reassignment_count"] >= 0
    assert summary["total_reassigned_defenders"] >= 0
    assert len(trace["assignment_history"]) == summary["assignment_update_count"]+1
    for update in trace["assignment_history"][1:]:
        if update["reason"] != "target_resolution_nearest_reassignment":
            continue
        affected = set(update["affected_defenders"])
        assert all(update["old_assignment"][i] == update["resolved_target"]
                   for i in affected)
        assert all(update["old_assignment"][i] == update["new_assignment"][i]
                   for i in set(update["old_assignment"]) & set(update["new_assignment"])-affected)
    assert sum(len(frame["assignment_updates"]) for frame in trace["frames"][1:]) == \
        summary["assignment_update_count"]
    assert len(trace["frames"][0]["defenders"]) == 20
    assert len(trace["frames"][0]["intruders"]) == 20
    assert trace["frames"][-1]["outcome"] != "ongoing"
    assert summary["intercepted_count"]+summary["breached_count"]+\
        summary["remaining_intruder_count"] <= 20
    pair = trace["frames"][3]["control"]["pairs"][0]
    j = pair["assigned_target"]
    positions = [np.asarray(trace["frames"][k]["intruders"][j][:3]) for k in range(3)]
    expected = (3*positions[2]-4*positions[1]+positions[0])/(2*.05)
    np.testing.assert_allclose(pair["estimated_target_velocity_mps"], expected, rtol=1e-8)
    assert "prediction_horizon_s" in pair and "predicted_target_position_m" in pair
    json.dumps(trace, allow_nan=False)


def test_benchmark_aggregates_without_running_fifty_episodes(monkeypatch):
    def fake_run(seed, avoidance):
        return dict(interception_ratio=.5, breach_ratio=.25, mission_success=False,
            intercepted_count=10, breached_count=5, friendly_collision_count=1,
            unassigned_contact_count=2,
            initial_unique_target_count=1, initial_max_defenders_per_target=20,
            assignment_update_count=3, target_resolution_reassignment_count=2,
            total_reassigned_defenders=4,
            minimum_defender_defender_separation_m=2.), {}
    monkeypatch.setattr(benchmark_module, "run_episode", fake_run)
    result = benchmark_module.benchmark(17, 2)
    assert len(result["per_seed"]) == 2
    assert result["aggregate"]["mean_ISR"] == .5
    assert result["aggregate"]["std_ISR"] == 0.
    assert result["aggregate"]["mean_BR"] == .25
    assert result["aggregate"]["MSR"] == 0.
    assert result["aggregate"]["friendly_collision_episode_rate"] == 1.
    assert result["aggregate"]["mean_target_resolution_reassignment_count"] == 2.
    assert result["aggregate"]["mean_total_reassigned_defenders"] == 4.
