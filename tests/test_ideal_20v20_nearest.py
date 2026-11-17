"""Independent 20v20 event-driven nearest assignment and full episode contracts."""
import json

import numpy as np
import pytest

from algorithm.ideal_20v20_nearest_guidance import run_episode, TRACE_VERSION
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
    assert len(trace["assignment_history"]) == summary["assignment_update_count"]+1
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
            initial_unique_target_count=1, initial_max_defenders_per_target=20,
            assignment_update_count=3, minimum_defender_defender_separation_m=2.), {}
    monkeypatch.setattr(benchmark_module, "run_episode", fake_run)
    result = benchmark_module.benchmark(17, 2)
    assert len(result["per_seed"]) == 2
    assert result["aggregate"]["mean_ISR"] == .5
    assert result["aggregate"]["std_ISR"] == 0.
    assert result["aggregate"]["mean_BR"] == .25
    assert result["aggregate"]["MSR"] == 0.
    assert result["aggregate"]["friendly_collision_episode_rate"] == 1.
