"""Zuo allocation equations and unchanged ThreePoint-PIP integration."""
import numpy as np
import pytest

from algorithm.ideal_20v20_zuo_assignment import (
    ZuoAssignmentTwentyEnv, load_balanced_assignment, run_episode)
from env.ideal_interception.assignment import nearest_distance_assignment
from env.ideal_interception.large_scale_20v20_env import TwentyVTwentyEnv


def test_published_load_penalty_spreads_equidistant_defenders():
    defenders = np.zeros((4, 3))
    targets = np.array(((10., 0., 0.), (10., 0., 0.)))
    result = load_balanced_assignment(defenders, targets, np.ones(4, bool), np.ones(2, bool))
    assert result == {0: 0, 1: 1, 2: 0, 3: 1}
    assert nearest_distance_assignment(defenders, targets, np.ones(4, bool),
                                       np.ones(2, bool)) == {0: 0, 1: 0, 2: 0, 3: 0}


def test_angle_threat_and_initial_load_are_used():
    defenders = np.zeros((1, 3))
    targets = np.array(((0., 10., 0.), (10., 0., 0.)))
    mask = np.ones(1, bool)
    assert load_balanced_assignment(defenders, targets, mask, np.ones(2, bool)) == {0: 1}
    assert load_balanced_assignment(defenders, targets, mask, np.ones(2, bool),
                                    initial_loads={1: 100}) == {0: 0}


def test_isolated_assignment_does_not_change_nearest_baseline():
    base = TwentyVTwentyEnv()
    _, baseline = base.reset(17)
    zuo = ZuoAssignmentTwentyEnv()
    _, comparison = zuo.reset(17)
    np.testing.assert_array_equal(baseline["defenders"], comparison["defenders"])
    np.testing.assert_array_equal(baseline["intruders"], comparison["intruders"])
    assert baseline["scenario_id"] == comparison["scenario_id"]
    assert baseline["intruder_paths"] == comparison["intruder_paths"]
    assert baseline["assignment"] != comparison["assignment"]
    assert base.assignment == baseline["assignment"]
    assert comparison["initial_unique_target_count"] > 1


def test_resolved_target_only_reassigns_its_active_pursuers():
    env = ZuoAssignmentTwentyEnv()
    env.reset(17)
    original = env.assignment.copy()
    target = original[0]
    affected = {i for i, j in original.items() if j == target}
    before_defenders = env.defender_mask.copy()
    before_intruders = env.intruder_mask.copy()
    env.intruder_mask[target] = False
    env.intruder_status[target] = "breached"
    env.intruders[target, 3:] = 0.
    env.update_assignments_after_member_event(before_defenders, before_intruders)
    assert set(env.assignment) == set(original)
    assert target not in env.assignment.values()
    assert all(env.assignment[i] == original[i] for i in set(original)-affected)
    update = env.assignment_history[-1]
    assert update["reason"] == "target_resolution_zuo_reassignment"
    assert set(update["affected_defenders"]) == affected


def test_full_episode_uses_three_point_pip_and_predictive_avoidance():
    summary, trace = run_episode(17)
    assert summary["guidance_method"] == "three_point_pip"
    assert summary["interception_ratio"] == pytest.approx(summary["intercepted_count"]/20)
    assert trace["frames"][-1]["outcome"] != "ongoing"
    controls = [frame["control"] for frame in trace["frames"][1:]]
    assert all(control["method"] == "three_point_pip" and
               control["avoidance_mode"] == "predictive_cpa_knn" for control in controls)
    active = sum(bool(pair["avoidance_active"]) for control in controls
                 for pair in control["pairs"].values())
    assert active == summary["avoidance_active_decisions"]
    assert active > 0
    modes = {pair["guidance_mode"] for control in controls for pair in control["pairs"].values()}
    assert "THREE_POINT_PIP" in modes
    assert "ZUO_LINEAR_PREDICTION" not in modes
