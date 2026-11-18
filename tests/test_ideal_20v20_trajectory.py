"""Independent 20v20 delayed-convergence geometry and replay compatibility."""
from types import SimpleNamespace

import numpy as np
import pytest

from algorithm.ideal_20v20_zuo_assignment import ZuoAssignmentTwentyEnv
from env.ideal_interception.large_scale_20v20_env import TwentyVTwentyEnv
from env.ideal_interception.large_scale_20v20_trajectory import (
    attack_frame, BETA_1, BETA_2, PROFILE)
from tools.ideal_20v20_intruder_path_audit import audit
from tools.ideal_interception_replay import show_reference_lines
from tools.ideal_interception_timeline import Timeline


def test_delayed_convergence_geometry_and_metadata():
    env = TwentyVTwentyEnv()
    _, info = env.reset(17)
    e_long, e_lat = attack_frame(env.config)
    assert np.linalg.norm(e_long) == pytest.approx(1.)
    assert np.linalg.norm(e_lat) == pytest.approx(1.)
    assert e_long@e_lat == pytest.approx(0., abs=1e-14)
    goal_xy = np.asarray(env.config.protected_center[:2])
    assert len(env.paths) == 20
    assert all(plan["trajectory_profile"] == PROFILE for plan in info["intruder_plans"])
    for path, meta in zip(env.paths, info["intruder_plans"]):
        start, goal = np.asarray(meta["start"]), np.asarray(meta["goal"])
        p1, p2 = np.asarray(meta["control_point_1"]), np.asarray(meta["control_point_2"])
        np.testing.assert_array_equal(goal[:2], goal_xy)
        assert goal[2] == start[2]
        long0 = float((start[:2]-goal_xy)@e_long)
        lane0 = float((start[:2]-goal_xy)@e_lat)
        assert meta["initial_lane_offset_m"] == pytest.approx(lane0)
        assert [float((point[:2]-goal_xy)@e_long) for point in (start,p1,p2,goal)] == \
            pytest.approx([long0,2*long0/3,long0/3,0.])
        assert meta["beta1"] == BETA_1 == .90 and meta["beta2"] == BETA_2 == .65
        assert (p1[:2]-goal_xy)@e_lat == pytest.approx(.90*lane0+meta["lateral_noise_1_m"])
        assert (p2[:2]-goal_xy)@e_lat == pytest.approx(.65*lane0+meta["lateral_noise_2_m"])
        assert all(abs(meta[f"lateral_noise_{k}_m"]) <= meta["lateral_noise_limit_m"]
                   for k in (1,2))
        assert all(abs(meta[f"vertical_noise_{k}_m"]) <= meta["vertical_noise_limit_m"]
                   for k in (1,2))
        assert path.specification()["polyline_sha256"] == meta["polyline_sha256"]
    signs = [(np.sign(m["lateral_noise_1_m"]), np.sign(m["lateral_noise_2_m"]))
             for m in info["intruder_plans"]]
    assert any(a == b for a,b in signs)  # No enforced opposite-sign S-turn.


def test_seed_reproducibility_shared_paths_and_retention():
    a = TwentyVTwentyEnv(); _, first = a.reset(17)
    b = TwentyVTwentyEnv(); _, repeat = b.reset(17)
    c = ZuoAssignmentTwentyEnv(); _, zuo = c.reset(17)
    d = TwentyVTwentyEnv(); _, other = d.reset(18)
    assert first["intruder_plans"] == repeat["intruder_plans"] == zuo["intruder_plans"]
    assert first["intruder_paths"] == repeat["intruder_paths"] == zuo["intruder_paths"]
    assert first["scenario_id"] == repeat["scenario_id"] == zuo["scenario_id"]
    np.testing.assert_array_equal(first["defenders"], zuo["defenders"])
    np.testing.assert_array_equal(first["intruders"], zuo["intruders"])
    assert first["intruder_paths"] != other["intruder_paths"]
    summary, _, _ = audit(17)
    assert summary["retention_one_third"] >= .70
    assert summary["retention_two_thirds"] >= .40
    assert summary["lane_correlation_one_third"] > .8
    assert summary["lane_correlation_two_thirds"] > .8
    assert summary["maximum_chord_deviation_range_m"][0] >= 3.
    assert summary["maximum_chord_deviation_range_m"][1] <= 80.
    assert summary["maximum_heading_deviation_range_deg"][0] >= 3.
    assert summary["maximum_heading_deviation_range_deg"][1] <= 60.


def test_reference_lines_default_only_new_protocol_is_hidden():
    new = SimpleNamespace(data={"environment_protocol":
        "uav-ideal-interception-20v20-delayed-convergence-nearest-v3"})
    old = SimpleNamespace(data={"environment_protocol":
        "uav-ideal-interception-20v20-target-resolution-nearest-v2"})
    assert not show_reference_lines(new)
    assert show_reference_lines(old)
    assert show_reference_lines(new,"on")
    assert not show_reference_lines(old,"off")
    with pytest.raises(ValueError):
        show_reference_lines(new,"invalid")


def test_old_20v20_nearest_trace_still_replays():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    source = root/"outputs/ideal_guidance/20v20_front_arc_nearest_seed17_v2/episode.json"
    if not source.is_file():
        pytest.skip("historical local trace not present")
    timeline = Timeline(source)
    assert timeline.duration > 0
    assert show_reference_lines(timeline)
