from dataclasses import asdict

import numpy as np

from algorithm.common.research import atomic_json
from algorithm.interception import episode
from env.interception.config import VERSION
from tools.interception_speed_tiers import TASK_NAMES, TIERS, config, levels, reset, verify_pair
from tools.interception_speed_report import turn_evidence
from tools.interception_timeline import Timeline


def test_speed_tiers_share_geometry_and_scale_both_initial_velocities():
    for task in TASK_NAMES:
        for seed in (1200,1203,1249):
            low = reset(config("low", task), seed)
            for tier, (red, blue) in TIERS.items():
                c = config(tier, task)
                actual = reset(c, seed)
                verify_pair(low, actual, red/6)
                for side, mask, limits in (
                    ("pursuers", "active_mask", c.pursuer_initial_speed),
                    ("evaders", "evader_mask", c.evader_initial_speed),
                ):
                    speed = np.linalg.norm(actual[side][actual[mask],3:],axis=1)
                    assert min(speed) >= limits[0]-1e-12 and max(speed) <= limits[1]+1e-12
                assert (c.dt,c.max_steps,c.horizontal_acceleration_max) == (.1,900,6)
                assert c.pursuer_collision_radius+c.evader_collision_radius == 1.
                assert (c.pursuer_vmax,c.evader_vmax) == (red,blue)


def test_proximity_diagnostics_keep_geometric_bands_separate():
    assert levels(3.) == {"10":True,"5":True,"3":True,"2":False,"1":False}
    assert levels(1.00001)["1"] is False
    assert levels(1.)["1"] is True


def test_turn_measurements_show_fixed_cap_affects_higher_tiers():
    turns = turn_evidence()
    assert len(turns) == 12
    for team in ("red","blue"):
        for angle in (90,180):
            names = [f"{tier}_{team}_{angle}" for tier in TIERS]
            assert turns[names[0]]["half_turn_or_reverse_s"] <= turns[names[-1]]["half_turn_or_reverse_s"]
            assert turns[names[-1]]["max_acceleration"] <= 6+1e-12


def test_failed_approach_replay_adds_only_a_diagnostic_marker(tmp_path):
    c = config("upper", "S2")
    result, frames = episode(c, 1200, {"scenario":"random"}, "predictive", trace=True, audit=True)
    closest, = result["closest_targets"]
    assert result["outcome"] == "breach" and not result["contact_events"]
    path = tmp_path/"trace.json"
    atomic_json(path, dict(version=VERSION, config=asdict(c), frames=frames, result=result,
                           diagnostic_events=[closest]))
    t = Timeline(path)
    marker = next(m for m in t.display_events if m["kind"]=="closest_approach")
    assert marker["time"] in t.key_times and t.at(marker["time"])["outcome"] == "ongoing"
    assert not t.captures
    np.testing.assert_allclose(marker["position"], np.mean(closest["positions"],axis=0))
