"""V3.4 motion profiles and historical trace isolation."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from algorithm.ideal_multi_predictive_guidance import METHODS,run_episode
from algorithm.ideal_interception import summarize
from env.ideal_interception.config import (PROTOCOL,DEFENDER_SLOTS,formal_6v6_config,
    formal_6v6_level2_config)
from env.ideal_interception.environment import InterceptionEnv
from env.ideal_interception.scenario import initialize,diagnostics
from tools.ideal_interception_timeline import Timeline
from tools.ideal_interception_trace import write_trace


LEVEL1_SEED17_HASHES=(
    "b1da67330e30dd3e1a34a9964ad308483dbc3c56bbfb39182358d799001b5784",
    "18b2a03ec2f842f994853b0fde6c49737123876d57bc7eba02e3cb84b64c47c7",
    "76cd431bac9fea18e389cd431125b176fa446284074015c74c40bf1469e20d8f",
    "bc26415344362f53e29ff88063cd31601aee274834dba030fcde7e0647a0512c",
    "2695d456e6577cde1d11040fb8005d87aa2f0c6c87420e3999f40a5870aaf2dc",
    "08e3273a6d13756bc2336d67f2cb8144fe94e6c455397e0a982f6f1d9f9f9add")


def test_method_namespace_and_level1_seed17_fingerprint():
    assert PROTOCOL=="uav-ideal-interception-v3.5.0"
    assert METHODS==("two_point_pip","three_point_pip")
    c=formal_6v6_config()
    defenders,intruders,paths,assignment,_,scenario_id=initialize(c,17)
    assert tuple(p.specification()["polyline_sha256"] for p in paths)==LEVEL1_SEED17_HASHES
    assert scenario_id=="faae7e6da80520609ddcf063de79fefe4e09895c6ba24680452f3d30a0f57c5e"
    assert len(assignment)==6
    with pytest.raises(ValueError,match="level2 requires"):
        formal_6v6_level2_config(intruder_mode="straight")
    with pytest.raises(ValueError,match="level2 requires"):
        formal_6v6_level2_config(intruder_goal_altitude_mode="asset_z")


def test_level2_100_seed_geometry_speed_and_determinism():
    c=formal_6v6_level2_config()
    smooth=formal_6v6_config()
    hashes=set()
    for seed in range(100):
        defenders,intruders,paths,assignment,_,scenario=initialize(c,seed)
        _,smooth_intruders,_,_,_,_=initialize(smooth,seed)
        np.testing.assert_array_equal(intruders[:6,:3],smooth_intruders[:6,:3])
        report=diagnostics(c,paths)
        assert report["scenario_validity"]["region_crossing"]
        positions=np.r_[defenders[:6,:3],intruders[:6,:3]]
        assert all(np.linalg.norm(positions[i]-positions[j])>c.contact_distance
            for i in range(12) for j in range(i+1,12))
        for j,(path,row) in enumerate(zip(paths,report["paths"])):
            assert np.isfinite(path.points).all() and np.all(path.lengths>0)
            assert row["within_bounds"] and row["entry_within_episode"]
            assert row["maximum_chord_deviation_m"]>=8
            assert row["maximum_heading_deviation_from_chord_deg"]>=8
            start,goal=path.points[0],path.points[-1]
            length=np.linalg.norm(goal-start)
            parallel=(goal-start)/length
            lateral=np.array((-parallel[1],parallel[0],0.))
            lateral/=np.linalg.norm(lateral)
            vertical=np.cross(parallel,lateral)
            first,second=map(np.asarray,path.controls)
            offset1=first-start-(goal-start)/3
            offset2=second-start-2*(goal-start)/3
            a1=float(offset1@lateral)/length
            a2=float(offset2@lateral)/length
            assert a1*a2<0
            assert .08-1e-12<=abs(a1)<=.14+1e-12
            assert .08-1e-12<=abs(a2)<=.14+1e-12
            assert abs(float(offset1@vertical)/length)<=.04+1e-12
            assert float(offset1@vertical)==pytest.approx(-float(offset2@vertical),abs=1e-10)
            for t in (0.,.17,path.duration/3,path.duration*.81):
                assert np.linalg.norm(path.sample(t)[1])==pytest.approx(c.intruder_speeds[j],abs=1e-9)
            hashes.add(path.specification()["polyline_sha256"])
        if seed==17:
            again=initialize(c,seed)
            assert scenario==again[-1]
            assert all(np.array_equal(a.points,b.points) for a,b in zip(paths,again[2]))
    assert len(hashes)==600


def test_level2_guidance_pairing_and_offline_only_truth():
    c=formal_6v6_level2_config()
    results={method:run_episode(c,method,17) for method in METHODS}
    summaries=[item[0] for item in results.values()]
    assert len({row["scenario_id"] for row in summaries})==1
    assert len({tuple(row["path_hashes"]) for row in summaries})==1
    assert all(row["same_state_pip"]["samples"]>0 for row in summaries)
    for summary,trace in results.values():
        assert summary["same_state_pip"]["two_point_velocity_error_mps"]["mean"] is not None
        for frame in trace["frames"][1:]:
            for pair in frame["control"]["pairs"].values():
                assert "two_point_velocity_error_mps" not in pair
                assert "true_local_acceleration_mps2" not in pair
    for action in (np.zeros((DEFENDER_SLOTS,3)),np.ones((DEFENDER_SLOTS,3))):
        env=InterceptionEnv(c)
        _,info=env.reset(17)
        original=deepcopy(info["intruder_plans"])
        original_paths=[path.specification() for path in env.paths]
        env.step(action)
        assert [path.plan_metadata for path in env.paths]==original
        assert [path.specification() for path in env.paths]==original_paths


def test_two_point_rename_preserves_v33_seed17_physical_values():
    summary,trace=run_episode(formal_6v6_config(),"two_point_pip",17)
    assert summary["episode_resolution_time_s"]==9.975873694036997
    assert summary["first_interception_time_s"]==6.589991349016294
    assert summary["last_interception_time_s"]==9.975873694036997
    pair=trace["frames"][2]["control"]["pairs"][0]
    assert pair["prediction_horizon_s"]==6.55355897940144
    assert pair["predicted_target_position_m"]==[
        61.78871711055476,1.8473658710035514,51.60678565993477]
    assert pair["estimated_target_velocity_mps"]==[
        -19.938041172664498,.7931533549787773,-1.3584583467346079]


def test_v33_strict_metadata_still_replays(tmp_path):
    path=Path(__file__).resolve().parents[1]/"outputs/ideal_guidance/multi6_v33_seed17_zuo_tp_verified/episode.json"
    timeline=Timeline(path)
    assert timeline.defender_slots==timeline.intruder_slots==6
    assert len(timeline.at(0)["entities"])==12


def test_mappo_6v6_summary_uses_member_counts():
    template=dict(scale="6v6",outcome="mixed_resolution",episode_return=0.,
        interception_count=3,time=10.,minimum_separation=1.,defender_exits=1,
        intruder_exits=6,initial_intruder_count=6,intercepted_count=3,
        breached_count=2,friendly_collision_count=1,defender_boundary_count=2,
        mission_success=False)
    rows=[template,template|dict(intercepted_count=6,breached_count=0,
        friendly_collision_count=0,defender_boundary_count=0,mission_success=True)]
    result=summarize(rows)
    assert result["interception_ratio"]==pytest.approx(9/12)
    assert result["breach_ratio"]==pytest.approx(2/12)
    assert result["episodes_with_breach_rate"]==.5
    assert result["episodes_with_friendly_collision_rate"]==.5
    assert result["mission_success_rate"]==.5
    assert result["total_friendly_collision_count"]==1
    assert result["total_defender_boundary_count"]==2
