"""Reset-time intruder plan invariants, independent of learned defenders."""
from dataclasses import replace

import numpy as np
import pytest

from algorithm.ideal_predictive_guidance import run_episode, main as guidance_main
from algorithm.ideal_interception import evaluation_protocol
from env.ideal_interception import InterceptionConfig, InterceptionEnv
from env.ideal_interception.config import PROTOCOL, zuo_paper_profile,DEFENDER_SLOTS
from tools.ideal_interception_timeline import Timeline
from tools.ideal_interception_trace import record, write_trace


@pytest.mark.parametrize("scale",["1v1","3v1","5v2"])
def test_formal_goal_region_crossing_and_speed(scale):
    config=InterceptionConfig(scale=scale)
    env=InterceptionEnv(config)
    _,info=env.reset(17)
    assert config.intruder_policy=="zuo_bezier" and config.intruder_mode is None
    assert len(info["intruder_plans"])==config.intruder_count
    assert all(row["path_class"]=="region_crossing" and row["entry_within_episode"]
               for row in info["scene_diagnostics"]["paths"])
    for j,(plan,path) in enumerate(zip(info["intruder_plans"],env.paths)):
        assert plan["policy"]=="zuo_bezier"
        assert plan["goal"][:2]==config.protected_center[:2]
        assert plan["goal"][2]==pytest.approx(plan["start"][2])
        assert path.specification()["end"]==list(plan["goal"])
        assert plan["altitude_change_m"]==pytest.approx(0.)
        assert config.protected_center[2]<=plan["goal"][2]<=config.protected_center[2]+config.protected_height
        assert path.specification()["controls"]==[list(p) for p in plan["control_points"]]
        assert path.specification()["polyline_sha256"]==plan["polyline_sha256"]
        for time in (.123,.789,path.duration/2):
            assert np.linalg.norm(path.sample(time)[1])==pytest.approx(config.intruder_speeds[j])


def test_seeded_start_controls_hash_and_scenario_id():
    config=InterceptionConfig(scale="5v2")
    a=InterceptionEnv(config).reset(17)[1]
    b=InterceptionEnv(config).reset(17)[1]
    c=InterceptionEnv(config).reset(18)[1]
    assert a["intruder_plans"]==b["intruder_plans"]
    assert a["intruder_paths"]==b["intruder_paths"]
    assert a["scenario_id"]==b["scenario_id"]
    assert a["scenario_id"]!=c["scenario_id"]
    assert a["intruder_plans"][0]["start"]!=c["intruder_plans"][0]["start"]
    assert a["intruder_plans"][0]["control_points"]!=c["intruder_plans"][0]["control_points"]
    assert a["intruder_plans"][0]["polyline_sha256"]!=a["intruder_plans"][1]["polyline_sha256"]
    starts=np.asarray([plan["start"] for plan in a["intruder_plans"]])
    assert starts[0,1]<0<starts[1,1]  # Two-anchor, small-scale wedge approximation.
    assert a["intruder_plans"][0]["goal"][:2]==a["intruder_plans"][1]["goal"][:2]


def test_100_seed_bounds_and_rejection_limit():
    config=InterceptionConfig(scale="5v2")
    hashes=set()
    for seed in range(100):
        env=InterceptionEnv(config)
        _,info=env.reset(seed)
        assert info["scene_diagnostics"]["scenario_validity"]["region_crossing"]
        for path in env.paths:
            assert np.all(path.points>=config.lower) and np.all(path.points<=config.upper)
            assert np.all(path.lengths>0)
            hashes.add(path.specification()["polyline_sha256"])
        start=np.r_[info["defenders"][info["defender_mask"],:3],
                    info["intruders"][info["intruder_mask"],:3]]
        assert all(np.linalg.norm(start[i]-start[j])>config.contact_distance
                   for i in range(len(start)) for j in range(i+1,len(start)))
    assert len(hashes)==200
    impossible=replace(config,max_steps=1,init_max_attempts=2)
    with pytest.raises(ValueError,match="initialization failed after 2 attempts"):
        InterceptionEnv(impossible).reset(17)


def test_preserved_altitude_and_asset_z_diagnostic_have_distinct_paths():
    config=InterceptionConfig()
    old=replace(config,intruder_goal_altitude_mode="asset_z")
    now=InterceptionEnv(config).reset(17)[1]
    previous=InterceptionEnv(old).reset(17)[1]
    new_plan=now["intruder_plans"][0]
    old_plan=previous["intruder_plans"][0]
    assert new_plan["start"]==old_plan["start"]
    assert new_plan["goal"][:2]==old_plan["goal"][:2]==config.protected_center[:2]
    assert new_plan["goal"][2]==pytest.approx(new_plan["start"][2])
    assert old_plan["goal"][2]==config.protected_center[2]
    assert new_plan["polyline_sha256"]!=old_plan["polyline_sha256"]
    assert now["scenario_id"]!=previous["scenario_id"]
    assert now["scene_diagnostics"]["scenario_validity"]["region_crossing"]
    assert previous["scene_diagnostics"]["scenario_validity"]["region_crossing"]


def test_reference_is_defender_independent_and_never_replanned():
    config=InterceptionConfig()
    two,two_trace=run_episode(config,"two_point_pip",17)
    pip,pip_trace=run_episode(config,"three_point_pip",17)
    assert two["intruder_path_sha256"]==pip["intruder_path_sha256"]
    assert two_trace["references"]==pip_trace["references"]
    assert two_trace["scenario_id"]==pip_trace["scenario_id"]
    env=InterceptionEnv(config)
    env.reset(17)
    before=env.paths[0].specification()
    for action in (np.zeros((DEFENDER_SLOTS,3)),np.ones((DEFENDER_SLOTS,3)),
                   np.full((DEFENDER_SLOTS,3),.5)):
        if env.outcome!="ongoing":
            break
        env.step(action)
        assert env.paths[0].specification()==before


def test_unopposed_target_breaches_before_path_completion():
    base=InterceptionConfig()
    config=replace(base,defender_starts=((-280.,-280.,10.),)+base.defender_starts[1:])
    env=InterceptionEnv(config)
    env.reset(17)
    while env.outcome=="ongoing":
        _,_,_,_,info=env.step(np.full((DEFENDER_SLOTS,3),.5))
    assert info["outcome"]=="breach"
    assert any(event["kind"]=="region_breach" for event in info["events"])
    assert all(event["kind"] not in ("path_complete","boundary_exit")
               for event in env.events)


@pytest.mark.parametrize("historical_protocol",[
    "uav-ideal-interception-v3.0.0","uav-ideal-interception-v3.1.0",
    "uav-ideal-interception-v3.2.0"])
def test_paper_profile_and_historical_v3_trace(tmp_path,historical_protocol):
    default=InterceptionConfig()
    paper=zuo_paper_profile()
    assert (default.dt,default.defender_speeds[0],default.intruder_speeds[0])==(.05,20.,20.)
    assert (paper.dt,paper.defender_speeds[0],paper.intruder_speeds[0],
            paper.contact_distance)==(.1,25.,30.,.1)
    assert PROTOCOL=="uav-ideal-interception-v3.5.0"
    protocol=evaluation_protocol(default,[{"scenario_id":"a"},{"scenario_id":"b"}])
    assert protocol["kind"]=="randomized_intruder_plans"
    assert protocol["unique_scenarios"]==2
    assert "intruder_bezier_control_points" in protocol["randomized"]
    legacy=InterceptionConfig(intruder_mode="straight",max_steps=2)
    trace=record(legacy,17)
    trace["environment_protocol"]=historical_protocol
    for key in ("defender_speeds","defender_starts","defender_directions"):
        trace["config"][key]=trace["config"][key][:5]
    for key in ("intruder_speeds","intruder_starts","intruder_ends"):
        trace["config"][key]=trace["config"][key][:2]
    for frame in trace["frames"]:
        for key in ("defenders","defender_mask","executed_velocities"):
            if key in frame:
                frame[key]=frame[key][:5]
        for key in ("intruders","intruder_mask"):
            frame[key]=frame[key][:2]
    for key in ("intruder_policy","intruder_start_jitter_m","intruder_bezier_jitter_ratio"):
        trace["config"].pop(key)
    path=tmp_path/"historical_v3.json"
    write_trace(path,trace)
    assert Timeline(path).duration==trace["result"]["time"]
    env=InterceptionEnv(default)
    env.reset(17)
    with pytest.raises(ValueError,match="incompatible"):
        env.restore(env.snapshot()|{"protocol":"uav-ideal-interception-v3.0.0"})
    with pytest.raises(ValueError,match="incompatible"):
        env.restore(env.snapshot()|{"protocol":"uav-ideal-interception-v3.1.0"})
    with pytest.raises(SystemExit):
        guidance_main(["--method","two_point_pip","--motion-level","level1_smooth",
                       "--intruder-mode","straight","--out",str(tmp_path/"invalid")])
