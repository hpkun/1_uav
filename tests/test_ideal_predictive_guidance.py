from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from algorithm.ideal_predictive_guidance import run_episode,canonical_1v1_config
from env.ideal_interception import InterceptionConfig
from env.ideal_interception.config import DEFENDER_SLOTS,INTRUDER_SLOTS
from tools.ideal_contact_forecast_eval import evaluate
from tools.ideal_interception_timeline import Timeline
from tools.ideal_interception_trace import record,write_trace


@pytest.mark.parametrize("method,active_mode,warmup",[
    ("two_point_pip","TWO_POINT_PIP",1),
    ("three_point_pip","THREE_POINT_PIP",2)])
def test_straight_1v1_guidance_trace_and_actions(tmp_path,method,active_mode,warmup):
    summary,trace=run_episode(canonical_1v1_config(),method,17)
    assert summary["steps"]<=600 and summary["steps"]==len(trace["frames"])-1
    assert summary["warmup_steps"]==warmup and summary["predictive_steps"]>0
    assert summary["fallback_steps"]>=0 and summary["minimum_separation_m"]>=0
    assert summary["min_prediction_horizon_s"]<=summary["max_prediction_horizon_s"]
    assert summary["initial_range_m"]>summary["minimum_range_m"]
    assert all("offline_evaluation" in frame for frame in trace["frames"][1:])
    assert trace["trace_version"]=="uav-ideal-trace-v2"
    times=[frame["time"] for frame in trace["frames"]]
    assert all(b>a for a,b in zip(times,times[1:]))
    for before,after in zip(trace["frames"][:-1],trace["frames"][1:]):
        control=after["control"]
        assert control["decision_time_s"]==before["time"]
        assert control["assigned_target"]==0
        assert control["guidance_mode"] in ("WARMUP_PP",active_mode,"FALLBACK_PP")
        action=np.asarray(control["action"])
        assert action.shape==(DEFENDER_SLOTS,3) and np.isfinite(action).all()
        assert ((0<=action)&(action<=1)).all()
        np.testing.assert_allclose(action[1:],.5)
        assert np.linalg.norm(control["guidance_direction_unit"])==pytest.approx(1.)
        assert "assignment" in after
        assert all(after["intruder_mask"][int(j)] for j in after["assignment"].values())
    path=tmp_path/"episode.json"
    write_trace(path,trace)
    timeline=Timeline(path)
    assert timeline.duration==summary["time_s"]
    tampered=deepcopy(trace)
    tampered["frames"][1]["control"]["decision_time_s"]+=.01
    bad=tmp_path/"bad_control.json"
    write_trace(bad,tampered)
    with pytest.raises(ValueError,match="step-start control"):
        Timeline(bad)
    with pytest.raises(ValueError,match="1v1"):
        run_episode(InterceptionConfig(scale="3v1"),method,17)


def test_canonical_geometry_and_oracle_is_evaluation_only(monkeypatch):
    from env.ideal_interception.scenario import initialize,diagnostics
    from prediction.intercept_prediction import solve_constant_velocity_pip
    import algorithm.ideal_predictive_guidance as guidance
    config=canonical_1v1_config()
    defenders,intruders,paths,_,_,_=initialize(config,17)
    assert tuple(defenders[0,:3])==(-100.,-30.,50.)
    assert tuple(intruders[0,:3])==(100.,0.,70.)
    assert paths[0].specification()["end"]==[-220.,0.,70.]
    report=diagnostics(config,paths)
    assert report["scenario_validity"]["region_crossing"]
    initial=solve_constant_velocity_pip(defenders[0,:3],20.,intruders[0,:3],
                                        paths[0].sample(0.)[1])
    assert initial.time_to_pip_s==pytest.approx(5.1625)
    assert initial.predicted_interception_position_m==pytest.approx((-3.25,0,70))
    assert report["paths"][0]["protected_region_entry_time"]>initial.time_to_pip_s
    bezier=canonical_1v1_config("bezier")
    assert diagnostics(bezier,initialize(bezier,17)[2])["scenario_validity"]["region_crossing"]
    original=guidance.analyze_trace
    def post_run_only(trace, config):
        assert trace["result"]["outcome"]!="ongoing"
        return original(trace,config)
    monkeypatch.setattr(guidance,"analyze_trace",post_run_only)
    summary,trace=run_episode(config,"three_point_pip",17)
    assert summary["oracle_scored_steps"]>0
    assert trace["frames"][3]["control"]["guidance_mode"]=="THREE_POINT_PIP"
    assert "oracle_horizon_s" not in trace["frames"][3]["control"]
    assert "oracle_horizon_s" in trace["frames"][3]["offline_evaluation"]


def test_bezier_future_truth_is_post_episode_only(monkeypatch):
    import algorithm.ideal_predictive_guidance as guidance
    original=guidance.analyze_trace
    def post_run_only(trace, config):
        assert trace["result"]["outcome"]!="ongoing"
        return original(trace,config)
    monkeypatch.setattr(guidance,"analyze_trace",post_run_only)
    summary,trace=run_episode(canonical_1v1_config("bezier"),"three_point_pip",17)
    assert summary["future_truth_scored_steps"]>0
    assert summary["oracle_scored_steps"]==0
    assert any("future_position_error_m" in frame["offline_evaluation"]
               for frame in trace["frames"][1:])
    assert all("future_position_error_m" not in frame["control"]
               for frame in trace["frames"][1:])


def test_no_positive_pip_root_falls_back_to_current_target():
    base=canonical_1v1_config()
    config=replace(base,defender_starts=((0.,-30.,50.),)+base.defender_starts[1:],
        intruder_starts=((-179.7,0.,70.),)+base.intruder_starts[1:])
    summary,trace=run_episode(config,"three_point_pip",17)
    assert summary["no_positive_pip_solution_steps"]>0
    fallback=[frame["control"] for frame in trace["frames"][1:]
              if frame["control"]["prediction_status"]=="NO_POSITIVE_PIP_SOLUTION"]
    assert fallback
    assert all(row["guidance_mode"]=="FALLBACK_PP" and
               row["fallback_reason"]=="NO_POSITIVE_PIP_SOLUTION" and
               row["predicted_target_position_m"] is None for row in fallback)


def test_two_point_retries_after_no_root_and_straight_estimators_agree():
    base=canonical_1v1_config()
    config=replace(base,defender_starts=((0.,-30.,50.),)+base.defender_starts[1:],
        intruder_starts=((-179.7,0.,70.),)+base.intruder_starts[1:])
    summary,trace=run_episode(config,"two_point_pip",17)
    assert summary["warmup_steps"]==1
    assert summary["no_positive_pip_solution_steps"]>=2
    for frame in trace["frames"][2:4]:
        control=frame["control"]
        assert control["guidance_mode"]=="FALLBACK_PP"
        assert control["fallback_reason"]=="NO_POSITIVE_PIP_SOLUTION"
        assert control["decision_time_s"]==pytest.approx(trace["frames"][frame["step"]-1]["time"])
    canonical,episode=run_episode(base,"two_point_pip",17)
    assert canonical["outcome"]=="all_intercepted"
    assert canonical["pip_estimator_comparison_steps"]>0
    assert canonical["mean_velocity_estimate_difference_mps"]<1e-10
    assert canonical["mean_pip_horizon_difference_s"]<1e-9
    assert canonical["mean_pip_point_difference_m"]<1e-9
    assert canonical["absolute_horizon_error_s_max"]<1e-9
    assert canonical["predicted_point_error_m_max"]<1e-8
    assert all("future_position_error_m" not in frame["control"] for frame in episode["frames"][1:])


def test_formal_two_method_pairing():
    config=InterceptionConfig(scale="1v1")
    results={method:run_episode(config,method,17) for method in
             ("two_point_pip","three_point_pip")}
    summaries={method:result[0] for method,result in results.items()}
    assert len({row["intruder_path_sha256"] for row in summaries.values()})==1
    assert summaries["two_point_pip"]["warmup_steps"]==1
    assert summaries["three_point_pip"]["warmup_steps"]==2
    assert summaries["two_point_pip"]["pip_estimator_comparison_steps"]>0


def test_v2_records_runtime_assignment_and_v1_read_policy(tmp_path):
    base=InterceptionConfig(scale="5v2",max_steps=2,intruder_mode="straight")
    config=replace(base,defender_starts=((-0.75,0.,60.),)+base.defender_starts[1:],
        intruder_starts=((0.75,0.,60.),)+base.intruder_starts[1:])
    trace=record(config,17)
    assert trace["frames"][0]["assignment"][0]==0
    assert trace["frames"][1]["assignment"]=={1:1,2:1,3:1,4:1}
    path=tmp_path/"v2.json"
    write_trace(path,trace)
    assert Timeline(path).data["frames"][1]["assignment"]=={"1":1,"2":1,"3":1,"4":1}
    assert evaluate(path)["assignment_source"]=="recorded_frame"
    tampered=deepcopy(trace)
    tampered["frames"][0]["assignment"][0]=1
    bad=tmp_path/"bad_assignment.json"
    write_trace(bad,tampered)
    with pytest.raises(ValueError,match="assignment mismatch"):
        Timeline(bad)
    old=deepcopy(trace)
    old["trace_version"]="uav-ideal-trace-v1"
    for frame in old["frames"]:
        frame.pop("assignment",None)
    legacy=tmp_path/"v1_v3.json"
    write_trace(legacy,old)
    Timeline(legacy)  # Geometry-only playback remains available.
    with pytest.raises(ValueError,match="lacks exact runtime assignment"):
        evaluate(legacy)
    old["environment_protocol"]="uav-ideal-interception-v2.0.0"
    for key in ("defender_speeds","defender_starts","defender_directions"):
        old["config"][key]=old["config"][key][:5]
    for key in ("intruder_speeds","intruder_starts","intruder_ends"):
        old["config"][key]=old["config"][key][:2]
    for frame in old["frames"]:
        for key in ("defenders","defender_mask","executed_velocities"):
            if key in frame:
                frame[key]=frame[key][:5]
        for key in ("intruders","intruder_mask"):
            frame[key]=frame[key][:2]
    for event in [e for frame in old["frames"] for e in frame["events"]]+old["result"]["events"]:
        event["members"]=[member-1 if member>=6 else member for member in event["members"]]
    legacy_v2=tmp_path/"v1_v2.json"
    write_trace(legacy_v2,old)
    assert evaluate(legacy_v2)["assignment_source"]=="fixed_scenario_rule"
