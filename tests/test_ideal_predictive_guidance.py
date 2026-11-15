from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from algorithm.ideal_predictive_guidance import run_episode
from env.ideal_interception import InterceptionConfig
from tools.ideal_contact_forecast_eval import evaluate
from tools.ideal_interception_timeline import Timeline
from tools.ideal_interception_trace import record,write_trace


@pytest.mark.parametrize("method,active_mode,warmup",[
    ("zuo_tp","ZUO_TP",1),("three_point_adaptive","THREE_POINT_ADAPTIVE",2)])
def test_straight_1v1_guidance_trace_and_actions(tmp_path,method,active_mode,warmup):
    summary,trace=run_episode(InterceptionConfig(),method,17)
    assert summary["steps"]<=600 and summary["steps"]==len(trace["frames"])-1
    assert summary["warmup_steps"]==warmup and summary["predictive_steps"]>0
    assert summary["fallback_steps"]>=0 and summary["minimum_separation_m"]>=0
    assert summary["min_prediction_horizon_s"]<=summary["max_prediction_horizon_s"]
    assert trace["trace_version"]=="uav-ideal-trace-v2"
    times=[frame["time"] for frame in trace["frames"]]
    assert all(b>a for a,b in zip(times,times[1:]))
    for before,after in zip(trace["frames"][:-1],trace["frames"][1:]):
        control=after["control"]
        assert control["decision_time_s"]==before["time"]
        assert control["assigned_target"]==0
        assert control["guidance_mode"] in ("WARMUP_PP",active_mode,"FALLBACK_PP")
        action=np.asarray(control["action"])
        assert action.shape==(5,3) and np.isfinite(action).all()
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


def test_v2_records_runtime_assignment_and_v1_read_policy(tmp_path):
    base=InterceptionConfig(scale="5v2",max_steps=2)
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
    legacy_v2=tmp_path/"v1_v2.json"
    write_trace(legacy_v2,old)
    assert evaluate(legacy_v2)["assignment_source"]=="fixed_scenario_rule"
