from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest

from env.ideal_interception import InterceptionConfig
from prediction import IntruderHistory,predict_two_steps
from prediction.evaluation import evaluate_trace
from tools.ideal_interception_trace import record,write_trace
from tools.ideal_interception_timeline import Timeline
from tools.ideal_prediction_demo import run


def test_straight_and_quadratic_predictions():
    p=np.array([[1.,2.,3.],[3.,4.,5.],[5.,6.,7.]])
    np.testing.assert_allclose(predict_two_steps(p,.1),[[7.,8.,9.],[9.,10.,11.]])
    t=np.array([0.,.1,.2,.3,.4])
    q=np.stack([2*t*t+3*t+1,-t*t+2*t,4*t*t-1],axis=1)
    np.testing.assert_allclose(predict_two_steps(q[:3],.1),q[3:],atol=1e-14)
    with pytest.raises(ValueError):
        predict_two_steps(p[::-1,:2],.1)
    with pytest.raises(ValueError):
        predict_two_steps(p,0)


def test_history_readiness_gaps_exits_and_scenario_reset():
    history=IntruderHistory(.1)
    for k in range(2):
        history.observe(k*.1,[[k,0,0],[0,k,0]],[True,True],"scene-a")
        assert history.status(0)=="NOT_READY"
        assert history.predict(0) is None
    history.observe(.2,[[2,0,0],[0,2,0]],[True,True],"scene-a")
    np.testing.assert_allclose(history.predict(0)[0],[3,0,0])
    np.testing.assert_allclose(history.predict(1)[0],[0,3,0])
    history.observe(.3,[[3,0,0],[0,2,0]],[True,False],"scene-a")
    assert history.status(1)=="INACTIVE" and history.predict(1) is None
    history.observe(.55,[[4,0,0],[0,2,0]],[True,False],"scene-a")
    assert history.status(0)=="NOT_READY"
    with pytest.raises(ValueError,match="reactivated"):
        history.observe(.65,[[5,0,0],[0,3,0]],[True,True],"scene-a")
    history.observe(.65,[[5,0,0],[0,0,0]],[True,True],"scene-b")
    assert history.status(0)=="NOT_READY" and history.status(1)=="NOT_READY"
    with pytest.raises(ValueError,match="nonincreasing"):
        history.observe(.65,[[5,0,0],[0,0,0]],[True,True],"scene-b")


def test_real_trace_straight_error_and_no_future_after_exit(tmp_path):
    config=InterceptionConfig(scale="1v1",intruder_mode="straight",max_steps=15)
    trace=tmp_path/"trace.json.gz"
    write_trace(trace,record(config,17))
    result=evaluate_trace(Timeline(trace))
    metrics=result["metrics"]["targets"]["I0"]
    assert metrics["ValidSamples_1"]>0 and metrics["ValidSamples_2"]>0
    assert metrics["MaxError_1"]<1e-9 and metrics["MaxError_2"]<1e-9
    assert any(not row["valid"] for row in result["predictions"])
    assert all(row["actual_x"] is None and row["error_m"] is None
               for row in result["predictions"] if not row["valid"])
    first=run(trace,tmp_path/"results",plot=True,threshold=.5)
    assert first==evaluate_trace(Timeline(trace),threshold=.5)
    assert (tmp_path/"results"/"predictions.csv").is_file()
    assert (tmp_path/"results"/"error_curve.png").is_file()


def test_independent_multi_target_real_trace(tmp_path):
    config=InterceptionConfig(scale="5v2",intruder_mode="bezier",initialization="random",max_steps=20)
    trace=tmp_path/"multi.json"
    write_trace(trace,record(config,17))
    a=evaluate_trace(Timeline(trace))
    b=evaluate_trace(Timeline(trace))
    assert a==b
    assert set(a["metrics"]["targets"])=={"I0","I1"}
    assert all(a["metrics"]["targets"][target]["ValidSamples_1"]>0 for target in ("I0","I1"))


def test_partial_terminal_frame_is_not_a_regular_sample(tmp_path):
    base=InterceptionConfig()
    config=replace(base,dt=1.,max_steps=3,
        defender_starts=((-5.,0.,60.),)+base.defender_starts[1:],
        intruder_starts=((5.,0.,60.),)+base.intruder_starts[1:],
        intruder_ends=((-50.,0.,60.),)+base.intruder_ends[1:])
    trace=tmp_path/"early.json"
    write_trace(trace,record(config,17))
    result=evaluate_trace(Timeline(trace))
    assert result["metrics"]["overall"]["ValidSamples_1"]==0
    assert result["metrics"]["overall"]["MAE_1"] is None
