from dataclasses import replace

import numpy as np
import pytest

from env.ideal_interception import InterceptionConfig, InterceptionEnv
from prediction.contact_forecast import estimate_contact
from prediction.comparison import compare_horizons
from prediction.forecast import forecast_assigned
from prediction.history import IntruderHistory
from prediction.three_point import predict_at, predict_two_steps
from prediction.zuo_linear import predict_zuo_at, predict_zuo_two_steps
from tools.ideal_contact_forecast_eval import evaluate
from tools.ideal_interception_trace import record, write_trace
from tools.ideal_blue_observations import generate


def test_arbitrary_times_and_legacy_two_step_consistency():
    line=np.array([[0.,1.,2.],[2.,1.,2.],[4.,1.,2.]])
    np.testing.assert_allclose(predict_zuo_at(line[-2:],.1,[.1,.2]),
                               predict_zuo_two_steps(line[-2:],.1),atol=1e-12)
    np.testing.assert_allclose(predict_at(line,.1,[.1,.2]),predict_two_steps(line,.1),atol=1e-12)
    np.testing.assert_allclose(predict_zuo_at(line[-2:],.1,[.05,.15])[:,0],[5.,7.])
    curved=np.array([[0.,0.,0.],[1.,0.,0.],[4.,0.,0.]])
    np.testing.assert_allclose(predict_at(curved,1.,[.5,1.5])[:,0],[6.25,12.25])
    for func, history in ((predict_zuo_at,line[-2:]),(predict_at,line)):
        with pytest.raises(ValueError):
            func(history,.1,[.2,.1])


def test_assignment_shares_target_and_respects_inactive_mask():
    env=InterceptionEnv(InterceptionConfig(scale="5v2"))
    _,info=env.reset(17)
    assignment=info["assignment"]
    history=IntruderHistory(.1)
    for k in range(3):
        history.observe(k*.1,[[k,0,0],[0,k,0]],[True,True],info["scenario_id"])
    forecasts=forecast_assigned(history,assignment,[.1,.25])
    assert len(forecasts)==4
    assert forecasts[(0,"zuo_linear")].future_times==(.30000000000000004,.45)
    assert forecasts[(1,"tan_quadratic")].history_count==3
    history.observe(.3,[[3,0,0],[0,0,0]],[True,False],info["scenario_id"])
    assert all(target==0 for target,_ in forecast_assigned(history,assignment,[.1]))


def test_continuous_contact_and_no_contact_are_conditional():
    history=IntruderHistory(1.)
    for k in range(3):
        history.observe(float(k),[[0,0,0],[0,0,0]],[True,False],"scene")
    target=forecast_assigned(history,{0:0},[1.])[(0,"zuo_linear")]
    hit=estimate_contact(0,0,target,[2.,3.],[[2.,0.,0.],[-2.,0.,0.]],.5)
    assert hit.contact_predicted and hit.status=="CONTACT_IN_FORECAST_WINDOW"
    assert hit.predicted_contact_time==pytest.approx(2.375)
    miss=estimate_contact(0,0,target,[2.,3.],[[2.,2.,0.],[-2.,2.,0.]],.5)
    assert not miss.contact_predicted and miss.status=="NO_CONTACT_IN_FORECAST_WINDOW"
    incomplete=estimate_contact(0,0,target,[2.,2.5],[[2.,0.,0.],[-2.,0.,0.]],.5)
    assert incomplete.status=="NOT_EVALUABLE"


def test_offline_contact_forecast_does_not_change_trace(tmp_path):
    base=InterceptionConfig(max_steps=10)
    config=replace(base,defender_starts=((-5.,0.,60.),)+base.defender_starts[1:],
        intruder_starts=((5.,0.,60.),)+base.intruder_starts[1:],
        intruder_ends=((-50.,0.,60.),)+base.intruder_ends[1:])
    trace=tmp_path/"contact.json.gz"
    original=record(config,17)
    write_trace(trace,original)
    result=evaluate(trace,2)
    assert any(e["kind"]=="interception_contact" for e in original["result"]["events"])
    assert result==evaluate(trace,2)
    assert result["methods"]["zuo_linear"]["actual_contacts"]>=1
    assert result["methods"]["tan_quadratic"]["actual_contacts"]>=1
    assert result["methods"]["zuo_linear"]["contact_detection_rate"]==1.
    assert result["methods"]["tan_quadratic"]["contact_detection_rate"]==1.
    assert original["result"]["outcome"]=="all_intercepted"


def test_extended_horizons_keep_paired_samples():
    data=generate(InterceptionConfig(intruder_mode="bezier"),17)
    result=compare_horizons(data,(1,2,5,10))
    methods=result["targets"]["I0"]
    for horizon in (1,2,5,10):
        assert (methods["zuo_linear"][str(horizon)]["ValidSamples"]
                == methods["tan_quadratic"][str(horizon)]["ValidSamples"])
    assert methods["zuo_linear"]["10"]["MAE"]>methods["zuo_linear"]["1"]["MAE"]
