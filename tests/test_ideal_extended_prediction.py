import csv
import json

import numpy as np
import pytest

from env.ideal_interception import InterceptionConfig
from prediction.comparison import compare_positions
from prediction.history import IntruderHistory
from prediction.forecast import forecast_assigned
from prediction.three_point import predict_at, recommended_horizon
from prediction.zuo_linear import predict_zuo_at
from tools.ideal_blue_observations import generate, write_observations
from tools.ideal_prediction_compare import run_extended


def test_analytic_gap_and_method_readiness():
    history=np.array([[0.,0.,0.],[1.,0.,0.],[4.,0.,0.]])
    dt=1.
    acceleration=history[-1]-2*history[-2]+history[-3]
    for offset in (.1,.5,2.):
        gap=np.linalg.norm(predict_at(history,dt,[offset])[0]
                           -predict_zuo_at(history[-2:],dt,[offset])[0])
        assert gap==pytest.approx(.5*np.linalg.norm(acceleration)*offset*(offset+dt))
    managed=IntruderHistory(dt)
    managed.observe(0.,[[0,0,0],[0,0,0]],[True,False],"a")
    assert managed.method_status(0,"zuo_linear")=="INSUFFICIENT_HISTORY"
    managed.observe(1.,[[1,0,0],[0,0,0]],[True,False],"a")
    assert managed.method_status(0,"zuo_linear")=="READY"
    assert managed.method_status(0,"tan_quadratic")=="INSUFFICIENT_HISTORY"
    managed.observe(2.,[[4,0,0],[0,0,0]],[True,False],"a")
    assert managed.method_status(0,"tan_quadratic")=="READY"
    forecasts=forecast_assigned(managed,{0:0,2:0},[.25,1.],
        tan_limit=dict(eta=1.,minimum=.1,maximum=.5,epsilon_acceleration=1e-6))
    assert len(forecasts)==2
    assert forecasts[(0,"tan_quadratic")].horizon_statuses==("READY","HORIZON_LIMITED")
    assert forecasts[(0,"tan_quadratic")].status=="READY"
    assert forecasts[(0,"zuo_linear")].horizon_cap_s is None
    managed.observe(3.,[[4,0,0],[0,0,0]],[False,False],"a")
    assert managed.method_status(0,"zuo_linear")=="INACTIVE"


def test_explicit_cap_and_full_truth_pairing(tmp_path):
    config=InterceptionConfig(intruder_mode="bezier")
    data=generate(config,17)
    limit=dict(eta=1.,minimum=.1,maximum=.5,epsilon_acceleration=1e-6)
    assert recommended_horizon([[0,0,0],[1,0,0],[4,0,0]],1.,**limit)==pytest.approx(.5)
    result=compare_positions(data,(1,5,10,20),limit)
    assert result==compare_positions(data,(1,5,10,20),limit)
    rows=result["rows"]
    assert {row["horizon_steps"] for row in rows}=={1,5,10,20}
    assert any(row["tan_horizon_status"]=="HORIZON_LIMITED" for row in rows)
    assert all(result["targets"]["I0"][str(h)]["valid_samples"]==len(data["targets"][0]["observations"])-2-h
               for h in (1,5,10,20))
    assert all(row["prediction_time"]<=data["targets"][0]["last_regular_sample_time"] for row in rows)
    for row in rows:
        assert row["position_gap_m"]==pytest.approx(np.linalg.norm(
            np.array([row[f"zuo_{a}"] for a in "xyz"])-
            np.array([row[f"three_point_{a}"] for a in "xyz"])))
        assert row["error_improvement_m"]==pytest.approx(row["zuo_error_m"]-row["three_point_error_m"])
    source=tmp_path/"observations.json.gz"
    write_observations(source,data)
    exported=run_extended(source,tmp_path/"report",(1,5,10,20),10,limit)
    assert exported==result
    assert {p.name for p in (tmp_path/"report").iterdir()}=={
        "comparison.csv","summary.json","trajectory_overlay.png","error_curve.png"}
    with (tmp_path/"report"/"comparison.csv").open() as stream:
        assert len(list(csv.DictReader(stream)))==len(rows)
    summary=json.loads((tmp_path/"report"/"summary.json").read_text())
    assert summary["targets"]["I0"]["20"]["valid_samples"]==len(data["targets"][0]["observations"])-22
