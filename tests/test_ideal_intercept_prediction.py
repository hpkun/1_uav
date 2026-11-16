import math
import inspect

import numpy as np
import pytest

from env.ideal_interception import InterceptionConfig,InterceptionEnv
from prediction.history import IntruderHistory
from prediction.intercept_prediction import (estimate_zuo_prediction_from_history,
    estimate_zuo_pip_time_from_history,estimate_three_point_pip_from_history,
    solve_constant_velocity_pip)


def history_for(position,velocity=(0,0,0),acceleration=(0,0,0),dt=.05):
    p,v,a=map(lambda x:np.asarray(x,float),(position,velocity,acceleration))
    history=IntruderHistory(dt)
    for k,offset in enumerate((-2*dt,-dt,0.)):
        observed=p+v*offset+.5*a*offset*offset
        history.observe(k*dt,[observed,[0.,0.,0.]],[True,False],"fixture")
    return history


def test_zuo_paper_formula_cold_start_and_inactive():
    history=IntruderHistory(.1)
    history.observe(0.,[[32,0,0],[0,0,0]],[True,False],"fixture")
    assert estimate_zuo_prediction_from_history(history,0,(0,0,0),20.).status=="INSUFFICIENT_HISTORY"
    history.observe(.1,[[30,0,0],[0,0,0]],[True,False],"fixture")
    result=estimate_zuo_prediction_from_history(history,0,(0,0,0),20.)
    assert result.status=="READY" and result.horizon_kind=="paper_lookahead"
    assert result.horizon_s==pytest.approx(1.875)
    assert result.predicted_target_position_m==pytest.approx((-7.5,0,0))
    assert result.paper_direction_unit==pytest.approx((-1,0,0))
    history.observe(.2,[[30,0,0],[0,0,0]],[False,False],"fixture")
    assert estimate_zuo_prediction_from_history(history,0,(0,0,0),20.).status=="INACTIVE"


def test_zuo_pip_time_uses_two_point_velocity_and_shared_solver(monkeypatch):
    import prediction.intercept_prediction as module
    history=IntruderHistory(.05)
    history.observe(0.,[[100,0,70],[0,0,0]],[True,False],"fixture")
    assert estimate_zuo_pip_time_from_history(history,0,(-100,-30,50),20.).status=="INSUFFICIENT_HISTORY"
    history.observe(.05,[[99,0,70],[0,0,0]],[True,False],"fixture")
    strict=estimate_zuo_prediction_from_history(history,0,(-100,-30,50),20.)
    solver=module.solve_constant_velocity_pip
    calls=[]
    def checked(*args):
        calls.append(args)
        return solver(*args)
    monkeypatch.setattr(module,"solve_constant_velocity_pip",checked)
    monkeypatch.setattr(module,"zuo_lookahead_time",lambda *args: (_ for _ in ()).throw(
        AssertionError("Eq.32 must not run in the ablation")))
    hybrid=estimate_zuo_pip_time_from_history(history,0,(-100,-30,50),20.)
    assert len(calls)==1 and hybrid.status=="READY"
    assert hybrid.history_count==2 and hybrid.estimated_target_acceleration_mps2 is None
    assert hybrid.estimated_target_velocity_mps==strict.estimated_target_velocity_mps
    assert hybrid.horizon_kind=="constant_velocity_pip_with_zuo_velocity"
    assert hybrid.predicted_target_position_m==pytest.approx(
        np.asarray((99,0,70))+np.asarray(hybrid.estimated_target_velocity_mps)*hybrid.horizon_s)
    assert hybrid.horizon_s<strict.horizon_s


def test_zuo_pip_time_no_positive_root_keeps_velocity_diagnostic():
    history=history_for((10,0,0),(20,0,0))
    result=estimate_zuo_pip_time_from_history(history,0,(0,0,0),20.)
    assert result.status=="NO_POSITIVE_PIP_SOLUTION"
    assert result.predicted_target_position_m is None and result.horizon_s is None
    assert result.estimated_target_velocity_mps==pytest.approx((20,0,0))


def test_pip_quadratic_3d_geometry_and_earliest_positive_root():
    result=solve_constant_velocity_pip((0,0,0),20.,(30,40,10),(-2,3,1))
    assert result.status=="READY" and result.equation_degree==2
    t=result.time_to_pip_s
    point=np.asarray(result.predicted_interception_position_m)
    np.testing.assert_allclose(point,[30-2*t,40+3*t,10+t])
    assert np.linalg.norm(point)==pytest.approx(20*t)
    two=solve_constant_velocity_pip((0,0,0),20.,(100,0,0),(-30,0,0))
    assert two.status=="READY" and two.time_to_pip_s==pytest.approx(2.)
    assert two.predicted_interception_position_m==pytest.approx((40,0,0))


def test_pip_equal_speed_linear_and_no_positive_root():
    linear=solve_constant_velocity_pip((-100,-30,50),20.,(100,0,70),(-20,0,0))
    assert linear.status=="READY" and linear.equation_degree==1
    assert linear.time_to_pip_s==pytest.approx(5.1625)
    assert linear.predicted_interception_position_m==pytest.approx((-3.25,0,70))
    away=solve_constant_velocity_pip((0,0,0),20.,(10,0,0),(20,0,0))
    assert away.status=="NO_POSITIVE_PIP_SOLUTION" and away.equation_degree==1
    assert away.time_to_pip_s is None
    assert solve_constant_velocity_pip((0,0,0),20.,(10,0,0),(30,0,0)).status=="NO_POSITIVE_PIP_SOLUTION"
    assert solve_constant_velocity_pip((0,0,0),20.,(0,10,0),(30,0,0)).status=="NO_POSITIVE_PIP_SOLUTION"
    signature=inspect.signature(estimate_three_point_pip_from_history)
    assert set(signature.parameters)=={"history","target_index","defender_position","defender_speed"}


def test_pip_near_tangent_roundoff_is_not_a_false_no_solution():
    tangent=solve_constant_velocity_pip((0,0,0),20.,(10,0,0),(-20,20+1e-12,0))
    assert tangent.status=="READY" and tangent.equation_degree==2
    assert tangent.discriminant==0.
    assert tangent.time_to_pip_s==pytest.approx(.5,abs=1e-10)
    clear_miss=solve_constant_velocity_pip((0,0,0),20.,(10,0,0),(-20,21,0))
    assert clear_miss.status=="NO_POSITIVE_PIP_SOLUTION"


def test_three_point_velocity_and_acceleration_diagnostic_only():
    history=history_for((30,4,2),(-20,1,0),(0,4,2))
    result=estimate_three_point_pip_from_history(history,0,(-100,-30,50),20.)
    assert result.status=="READY" and result.horizon_kind=="constant_velocity_pip"
    assert result.estimated_target_velocity_mps==pytest.approx((-20,1,0))
    assert result.estimated_target_acceleration_mps2==pytest.approx((0,4,2))
    direct=solve_constant_velocity_pip((-100,-30,50),20.,(30,4,2),(-20,1,0))
    assert result.horizon_s==pytest.approx(direct.time_to_pip_s)
    assert result.predicted_target_position_m==pytest.approx(direct.predicted_interception_position_m)
    assert result.pip_equation_degree==direct.equation_degree
    assert result==estimate_three_point_pip_from_history(history,0,(-100,-30,50),20.)


def test_pip_history_status_and_environment_chain():
    history=IntruderHistory(.05)
    history.observe(0.,[[4,3,2],[0,0,0]],[True,False],"fixture")
    assert estimate_three_point_pip_from_history(history,0,(0,0,0),20.).status=="INSUFFICIENT_HISTORY"
    history.observe(.05,[[4,3,2],[0,0,0]],[False,False],"fixture")
    assert estimate_three_point_pip_from_history(history,0,(0,0,0),20.).status=="INACTIVE"
    config=InterceptionConfig(scale="1v1",intruder_mode="straight")
    env=InterceptionEnv(config)
    _,info=env.reset(17)
    history=IntruderHistory(config.dt)
    for step in range(3):
        history.observe(info["time"],info["intruders"][:,:3],info["intruder_mask"],info["scenario_id"])
        if step==2:
            result=estimate_three_point_pip_from_history(history,0,info["defenders"][0,:3],20.)
            assert result.status=="READY" and math.isfinite(result.horizon_s)
            np.testing.assert_allclose(np.diff(history.observed(0)[0]),[.05,.05])
        if step<2:
            _,_,_,_,info=env.step(np.full((5,3),.5))
