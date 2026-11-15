import math

import numpy as np
import pytest

from env.ideal_interception import InterceptionConfig,InterceptionEnv
from prediction.history import IntruderHistory
from prediction.intercept_prediction import (estimate_zuo_prediction_from_history,
    estimate_three_point_adaptive_from_history)


def history_for(position,velocity=(0,0,0),acceleration=(0,0,0),dt=.05):
    p,v,a=map(lambda x:np.asarray(x,float),(position,velocity,acceleration))
    history=IntruderHistory(dt)
    for k,offset in enumerate((-2*dt,-dt,0.)):
        observed=p+v*offset+.5*a*offset*offset
        history.observe(k*dt,[observed,[0.,0.,0.]],[True,False],"fixture")
    return history


def adaptive(history,defender=(0,0,0),**options):
    return estimate_three_point_adaptive_from_history(history,0,defender,20.,.1,**options)


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


def test_adaptive_stationary_uniform_and_contact():
    stationary=adaptive(history_for((30,0,0)))
    assert stationary.status=="READY" and stationary.maneuver_horizon_s==math.inf
    assert stationary.range_horizon_s==pytest.approx(29.9/20)
    assert stationary.horizon_s==1. and stationary.horizon_limited_by=="maximum"
    uniform=adaptive(history_for((3,0,0),(0,10,0)))
    assert uniform.status=="READY" and uniform.maneuver_horizon_s==math.inf
    assert uniform.horizon_s==pytest.approx(2.9/20)
    contact=adaptive(history_for((.05,0,0)))
    assert contact.horizon_s==0 and contact.horizon_limited_by=="contact"
    assert contact.predicted_target_position_m==pytest.approx((.05,0,0))
    assert contact.equation_residual_m is None


def test_maneuver_radial_and_clamp_semantics():
    mild=adaptive(history_for((3,0,0),(0,10,0),(0,10,0)))
    strong=adaptive(history_for((3,0,0),(0,10,0),(0,20,0)))
    assert strong.maneuver_horizon_s<mild.maneuver_horizon_s
    assert strong.horizon_s<mild.horizon_s
    assert mild.horizon_limited_by=="maneuver"
    away=adaptive(history_for((3,0,0),(10,0,0)))
    toward=adaptive(history_for((3,0,0),(-10,0,0)))
    lateral=adaptive(history_for((3,0,0),(0,10,0)))
    assert away.radial_target_speed_mps==pytest.approx(10.)
    assert toward.radial_target_speed_mps==pytest.approx(-10.)
    assert lateral.radial_target_speed_mps==pytest.approx(0.)
    assert away.range_horizon_s>lateral.range_horizon_s>toward.range_horizon_s
    floor=adaptive(history_for((3,0,0),(30,0,0)))
    assert floor.effective_closing_speed_mps<0 and math.isfinite(floor.horizon_s)
    assert floor.horizon_s==1. and floor.horizon_limited_by=="maximum"
    minimum=adaptive(history_for((.2,0,0)))
    assert minimum.horizon_s==pytest.approx(.05)
    assert minimum.horizon_limited_by=="minimum"


def test_noncollinear_3d_determinism_and_history_status():
    history=history_for((4,3,2),(1,2,3),(.2,.4,.6))
    first=adaptive(history,defender=(-1,-2,-3))
    assert first==adaptive(history,defender=(-1,-2,-3))
    assert first.status=="READY" and first.horizon_kind=="adaptive_lookahead"
    assert .05<=first.horizon_s<=1.
    assert np.isfinite(first.predicted_target_position_m).all()
    assert np.isfinite(first.estimated_target_velocity_mps).all()
    assert np.isfinite(first.estimated_target_acceleration_mps2).all()
    history.observe(.15,[[4,3,2],[0,0,0]],[False,False],"fixture")
    assert adaptive(history).status=="INACTIVE"
    short=IntruderHistory(.05)
    short.observe(0.,[[4,3,2],[0,0,0]],[True,False],"fixture")
    assert adaptive(short).status=="INSUFFICIENT_HISTORY"


def test_real_environment_history_prediction_chain():
    config=InterceptionConfig(scale="1v1",intruder_mode="straight")
    env=InterceptionEnv(config)
    _,info=env.reset(17)
    history=IntruderHistory(config.dt)
    for step in range(3):
        history.observe(info["time"],info["intruders"][:,:3],info["intruder_mask"],info["scenario_id"])
        if step==1:
            zuo=estimate_zuo_prediction_from_history(history,0,info["defenders"][0,:3],20.)
            assert zuo.status=="READY" and np.isfinite(zuo.predicted_target_position_m).all()
        if step==2:
            three=adaptive(history,info["defenders"][0,:3])
            assert three.status=="READY" and np.isfinite(three.predicted_target_position_m).all()
            np.testing.assert_allclose(np.diff(history.observed(0)[0]),[.05,.05])
        if step<2:
            _,_,_,_,info=env.step(np.full((5,3),.5))
