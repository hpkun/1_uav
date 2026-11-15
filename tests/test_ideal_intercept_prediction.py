import math

import numpy as np
import pytest

from prediction.history import IntruderHistory
from prediction.intercept_prediction import (estimate_zuo_prediction_from_history,
    estimate_three_point_contact_from_history)
from prediction.intercept_time import estimate_constant_velocity_contact


def history_of(values, dt=1.):
    history=IntruderHistory(dt)
    for k,x in enumerate(values):
        history.observe(k*dt,[[x,0.,0.],[0.,0.,0.]],[True,False],"fixture")
    return history


def test_zuo_paper_lookahead_and_cold_start():
    history=history_of((32.,))
    early=estimate_zuo_prediction_from_history(history,0,(0,0,0),20.)
    assert early.status=="INSUFFICIENT_HISTORY" and early.horizon_s is None
    history=history_of((32.,30.),.1)
    result=estimate_zuo_prediction_from_history(history,0,(0,0,0),20.)
    assert result.status=="READY" and result.horizon_kind=="paper_lookahead"
    assert result.horizon_s==pytest.approx(30/16)
    assert result.predicted_target_position_m==pytest.approx((-7.5,0,0))
    assert result.paper_direction_unit==pytest.approx((-1,0,0))
    assert result.equation_residual_m is None and result.history_count==2


def test_three_point_constant_velocity_degeneracy_and_current_contact():
    history=history_of((50.,40.,30.))
    result=estimate_three_point_contact_from_history(history,0,(0,0,0),20.,.5)
    baseline=estimate_constant_velocity_contact((0,0,0),20.,(30,0,0),(-10,0,0),.5,2.)
    assert result.status=="CONTACT_POSSIBLE"
    assert result.estimated_target_acceleration_mps2==pytest.approx((0,0,0))
    assert result.horizon_s==pytest.approx(baseline.time_to_contact_s)
    assert result.predicted_target_position_m==pytest.approx(baseline.predicted_target_position_m)
    already=estimate_three_point_contact_from_history(history,0,(30.3,0,0),20.,.5)
    assert already.horizon_s==0 and already.equation_residual_m==0


def test_quadratic_analytic_root_residual_and_earliest_selection():
    history=history_of((46.,39.,30.))
    result=estimate_three_point_contact_from_history(history,0,(0,0,0),20.,.5)
    expected=(-30+math.sqrt(1018))/2
    assert result.status=="CONTACT_POSSIBLE"
    assert result.estimated_target_velocity_mps==pytest.approx((-10,0,0))
    assert result.estimated_target_acceleration_mps2==pytest.approx((-2,0,0))
    assert result.horizon_s==pytest.approx(expected,abs=1e-10)
    assert result.predicted_target_position_m==pytest.approx((30-10*expected-expected**2,0,0))
    assert result.equation_residual_m<1e-9
    # The squared polynomial also has a later positive crossing; choose the first.
    roots=np.roots([1.,20.,100.-60.-400.,-600.-20.,900.-.25])
    assert sum(abs(z.imag)<1e-8 and z.real>0 for z in roots)>=2
    limited=estimate_three_point_contact_from_history(history,0,(0,0,0),20.,.5,.5)
    assert limited.status=="NO_CONTACT_IN_HORIZON" and limited.horizon_s is None


def test_no_solution_history_exit_and_no_future_reference():
    history=history_of((-166.,-69.,30.))
    result=estimate_three_point_contact_from_history(history,0,(0,0,0),20.,.5)
    assert result.status=="NO_QUADRATIC_CONTACT_SOLUTION"
    assert result.predicted_target_position_m is None
    straight_away=history_of((-10.,10.,30.))
    assert estimate_three_point_contact_from_history(straight_away,0,(0,0,0),20.,.5,.5).status=="NO_QUADRATIC_CONTACT_SOLUTION"
    history.observe(3.,[[30,0,0],[0,0,0]],[False,False],"fixture")
    assert estimate_zuo_prediction_from_history(history,0,(0,0,0),20.).status=="INACTIVE"
    assert estimate_three_point_contact_from_history(history,0,(0,0,0),20.,.5).status=="INACTIVE"
