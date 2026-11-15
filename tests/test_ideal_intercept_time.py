import math

import numpy as np
import pytest

from prediction.history import IntruderHistory
from prediction.intercept_time import (estimate_constant_velocity_contact, estimate_from_history,
                                       zuo_lookahead_time)


def test_zuo_lookahead_is_not_contact_time():
    horizon=zuo_lookahead_time((0,0,0),(30,0,0),20.,20.)
    assert horizon==pytest.approx(30./16.)
    estimate=estimate_constant_velocity_contact((0,0,0),20.,(30,0,0),(-20,0,0),.5)
    assert estimate.time_to_contact_s==pytest.approx(29.5/40.)
    assert horizon!=pytest.approx(estimate.time_to_contact_s)


def test_analytic_encounter_and_absolute_clock():
    result=estimate_constant_velocity_contact((0,0,0),20.,(30,0,0),(-20,0,0),.5,observation_time=3.)
    assert result.status=="CONTACT_POSSIBLE"
    assert result.time_to_contact_s==pytest.approx(.7375)
    assert result.predicted_contact_time_s==pytest.approx(3.7375)
    assert result.predicted_target_position_m==pytest.approx((15.25,0,0))
    defender_center=np.array((0.,0.,0.))+np.array((20.,0.,0.))*result.time_to_contact_s
    assert np.linalg.norm(defender_center-result.predicted_target_position_m)==pytest.approx(.5)


def test_no_solution_horizon_and_initial_contact():
    away=estimate_constant_velocity_contact((0,0,0),20.,(30,0,0),(20,0,0),.5)
    assert away.status=="NO_CONSTANT_VELOCITY_SOLUTION" and away.time_to_contact_s is None
    limited=estimate_constant_velocity_contact((0,0,0),20.,(30,0,0),(-20,0,0),.5,max_horizon_s=.5)
    assert limited.status=="NO_CONTACT_IN_HORIZON"
    touching=estimate_constant_velocity_contact((0,0,0),20.,(.5,0,0),(0,0,0),.5)
    assert touching.time_to_contact_s==0
    for bad in (math.nan,0.,-1.):
        with pytest.raises(ValueError):
            estimate_constant_velocity_contact((0,0,0),bad,(1,0,0),(0,0,0),.5)


def test_history_adapter_is_causal_and_respects_exit():
    history=IntruderHistory(.1)
    history.observe(0.,[[30,0,0],[0,0,0]],[True,False],"scene")
    assert estimate_from_history(history,0,(0,0,0),20.,.5).status=="INSUFFICIENT_HISTORY"
    history.observe(.1,[[28,0,0],[0,0,0]],[True,False],"scene")
    result=estimate_from_history(history,0,(0,0,0),20.,.5)
    assert result.status=="CONTACT_POSSIBLE"
    assert result.time_to_contact_s==pytest.approx(27.5/40.)
    assert result.predicted_contact_time_s==pytest.approx(.1+27.5/40.)
    history.observe(.2,[[0,0,0],[0,0,0]],[False,False],"scene")
    assert estimate_from_history(history,0,(0,0,0),20.,.5).status=="INACTIVE"
