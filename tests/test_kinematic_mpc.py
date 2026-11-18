"""Direct-control MPC interfaces and constraint-aware execution."""
import numpy as np
import pytest

from kinematic_interception.dynamics import DynamicsConfig,PointState,WingState
from kinematic_interception.mpc.dynamics import AControl,BControl,direct_step
from kinematic_interception.mpc.controller import MPCController,MPCConfig
from kinematic_interception.prediction import TargetHistory
from kinematic_interception.environment import KinematicInterceptionEnv,one_vs_one_scene


def _history():
    history=TargetHistory(slots=2)
    for k in range(3):
        history.observe(.05*k,np.array(((50.-k,0.,30.),(40.-k,10.,30.))),
                        np.array((True,True)))
    return history


def test_a_direct_control_matches_constant_acceleration_and_rejects_limits():
    config=DynamicsConfig()
    state=PointState(np.zeros(3),np.array((10.,0.,0.)))
    control=AControl((0.,10.,0.))
    end=direct_step(state,control,config,.05)
    assert np.allclose(end.position,(.5,.0125,0.))
    assert np.allclose(end.velocity,(10.,.5,0.))
    with pytest.raises(ValueError,match="acceleration"):
        direct_step(state,AControl((11.,0.,0.)),config,.05)


def test_b_direct_control_respects_rate_and_speed_constraints():
    config=DynamicsConfig()
    state=WingState(np.zeros(3),20.,0.,0.)
    control=BControl(-4.,.4,.1)
    for _ in range(100):
        state=direct_step(state,control,config,.01)
        assert config.v_min<=state.speed<=config.v_max
        assert abs(state.flightpath)<=config.flightpath_max+1e-12
    with pytest.raises(ValueError,match="lateral"):
        direct_step(WingState(np.zeros(3),20.,0.,0.),BControl(0.,.5,.3),config,.01)


@pytest.mark.parametrize("model",("A","B"))
def test_mpc_outputs_executable_control_and_invalidates_target_switch(model):
    config=DynamicsConfig()
    history=_history()
    controller=MPCController(config,history,MPCConfig(b_max_iterations=5))
    state=(PointState(np.array((0.,0.,30.)),np.array((20.,0.,0.))) if model=="A"
           else WingState(np.array((0.,0.,30.)),20.,0.,0.))
    control,detail=controller.command(0,state,0,np.array((48.,0.,30.)),.1)
    assert detail["status"] in ("MPC_READY","FALLBACK")
    assert isinstance(control,AControl if model=="A" else BControl)
    direct_step(state,control,config,.01)
    _,held=controller.command(0,state,0,np.array((48.,0.,30.)),.15)
    assert held["status"]=="HELD_MPC"
    _,switched=controller.command(0,state,1,np.array((48.,0.,30.)),.15)
    assert switched["status"]!="HELD_MPC"


@pytest.mark.parametrize("model",("A","B"))
def test_direct_control_prediction_matches_environment_substeps(model):
    env=KinematicInterceptionEnv(one_vs_one_scene(17),model)
    initial=env.states[0].copy()
    control=AControl((-1.,2.,0.)) if model=="A" else BControl(0.,.1,0.)
    predicted=initial
    for _ in range(5):
        predicted=direct_step(predicted,control,env.dynamics,.01)
    env.step([control])
    actual=env.states[0]
    assert np.allclose(actual.position,predicted.position,atol=1e-12)
    if model=="A":
        assert np.allclose(actual.velocity,predicted.velocity,atol=1e-12)
    else:
        assert actual.speed==pytest.approx(predicted.speed)
        assert actual.yaw==pytest.approx(predicted.yaw)
