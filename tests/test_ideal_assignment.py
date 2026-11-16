from dataclasses import replace

import numpy as np
import pytest

from env.ideal_interception import InterceptionConfig, InterceptionEnv
from env.ideal_interception.assignment import nearest_distance_assignment
from env.ideal_interception.config import PROTOCOL


def test_nearest_assignment_scales_masks_ties_and_many_to_one():
    defenders=np.array([[0,0,0],[10,0,0],[1,0,0],[9,0,0],[9,0,1.]],float)
    targets=np.array([[0,0,2],[10,0,2]],float)
    own=np.ones(5,bool)
    active=np.ones(2,bool)
    choose=nearest_distance_assignment
    assert choose(defenders[:1],targets[:1],own[:1],active[:1])=={0:0}
    assert choose(defenders[:3],targets[:1],own[:3],active[:1])=={0:0,1:0,2:0}
    assert choose(defenders,targets,own,active)=={0:0,1:1,2:0,3:1,4:1}
    assert choose(np.zeros((5,3)),targets,own,active)=={i:0 for i in range(5)}
    tied=np.array([[5.,0.,2.]])
    assert all(choose(tied,targets,[True],active)=={0:0} for _ in range(10))
    assert choose(defenders,targets,own,[False,True])=={i:1 for i in range(5)}
    assert choose(defenders,targets,[True,False,True,False,True],active)=={0:0,2:0,4:1}
    assert choose(defenders,targets,own,[False,False])=={}
    with pytest.raises(ValueError):
        choose(defenders,targets,own,[True])


def test_defaults_budget_reset_observations_and_snapshot():
    config=InterceptionConfig(scale="5v2")
    assert (config.dt,config.max_steps,config.contact_distance)==(.05,600,.1)
    assert config.dt*config.max_steps==pytest.approx(30.)
    assert config.defender_speeds[0]*config.dt==pytest.approx(1.)
    env=InterceptionEnv(config)
    obs,info=env.reset(17)
    expected=nearest_distance_assignment(info["defenders"][:,:3],info["intruders"][:,:3],
                                          info["defender_mask"],info["intruder_mask"])
    assert info["assignment"]==env.assignment==expected
    assert info["scenario_id"]==InterceptionEnv(config).reset(17)[1]["scenario_id"]
    assert info["centralized_state"][-1]==pytest.approx(1.)
    path=info["scene_diagnostics"]["paths"][0]
    assert path["duration_fraction"]==pytest.approx(path["duration"]/30.)
    for i,j in expected.items():
        np.testing.assert_array_equal(obs[i,55:57],np.eye(2)[j])
        np.testing.assert_array_equal(info["centralized_state"][49+2*i:51+2*i],np.eye(2)[j])
    snapshot=env.snapshot()
    env.step(np.full((5,3),.5))
    assert env.output()[1]["centralized_state"][-1]==pytest.approx(1.-config.dt/30.,abs=1e-7)
    env.restore(snapshot)
    assert env.assignment==expected
    with pytest.raises(ValueError,match="incompatible"):
        env.restore(snapshot|{"protocol":"uav-ideal-interception-v2.0.0"})
    assert PROTOCOL=="uav-ideal-interception-v3.1.0"


def test_substep_contact_reassigns_surviving_members():
    base=InterceptionConfig(scale="5v2",intruder_mode="straight")
    config=replace(base,
        defender_starts=((-0.75,0.,60.),)+base.defender_starts[1:],
        intruder_starts=((0.75,0.,60.),)+base.intruder_starts[1:])
    env=InterceptionEnv(config)
    _,initial=env.reset(17)
    assert initial["assignment"][0]==0
    obs,_,terminated,_,info=env.step(np.full((5,3),.5))
    contact=next(e for e in info["events"] if e["kind"]=="interception_contact")
    assert 0<contact["time"]<config.dt
    separation_at_contact=np.linalg.norm(
        initial["defenders"][0,:3]+initial["defenders"][0,3:]*contact["time"]-
        initial["intruders"][0,:3]-initial["intruders"][0,3:]*contact["time"])
    assert separation_at_contact==pytest.approx(config.contact_distance,abs=1e-9)
    # Both endpoints lie outside the 0.1 m sphere; detection is continuous.
    assert np.linalg.norm(initial["defenders"][0,:3]-initial["intruders"][0,:3])>config.contact_distance
    assert np.linalg.norm((initial["defenders"][0,:3]+initial["defenders"][0,3:]*config.dt)-
                          (initial["intruders"][0,:3]+initial["intruders"][0,3:]*config.dt))>config.contact_distance
    assert not terminated and info["outcome"]=="ongoing"
    assert not info["defender_mask"][0] and not info["intruder_mask"][0]
    assert info["assignment"]=={1:1,2:1,3:1,4:1}
    assert info["assignment"]==nearest_distance_assignment(
        info["defenders"][:,:3],info["intruders"][:,:3],info["defender_mask"],info["intruder_mask"])
    assert not obs[0].any()
    for i in range(1,5):
        np.testing.assert_array_equal(obs[i,55:57],[0.,1.])
        np.testing.assert_array_equal(info["centralized_state"][49+2*i:51+2*i],[0.,1.])
    snapshot=env.snapshot()
    restored=InterceptionEnv(config)
    restored.restore(snapshot)
    assert restored.assignment==info["assignment"]
    _,_,_,_,next_info=env.step(np.full((5,3),.5))
    assert next_info["assignment"]==info["assignment"]


def test_terminal_contact_has_no_stale_target_assignment():
    base=InterceptionConfig(intruder_mode="straight")
    config=replace(base,defender_starts=((-0.75,0.,60.),)+base.defender_starts[1:],
        intruder_starts=((0.75,0.,60.),)+base.intruder_starts[1:])
    env=InterceptionEnv(config)
    env.reset(17)
    obs,_,done,_,info=env.step(np.full((5,3),.5))
    assert done and info["outcome"]=="all_intercepted"
    assert info["assignment"]=={}
    assert not obs.any()
    assert not info["centralized_state"][49:59].any()
