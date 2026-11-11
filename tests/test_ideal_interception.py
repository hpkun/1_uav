from dataclasses import replace
import numpy as np
import pytest

from env.ideal_interception import InterceptionConfig, InterceptionEnv
from env.ideal_interception.config import OBS_DIM, STATE_DIM
from env.ideal_interception.geometry import region_entry


def actions(direction=(.5,.5,.5)):
    return np.tile(direction,(5,1))


def test_bilateral_shapes_constant_speeds_and_zero_direction():
    env = InterceptionEnv(InterceptionConfig())
    obs,info = env.reset(7)
    assert obs.shape==(5,OBS_DIM) and info["centralized_state"].shape==(STATE_DIM,)
    assert info["defender_mask"].sum()==1 and info["intruder_mask"].sum()==1
    start = info["defenders"][0].copy()
    target = info["intruders"][0].copy()
    obs,_,term,trunc,info = env.step(actions())
    assert not term and not trunc
    np.testing.assert_allclose(info["defenders"][0,:3],start[:3]+start[3:]*.5)
    np.testing.assert_allclose(info["intruders"][0,:3],target[:3]+target[3:]*.5)
    assert np.linalg.norm(info["defenders"][0,3:])==pytest.approx(2.)
    assert np.linalg.norm(info["intruders"][0,3:])==pytest.approx(1.)
    assert not obs[1:].any()


def test_geometric_interception_is_event_and_member_exit():
    base = InterceptionConfig()
    config = replace(base,dt=4.,defender_starts=((-5.,0.,10.),)+base.defender_starts[1:],
                     intruder_starts=((5.,0.,10.),base.intruder_starts[1]),
                     intruder_ends=((-20.,0.,10.),base.intruder_ends[1]))
    env = InterceptionEnv(config)
    env.reset()
    _,reward,terminated,truncated,info = env.step(actions())
    assert terminated and not truncated and info["outcome"]=="all_intercepted"
    assert info["time"]==pytest.approx(9.5/3)
    assert not info["defender_mask"][0] and not info["intruder_mask"][0]
    assert info["defender_status"][0]=="contact_exit"
    assert info["intruder_status"][0]=="intercepted"
    assert info["interception_count"]==1
    assert [e["kind"] for e in info["events"]]==[
        "interception_contact","member_inactive","member_inactive","episode_end"]
    assert info["reward_parts"]["interception"]==config.interception_reward
    assert info["reward_parts"]["protection"]==config.protection_reward
    assert sum(reward)==pytest.approx(info["team_reward"])
    with pytest.raises(RuntimeError,match="reset required"):
        env.step(actions())


def test_multi_target_contact_exits_pair_and_episode_continues():
    base = InterceptionConfig(scale="5v2")
    defenders = ((-5.,0.,10.),)+base.defender_starts[1:]
    intruders = ((5.,0.,10.),base.intruder_starts[1])
    env = InterceptionEnv(replace(base,dt=4.,defender_starts=defenders,
                     intruder_starts=intruders,
                     intruder_ends=((-20.,0.,10.),base.intruder_ends[1])))
    env.reset()
    _,_,terminated,_,info = env.step(actions())
    assert not terminated and info["outcome"]=="ongoing" and info["time"]==4.
    assert not info["defender_mask"][0] and not info["intruder_mask"][0]
    assert info["defender_mask"][1:].all() and info["intruder_mask"][1]
    frozen = info["defenders"][0].copy()
    _,_,_,_,info = env.step(actions())
    np.testing.assert_array_equal(info["defenders"][0],frozen)
    assert info["interception_count"]==1


def test_region_breach_boundary_and_friendly_contact_times():
    base = InterceptionConfig()
    breach = replace(base,dt=10.,defender_starts=((-25.,-20.,10.),)+base.defender_starts[1:],
                     intruder_starts=((-10.,0.,10.),base.intruder_starts[1]),
                     intruder_ends=((-25.,0.,10.),base.intruder_ends[1]))
    env = InterceptionEnv(breach)
    env.reset()
    _,_,term,_,info = env.step(actions())
    assert term and info["outcome"]=="breach" and info["time"]==pytest.approx(6.)
    assert info["breach_count"]==1
    assert any(e["kind"]=="region_breach" for e in info["events"])
    assert region_entry(np.array([-10.,0.,10.]),np.array([-1.,0.,0.]),
        np.array([-20.,0.,0.]),4.,35.,10.)==pytest.approx(6.)
    boundary = replace(base,dt=2.,defender_starts=((28.,-10.,10.),)+base.defender_starts[1:])
    env = InterceptionEnv(boundary)
    env.reset()
    _,_,term,_,info = env.step(actions())
    assert term and info["outcome"]=="defender_boundary" and info["time"]==pytest.approx(1.)
    starts = ((-5.,-2.,10.),(-5.,2.,10.))+base.defender_starts[2:]
    env = InterceptionEnv(replace(base,scale="3v1",dt=2.,defender_starts=starts))
    env.reset()
    move = actions()
    move[0],move[1]=[.5,1.,.5],[.5,0.,.5]
    _,_,term,_,info = env.step(move)
    assert term and info["outcome"]=="friendly_collision"
    assert info["time"]==pytest.approx(.875)
    assert info["friendly_collision_count"]==1


def test_intruder_contact_snapshot_determinism_and_protocol_rejection():
    base = InterceptionConfig(scale="5v2")
    intruders = ((5.,-2.,15.),(5.,2.,15.))
    ends = ((5.,10.,15.),(5.,-10.,15.))
    config = replace(base,dt=2.,intruder_starts=intruders,intruder_ends=ends)
    a,b = InterceptionEnv(config),InterceptionEnv(config)
    first = a.reset(11)[1]
    second = b.reset(11)[1]
    assert first["scenario_id"]==second["scenario_id"]
    snapshot = a.snapshot()
    b.reset(12)
    b.restore(snapshot)
    x,y = a.step(actions()),b.step(actions())
    assert x[4]["events"]==y[4]["events"]
    np.testing.assert_array_equal(x[4]["intruders"],y[4]["intruders"])
    assert x[4]["outcome"]=="intruder_collision"
    assert x[4]["time"]==pytest.approx(1.75)
    with pytest.raises(ValueError,match="incompatible"):
        b.restore(snapshot|{"protocol":"uav-ideal-kinematic-v1.2.0"})


def test_two_interceptions_in_one_step_keep_other_members_moving():
    base = InterceptionConfig(scale="5v2")
    defenders = ((-5.,-5.,10.),(-5.,5.,10.))+base.defender_starts[2:]
    intruders = ((5.,-5.,10.),(10.,5.,10.))
    ends = ((-25.,-5.,10.),(-25.,5.,10.))
    env = InterceptionEnv(replace(base,dt=6.,defender_starts=defenders,
                                  intruder_starts=intruders,intruder_ends=ends))
    env.reset()
    _,_,terminated,_,info = env.step(actions())
    contacts = [e for e in info["events"] if e["kind"]=="interception_contact"]
    assert terminated and info["outcome"]=="all_intercepted"
    assert len(contacts)==2 and info["interception_count"]==2
    assert contacts[0]["time"]==pytest.approx(9.5/3)
    assert contacts[1]["time"]==pytest.approx(14.5/3)
    assert info["time"]==pytest.approx(14.5/3)
    assert not info["defender_mask"][:2].any() and info["defender_mask"][2:].all()
    assert not info["intruder_mask"].any()
    assert info["defenders"][2,0] > defenders[2][0]
    assert all(e["time"] <= info["time"] for e in info["events"])


def test_breach_inside_step_with_both_endpoints_outside_and_tie_priority():
    base = InterceptionConfig()
    intruders = ((-10.,0.,10.),base.intruder_starts[1])
    ends = ((-25.,0.,10.),base.intruder_ends[1])
    far = ((-25.,-20.,10.),)+base.defender_starts[1:]
    config = replace(base,dt=2.,defender_starts=far,intruder_starts=intruders,
                     intruder_ends=ends,intruder_speeds=(10.,1.))
    env = InterceptionEnv(config)
    env.reset()
    _,_,terminated,_,info = env.step(actions())
    assert terminated and info["outcome"]=="breach"
    assert info["time"]==pytest.approx(.6)
    assert info["intruders"][0,0]==pytest.approx(-16.)
    # Same physical time: region breach wins over defender-intruder contact.
    tie = replace(base,dt=7.,defender_starts=((-28.5,0.,10.),)+base.defender_starts[1:],
                  intruder_starts=intruders,intruder_ends=ends)
    env = InterceptionEnv(tie)
    env.reset()
    _,_,_,_,info = env.step(actions())
    assert info["time"]==pytest.approx(6.)
    assert info["outcome"]=="breach" and info["interception_count"]==0


def test_tangent_protected_region_entry_is_a_continuous_event():
    entry = region_entry(np.array([-5.,4.,10.]),np.array([1.,0.,0.]),
                         np.array([0.,0.,0.]),4.,35.,10.)
    assert entry == pytest.approx(5.)
