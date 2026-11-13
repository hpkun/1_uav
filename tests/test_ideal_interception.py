from dataclasses import replace
import numpy as np
import pytest

from env.ideal_interception import InterceptionConfig, InterceptionEnv
from env.ideal_interception.config import OBS_DIM, STATE_DIM
from env.ideal_interception.geometry import region_entry
from env.ideal_interception.trajectory import Reference


def actions(direction=(.5,.5,.5)):
    return np.tile(direction,(5,1))


def legacy_geometry(scale="1v1"):
    """Small exact geometry fixtures, independent of the V2 default scale."""
    return InterceptionConfig(scale=scale,dt=.5,max_steps=80,
        lower=(-30.,-30.,0.),upper=(30.,30.,40.),
        defender_speeds=(2.,)*5,intruder_speeds=(1.,)*2,
        defender_starts=((-10.,-10.,10.),(-10.,0.,15.),(-10.,10.,20.),
                         (-14.,-5.,25.),(-14.,5.,30.)),
        intruder_starts=((18.,0.,15.),(18.,8.,25.)),
        intruder_ends=((-25.,0.,15.),(-25.,8.,25.)),
        protected_center=(-20.,0.,0.),protected_radius=4.,protected_height=35.,
        position_scale=30.,speed_scale=3.,random_range=1.)


def test_bilateral_shapes_constant_speeds_and_zero_direction():
    env = InterceptionEnv(InterceptionConfig())
    obs,info = env.reset(7)
    assert obs.shape==(5,OBS_DIM) and info["centralized_state"].shape==(STATE_DIM,)
    assert info["defender_mask"].sum()==1 and info["intruder_mask"].sum()==1
    start = info["defenders"][0].copy()
    target = info["intruders"][0].copy()
    obs,_,term,trunc,info = env.step(actions())
    assert not term and not trunc
    np.testing.assert_allclose(info["defenders"][0,:3],start[:3]+start[3:]*env.config.dt)
    np.testing.assert_allclose(info["intruders"][0,:3],target[:3]+target[3:]*env.config.dt)
    assert np.linalg.norm(info["defenders"][0,3:])==pytest.approx(20.)
    assert np.linalg.norm(info["intruders"][0,3:])==pytest.approx(20.)
    assert not obs[1:].any()


def test_geometric_interception_is_event_and_member_exit():
    base = legacy_geometry()
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
    base = legacy_geometry("5v2")
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
    base = legacy_geometry()
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
    base = legacy_geometry("5v2")
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
        b.restore(snapshot|{"protocol":"uav-ideal-interception-v1.0.0"})


def test_two_interceptions_in_one_step_keep_other_members_moving():
    base = legacy_geometry("5v2")
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
    base = legacy_geometry()
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


@pytest.mark.parametrize("scale",["1v1","3v1","5v2"])
def test_v2_defaults_are_valid_fixed_speed_scenarios(scale):
    config = InterceptionConfig(scale=scale)
    env = InterceptionEnv(config)
    obs,info = env.reset(17)
    assert obs.shape==(5,68)
    assert info["centralized_state"].shape==(66,)
    assert np.isfinite(obs).all() and np.isfinite(info["centralized_state"]).all()
    assert len(info["intruder_paths"])==config.intruder_count
    assert all(path["duration"]>0 and path["polyline_sha256"] for path in info["intruder_paths"])
    assert np.allclose(np.linalg.norm(info["defenders"][info["defender_mask"],3:],axis=1),20.)
    assert np.allclose(np.linalg.norm(info["intruders"][info["intruder_mask"],3:],axis=1),20.)
    _,_,_,_,next_info=env.step(actions())
    assert np.isfinite(next_info["centralized_state"]).all()
    assert all(e["kind"] not in ("friendly_warning","safety_violation") for e in next_info["events"])


def test_path_completion_exits_intruder_without_stationary_active_member():
    config=replace(InterceptionConfig(),dt=1.,intruder_starts=((0.,0.,60.),(200.,10.,80.)),
                   intruder_ends=((-10.,0.,60.),(-260.,10.,80.)))
    env=InterceptionEnv(config)
    env.reset()
    _,_,terminated,_,info=env.step(actions())
    assert terminated and info["outcome"]=="intruder_path_complete"
    assert info["time"]==pytest.approx(.5)
    assert not info["intruder_mask"][0] and info["intruder_status"][0]=="path_complete"
    assert np.all(info["intruders"][0,3:]==0.)
    assert [e["kind"] for e in info["events"]]==["path_complete","member_inactive","episode_end"]
    assert info["reward_parts"]["path_complete"]==-config.failure_penalty


def test_fixed_motion_contact_time_is_control_period_invariant():
    base=legacy_geometry()
    starts=((-5.,0.,10.),)+base.defender_starts[1:]
    ends=((-20.,0.,10.),base.intruder_ends[1])
    intruders=((5.,0.,10.),base.intruder_starts[1])
    times=[]
    for dt in (4.,.1):
        env=InterceptionEnv(replace(base,dt=dt,defender_starts=starts,
                                    intruder_starts=intruders,intruder_ends=ends))
        env.reset()
        while True:
            _,_,done,_,info=env.step(actions())
            if done:
                times.append(info["time"])
                assert info["outcome"]=="all_intercepted"
                break
    assert times==pytest.approx([9.5/3]*2)


def test_seeded_random_path_identity_and_bezier_segment_speed():
    config=replace(InterceptionConfig(scale="5v2",initialization="random",intruder_mode="bezier"),
                   random_range=3.)
    a,b=InterceptionEnv(config),InterceptionEnv(config)
    first=a.reset(123)[1]
    second=b.reset(123)[1]
    assert first["scenario_id"]==second["scenario_id"]
    assert first["intruder_paths"]==second["intruder_paths"]
    assert first["scenario_id"]!=a.reset(124)[1]["scenario_id"]
    path=Reference(first["intruder_paths"][0]["start"],config.intruder_ends[0],20.,"bezier")
    segments=list(path.segments(0.,.1))
    assert len(segments)>1
    assert all(np.linalg.norm(velocity)==pytest.approx(20.) for _,_,_,velocity in segments)
    for (left,right,position,velocity),(next_left,_,next_position,_) in zip(segments,segments[1:]):
        assert right==pytest.approx(next_left)
        np.testing.assert_allclose(position+(right-left)*velocity,next_position,atol=1e-9)


def test_bezier_polyline_midpoint_error_respects_curvature_bound():
    path=Reference((200.,-10.,60.),(-260.,-10.,60.),20.,"bezier")
    p0,p3=path.points[0],path.points[-1]
    p1,p2=map(np.asarray,path.controls)
    t=(np.arange(1024)+.5)/1024
    true=((1-t)[:,None]**3*p0+3*((1-t)**2*t)[:,None]*p1
          +3*((1-t)*t*t)[:,None]*p2+(t**3)[:,None]*p3)
    midpoint=(path.points[:-1]+path.points[1:])/2
    error=np.linalg.norm(true-midpoint,axis=1).max()
    second_derivative_bound=6*max(np.linalg.norm(p2-2*p1+p0),
                                  np.linalg.norm(p3-2*p2+p1))
    assert error <= second_derivative_bound/(8*1024**2)+1e-10
    assert error < .001
