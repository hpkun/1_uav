from dataclasses import replace

import numpy as np
import pytest

from env.interception import InterceptionConfig
from env.interception.motion import advance, execute_velocity
from env.preset_tracking import run_tracking
from env.preset_trajectory import TrajectoryConfig, build_path, initial_states, make_paths


@pytest.mark.parametrize("kind",["straight","bezier"])
def test_endpoints_and_derivatives_are_continuous(kind):
    c=TrajectoryConfig()
    path,=make_paths(kind,initial_states(c,1200),c,1200)
    np.testing.assert_array_equal(path.sample(-1)[0],path.points[0])
    np.testing.assert_array_equal(path.sample(0)[0],path.points[0])
    for t in (path.duration,path.duration+1,1000.):
        p,v,a,mode=path.sample(t)
        np.testing.assert_array_equal(p,path.points[-1])
        np.testing.assert_array_equal(v,0.)
        np.testing.assert_array_equal(a,0.)
        assert mode=="hold"
    eps=1e-5
    for t in (0.,path.duration/2,path.duration):
        p,v,a,_=path.sample(t)
        pm,vm,_,_=path.sample(t-eps)
        pp,vp,_,_=path.sample(t+eps)
        np.testing.assert_allclose((pp-pm)/(2*eps),v,atol=1e-8)
        np.testing.assert_allclose((vp-vm)/(2*eps),a,atol=1e-5)
        assert np.linalg.norm(p-pp)<1e-3
    if kind=="straight":
        vector=np.subtract(path.points[-1],path.points[0])
        for t in np.linspace(0,path.duration,20):
            np.testing.assert_allclose(np.cross(path.sample(t)[0]-path.points[0],vector),0.,atol=1e-11)


@pytest.mark.parametrize("kind",["straight","bezier"])
def test_zero_length_and_invalid_control_points(kind):
    c=TrajectoryConfig()
    count=2 if kind=="straight" else 4
    p=np.tile([0.,0.,10.],(count,1))
    path=build_path(kind,p,c)
    assert path.length==path.duration==0 and path.sample(0)[-1]=="hold"
    np.testing.assert_array_equal(path.sample(0)[1],0.)
    p[1,2]=c.upper[2]
    with pytest.raises(ValueError,match="legal flight envelope"):
        build_path(kind,p,c)
    p[1,2]=np.nan
    with pytest.raises(ValueError,match="legal flight envelope"):
        build_path(kind,p,c)
    with pytest.raises(ValueError,match="reference time"):
        path.sample(float("nan"))


def test_degenerate_bezier_derivative_does_not_need_arc_length_inversion():
    c=TrajectoryConfig()
    p=[[0,0,10],[0,0,10],[4,0,10],[4,0,10]]
    path=build_path("bezier",p,c)
    assert path.length==pytest.approx(4.)
    for t in np.linspace(0,path.duration,100):
        q,v,a,_=path.sample(t)
        assert np.isfinite(np.r_[q,v,a]).all()
    assert path.sample(path.duration/2)[0][0]==pytest.approx(2.)
    tiny=build_path("straight",[[0,0,10],[1e-300,0,10]],c)
    assert tiny.duration==c.dt
    assert np.isfinite(np.r_[tiny.sample(tiny.duration/2)[:3]]).all()


@pytest.mark.parametrize("value",[0,-1,np.nan,np.inf,13.])
def test_invalid_reference_speed(value):
    with pytest.raises(ValueError,match="reference_speed|reference speed"):
        TrajectoryConfig(reference_speed=value)


def test_determinism_pairing_and_per_aircraft_identity():
    c=TrajectoryConfig(starts=((-22,-12,15),(-18,-6,24)),
                       endpoints=((22,12,23),(12,15,29)),initial_velocities=((.4,.2,0),(.2,.3,0)))
    states=initial_states(c,1200)
    paths=make_paths("bezier",states,c,1200)
    assert paths==make_paths("bezier",states,c,1200)
    assert paths!=make_paths("bezier",states,c,1201)
    original=states.copy()
    for path in paths:
        p,*_=path.sample(path.duration/2)
        p[:] = -100.
    np.testing.assert_array_equal(states,original)
    assert paths==make_paths("bezier",states,c,1200)
    a,b=(run_tracking(kind,1200,c,trace=False) for kind in ("straight","bezier"))
    np.testing.assert_array_equal(a["initial_states"],b["initial_states"])
    assert a["active_mask"]==b["active_mask"]==[True,True]
    assert [m["reference_seed"] for m in b["members"]]==[[1200,1,0],[1200,1,1]]
    for ma,mb in zip(a["members"],b["members"]):
        assert ma["reference"]["points"][-1]==mb["reference"]["points"][-1]


@pytest.mark.parametrize("kind",["straight","bezier"])
@pytest.mark.parametrize("limit",[4.,7.,12.])
def test_continuous_reference_certificate_and_tracking_limits(kind,limit):
    c=TrajectoryConfig(vmax=limit,reference_speed=min(.8*limit,6))
    path,=make_paths(kind,initial_states(c,1227),c,1227)
    lo,hi=c.flight_bounds
    for t in np.linspace(0,path.duration,101):
        p,v,a,_=path.sample(t)
        assert ((p>=lo)&(p<=hi)).all()
        assert np.linalg.norm(v)<=c.reference_speed+1e-12
        required=v+c.tau*a
        np.testing.assert_allclose(execute_velocity(required,v,c.vmax,c.tau,c),required,atol=1e-12)
        assert np.linalg.norm(a[:2])<=c.horizontal_acceleration_max+1e-12
    record=run_tracking(kind,1227,c)
    assert record["outcome"]=="completed" and record["numerical_failure"] is None
    assert record["members"][0]["settled"]
    for previous,after in zip(record["frames"],record["frames"][1:]):
        predicted=advance(previous["actual_states"],previous["executed_velocities"],c.tau,
                          previous["command_duration"])
        np.testing.assert_array_equal(after["actual_states"],predicted)
    item,=record["members"]
    assert item["max_speed"]<=limit+1e-12
    assert item["max_horizontal_acceleration"]<=c.horizontal_acceleration_max+1e-12
    assert -c.descent_vmax-1e-12<=item["min_vertical_speed"]<=item["max_vertical_speed"]<=c.climb_vmax+1e-12


def test_active_limiter_and_incomplete_reference():
    c=TrajectoryConfig(vmax=4.,reference_speed=3.,initial_velocities=((3.,0.,2.5),),
                       velocity_gain=3.,horizontal_acceleration_max=.8,max_steps=2)
    out=run_tracking("bezier",1200,c)
    row=out["frames"][0]
    assert all(flag[0] for flag in row["constraints"].values())
    assert np.linalg.norm(row["actual_accelerations"][0,:2])==pytest.approx(.8)
    assert out["outcome"]=="time_limit"
    assert not out["members"][0]["reference_completed"]


def test_continuous_boundary_and_long_hold_repeatable():
    base=TrajectoryConfig()
    c=replace(base,starts=((5.4,0,10),),endpoints=((0,0,12),),initial_velocities=((3,0,0),),
              start_jitter=(0,0,0),upper=(6,20,30))
    out=run_tracking("straight",1,c)
    assert out["outcome"]=="boundary" and 0 < out["duration"] < c.dt
    assert out["frames"][-1]["actual_states"][0,0]==pytest.approx(5.5,abs=1e-6)
    c=replace(base,hold_seconds=60.)
    a=run_tracking("bezier",1200,c)
    assert a["duration"]>60 and a["members"][0]["settled"]
    assert all(np.isfinite(f["actual_states"]).all() for f in a["frames"])
    b=run_tracking("bezier",1200,c)
    np.testing.assert_equal(a,b)


@pytest.mark.parametrize(("kind","steps","digest"),[
    ("goal_seeking",87,"7eedf7be4aef7f13653a62694a7438c7e067954c515f04bc6e17ec6b0b163ef6"),
    ("bezier",110,"8f410713094a875cc2884294e6584664bd1d30d460c56283b5d45df5c79bd478"),
])
def test_historical_mode_command_state_and_reward_unchanged(kind, steps, digest, check_interception_history):
    check_interception_history(kind, steps, digest, InterceptionConfig(evader_mode=kind))
