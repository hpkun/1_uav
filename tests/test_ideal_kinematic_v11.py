from dataclasses import replace
import numpy as np
import pytest
import torch

from algorithm.mappo.gae import gae
from env.ideal_kinematic import IdealConfig, IdealEnv
from env.ideal_kinematic.geometry import inside_intervals
from env.ideal_kinematic.reference import Reference


@pytest.mark.parametrize("mode", ["straight", "bezier"])
def test_reference_ping_pong_knots_and_cycles(mode):
    path = Reference((0., 0., 10.), (4., 0., 10.), 2., mode, "ping_pong")
    for cycle in range(3):
        for fraction in (.1, .5, .9):
            t = (2*cycle+fraction)*path.duration
            forward, velocity = path.sample(t)
            backward, reverse = path.sample((2*cycle+2-fraction)*path.duration)
            np.testing.assert_allclose(forward, backward, atol=1e-10)
            np.testing.assert_allclose(velocity, -reverse, atol=1e-10)
            assert np.linalg.norm(velocity) == pytest.approx(2.)
    for turn in (path.duration, 2*path.duration, 3*path.duration):
        before = path.sample(turn-1e-8)[0]
        after, velocity = path.sample(turn)
        assert np.linalg.norm(before-after) < 3e-8
        assert np.linalg.norm(velocity) == pytest.approx(2.)
    segments = list(path.segments(path.duration-.2, path.duration+.2))
    assert sum(b-a for a,b,_,_ in segments) == pytest.approx(.4)
    assert all(np.linalg.norm(v) == pytest.approx(2.) for _,_,_,v in segments)
    with pytest.raises(ValueError):
        Reference((0.,0.,10.), (0.,0.,10.), 2., mode, "ping_pong")
    with pytest.raises(ValueError):
        Reference((0.,0.,10.), (4.,0.,10.), 0., mode, "ping_pong")


@pytest.mark.parametrize("task", ["IK2", "IK3"])
def test_default_dynamic_reference_moves_through_entire_horizon(task):
    env = IdealEnv(IdealConfig(task=task))
    env.reset(9)
    path = env.references[0]
    assert all(np.linalg.norm(path.sample(t)[1]) == pytest.approx(1.)
               for t in np.linspace(0, 40, 81))
    assert not np.array_equal(path.sample(12.)[0], path.sample(30.)[0])


def _tracking_config(*, late=False):
    base = IdealConfig()
    x = -4. if late else 0.
    starts = ((x,-2.5,20.), (x,0.,22.5), (x,2.5,20.)) + base.starts[3:]
    return replace(base, task="IK2", dt=1., max_steps=4, random_range=0.,
                   starts=starts, speeds=(1.,)*5,
                   target_starts=((0.,0.,20.), base.target_starts[1]),
                   target_ends=(((1. if late else 12.),0.,20.), base.target_ends[1]),
                   reference_end="stop" if late else "ping_pong",
                   tracking_success_fraction=.7)


def test_continuous_tracking_success_and_last_step_failure():
    for late in (False, True):
        env = IdealEnv(_tracking_config(late=late))
        env.reset()
        for _ in range(4):
            _, _, terminated, truncated, info = env.step(np.full((5,3), .5))
        assert terminated and not truncated and info["time"] == 4.
        fractions = [row["valid_fraction"] for row in info["tracking"]]
        if late:
            assert info["outcome"] == "tracking_failure"
            assert all(0 < value < .7 for value in fractions)
            assert all(row["instantaneous_error"] <= env.config.tracking_distance
                       for row in info["tracking"])
        else:
            assert info["outcome"] == "success"
            assert fractions == pytest.approx([1., 1., 1.])
            assert [row["mean_error"] for row in info["tracking"]] == pytest.approx([2.5]*3)


def test_tracking_integral_is_stable_under_time_step_refinement():
    coarse = _tracking_config()
    fine = replace(coarse, dt=.25, max_steps=16)
    results = []
    for config in (coarse, fine):
        env = IdealEnv(config)
        env.reset()
        for _ in range(config.max_steps):
            _,_,_,_,info = env.step(np.full((5,3),.5))
        results.append([(row["mean_error"], row["valid_fraction"]) for row in info["tracking"]])
    np.testing.assert_allclose(results[0], results[1], rtol=0, atol=1e-9)


def test_tracking_exit_and_recovery_intervals():
    base = IdealConfig()
    starts = ((0.,3.,20.), (0.,0.,23.), (0.,-3.,20.)) + base.starts[3:]
    config = replace(base, task="IK2", dt=1., max_steps=3, random_range=0.,
                     starts=starts, speeds=(2.,)*5, target_speed=1.,
                     target_starts=((0.,0.,20.),base.target_starts[1]),
                     target_ends=((12.,0.,20.),base.target_ends[1]),
                     tracking_distance=4., tracking_success_fraction=.5)
    env = IdealEnv(config)
    env.reset()
    actions = np.full((5,3), .5)
    actions[0] = [1.,.5,.5]
    env.step(actions)
    actions[0] = [.5,1.,.5]
    env.step(actions)
    actions[0] = [.5,0.,.5]
    _,_,_,_,info = env.step(actions)
    row = info["tracking"][0]
    assert 0 < row["valid_fraction"] < 1
    assert row["departure_count"] >= 1
    assert row["longest_valid_time"] > 0 and row["outside_time"] > 0


def test_safety_enter_stay_exit_reenter_and_midstep_crossing():
    base = IdealConfig()
    starts = ((-10.,0.,10.),(-4.,1.,10.)) + base.starts[2:]
    env = IdealEnv(replace(base, task="IK1", dt=2., random_range=0.,
                           starts=starts, speeds=(3.,)*5))
    env.reset()
    actions = np.full((5,3), .5)
    actions[0], actions[1] = [1.,.5,.5], [0.,.5,.5]
    _,_,terminated,_,info = env.step(actions)
    assert not terminated
    assert info["warning_count"] == info["safety_count"] == 1
    assert info["safety_duration"] > 0
    assert not env.warning_pairs and not env.safety_pairs
    assert np.linalg.norm(info["actual_states"][0,:3]-info["actual_states"][1,:3]) > 2.
    actions[0], actions[1] = [0.,.5,.5], [1.,.5,.5]
    _,_,_,_,info = env.step(actions)
    assert info["warning_count"] == info["safety_count"] == 2
    assert len(inside_intervals(np.array([-3.,0.,0.]),np.array([3.,0.,0.]),
                                np.array([3.,1.,0.]),np.array([-3.,0.,0.]),2.,2.)) == 1


def test_random_reset_retries_and_snapshot_identity(monkeypatch):
    base = IdealConfig()
    starts = ((0.,0.,10.),(2.5,2.5,12.5)) + base.starts[2:]
    config = replace(base, task="IK1", starts=starts, random_range=1., initialization="random")
    invalid_offset = np.array(((1.,1.,1.),(-1.,-1.,-1.),(0.,0.,0.)))
    assert np.linalg.norm(np.asarray(starts[0])-starts[1])-2 > config.safety_distance
    probe = IdealEnv(config)
    probe.reset(4)
    assert probe._initial_failure(np.asarray(starts[:3])+invalid_offset) == "UAV safety overlap"
    class ControlledRng:
        def __init__(self):
            self.calls = 0
        def uniform(self, _low, _high, _shape):
            self.calls += 1
            return invalid_offset if self.calls == 1 else np.zeros((3,3))
    original = np.random.default_rng
    with monkeypatch.context() as patch:
        patch.setattr(np.random, "default_rng", lambda _: ControlledRng())
        with pytest.raises(ValueError, match="1 attempts"):
            IdealEnv(replace(config, init_max_attempts=1)).reset(4)
        IdealEnv(replace(config, init_max_attempts=2)).reset(4)
    a, b = IdealEnv(config), IdealEnv(config)
    _, first = a.reset(4)
    _, second = b.reset(4)
    np.testing.assert_array_equal(first["actual_states"], second["actual_states"])
    snapshot = a.snapshot()
    b.reset(5)
    b.restore(snapshot)
    action = np.full((5,3), .5)
    first_step, second_step = a.step(action), b.step(action)
    np.testing.assert_array_equal(first_step[4]["actual_states"],
                                  second_step[4]["actual_states"])
    assert first_step[4]["events"] == second_step[4]["events"]
    with pytest.raises(ValueError, match="incompatible"):
        b.restore(snapshot | {"protocol": "uav-ideal-kinematic-v1.0.0"})
    with pytest.raises(ValueError, match="incompatible"):
        IdealEnv(replace(config, task="IK0")).restore(snapshot)
def test_deadline_failure_has_no_gae_bootstrap():
    env = IdealEnv(replace(IdealConfig(), max_steps=1))
    env.reset()
    _, reward, terminated, truncated, info = env.step(np.full((5,3), .5))
    assert info["outcome"] == "deadline_failure" and terminated and not truncated
    r = torch.tensor(reward[:1]).reshape(1,1,1)
    value, successor = torch.ones_like(r), torch.full_like(r, 100.)
    mask = torch.ones_like(r)
    advantage, _ = gae(r, value, successor, mask, mask,
                       torch.tensor([[terminated]]), torch.tensor([[truncated]]), .99, .95)
    torch.testing.assert_close(advantage, r-value)


def test_arrival_precedes_later_boundary_and_members_stop_separately():
    base = IdealConfig()
    waypoints = ((-13.,-12.,10.),(-20.,0.,15.),(-20.,12.,20.)) + base.waypoints[3:]
    config = replace(base, task="IK1", dt=3., upper=(-12.,30.,40.),
                     waypoints=waypoints)
    env = IdealEnv(config)
    env.reset()
    _,_,terminated,truncated,info = env.step(np.full((5,3),.5))
    assert terminated and not truncated and info["outcome"] == "boundary"
    assert info["time"] == pytest.approx(1.5)
    assert info["status"][0] == "completed"
    assert info["actual_states"][0,0] == pytest.approx(-14.5)
    assert info["actual_states"][1,0] == pytest.approx(-12.)
    assert [e["kind"] for e in info["events"]] == ["arrival","boundary","boundary"]
    assert info["events"][0]["time"] == pytest.approx(.25)
    assert all(e["time"] <= info["time"] for e in info["events"])


def test_earlier_contact_cancels_later_waypoint_arrival():
    base = IdealConfig()
    starts = ((-10.,-2.,10.),(-10.,2.,10.)) + base.starts[2:]
    waypoints = ((-10.,2.,10.),) + base.waypoints[1:]
    env = IdealEnv(replace(base, task="IK1", dt=2., random_range=0.,
                           starts=starts, waypoints=waypoints))
    env.reset()
    action = np.full((5,3),.5)
    action[0], action[1] = [.5,1.,.5], [.5,0.,.5]
    _,_,terminated,_,info = env.step(action)
    assert terminated and info["outcome"] == "contact"
    assert info["time"] == pytest.approx(.875)
    assert info["status"][0] == "active"
    assert not any(e["kind"] == "arrival" for e in info["events"])


def test_unassigned_moving_target_contact_is_safety_failure():
    base = IdealConfig()
    config = replace(base, task="IK3", dt=3.,
                     target_starts=(base.target_starts[0],(-12.,-12.,10.)),
                     target_ends=(base.target_ends[0],(-11.,-12.,10.)))
    env = IdealEnv(config)
    env.reset()
    assert env.assignment[0] == 0
    actions = np.full((5,3),.5)
    actions[0] = [1.,.5,.5]
    _,_,terminated,_,info = env.step(actions)
    assert terminated and info["outcome"] == "target_contact"
    assert any(e["members"] == [0,6] for e in info["events"] if e["kind"] == "target_contact")
