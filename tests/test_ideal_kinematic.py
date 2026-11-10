from dataclasses import replace
import numpy as np
import pytest

from env.ideal_kinematic import IdealConfig, IdealEnv
from env.ideal_kinematic.geometry import boundary, closest, direction, proximity
from env.ideal_kinematic.reference import Reference
from env.ideal_kinematic.config import OBS_DIM, STATE_DIM


def test_ideal_motion_and_zero_direction():
    config = IdealConfig(dt=.25, speeds=(3., 2., 2., 2., 2.))
    env = IdealEnv(config)
    _, info = env.reset(42)
    initial = info["actual_states"][0].copy()
    action = np.full((5, 3), .5)
    _, _, _, _, info = env.step(action)
    np.testing.assert_allclose(info["actual_states"][0, :3], initial[:3]+[.75, 0, 0])
    np.testing.assert_allclose(info["actual_states"][0, 3:], [3., 0., 0.])
    assert info["time"] == .25
    action[0] = [.5, 1., .5]
    _, _, _, _, info = env.step(action)
    np.testing.assert_allclose(info["actual_states"][0, 3:], [0., 3., 0.])
    np.testing.assert_allclose(info["actual_states"][0, :3], initial[:3]+[.75, .75, 0.])
    assert np.linalg.norm(info["actual_states"][0, 3:]) == 3.
    np.testing.assert_allclose(direction([.5, .5, .5], [1., 0., 0.]), [1., 0., 0.])


@pytest.mark.parametrize("task,count,targets", [("IK0",1,0),("IK1",3,0),("IK2",3,1),("IK3",5,2)])
def test_tasks_determinism_dimensions_and_mask(task, count, targets):
    config = IdealConfig(task=task, initialization="random")
    a, b = IdealEnv(config), IdealEnv(config)
    oa, ia = a.reset(49)
    ob, ib = b.reset(49)
    np.testing.assert_array_equal(oa, ob)
    np.testing.assert_array_equal(ia["actual_states"], ib["actual_states"])
    assert oa.shape == (5, OBS_DIM) and ia["centralized_state"].shape == (STATE_DIM,)
    assert ia["active_mask"].sum() == count and len(ia["assignment"]) == count
    assert np.count_nonzero(np.linalg.norm(ia["reference"][:, :3], axis=1)) == targets
    actions = np.full((5, 3), .7)
    for _ in range(3):
        _, ra, ta, xa, ia = a.step(actions)
        _, rb, tb, xb, ib = b.step(actions)
        np.testing.assert_array_equal(ra, rb)
        np.testing.assert_array_equal(ia["actual_states"], ib["actual_states"])
        assert (ta, xa) == (tb, xb)
        if ta or xa:
            break


def test_config_and_initial_geometry_rejection():
    with pytest.raises(ValueError, match="fixed speeds"):
        IdealConfig(speeds=(0., 2., 2., 2., 2.))
    with pytest.raises(ValueError, match="initial directions"):
        IdealConfig(directions=((0., 0., 0.),)*5)
    with pytest.raises(ValueError, match="overlap"):
        IdealConfig(task="IK1", starts=((-15., 0., 10.),)*5)
    with pytest.raises(ValueError, match="invalid dt"):
        IdealConfig(dt=float("nan"))


def test_continuous_crossings_and_rotation():
    p, q = np.array([-2., 0., 0.]), np.array([2., 0., 0.])
    v, w = np.array([2., 0., 0.]), np.array([-2., 0., 0.])
    assert proximity(p, v, q, w, .5, 2.) == pytest.approx(.875)
    assert closest(p, v, q, w, 2.) == 0.
    assert boundary(np.array([0., 0., 0.]), np.array([4., 0., 0.]),
                    np.array([-1., -1., -1.]), np.array([1., 1., 1.]), 1.) == .25
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    assert proximity(rotation@p, rotation@v, rotation@q, rotation@w, .5, 2.) == pytest.approx(.875)
    np.testing.assert_allclose(rotation@(p+v*.3), rotation@p+(rotation@v)*.3)


def test_bezier_reference_has_constant_segment_speed():
    path = Reference((-10., 0., 10.), (10., 0., 10.), 2., "bezier")
    for time in np.linspace(0., path.duration-.01, 50):
        assert np.linalg.norm(path.sample(time)[1]) == pytest.approx(2., rel=1e-12)
    np.testing.assert_allclose(path.sample(path.duration)[1], [0., 0., 0.])


def test_early_exit_and_continuous_boundary_event():
    config = IdealConfig(task="IK1", dt=1., arrival_distance=1.,
        waypoints=((-13., -12., 10.), (15., 0., 15.), (15., 12., 20.),
                   (15., -6., 25.), (15., 6., 30.)))
    env = IdealEnv(config)
    env.reset()
    action = np.full((5, 3), .5)
    action[0] = [1., .5, .5]
    _, _, term, trunc, info = env.step(action)
    assert not term and not trunc and not info["active_mask"][0]
    assert info["status"][0] == "completed"
    assert np.linalg.norm(info["actual_states"][0, 3:]) == 0
    assert any(event["kind"] == "arrival" and event["time"] == pytest.approx(.5)
               for event in info["events"])
    config = replace(IdealConfig(), dt=2., starts=((27., -12., 10.),) + IdealConfig().starts[1:])
    env = IdealEnv(config)
    env.reset()
    _, _, term, _, info = env.step(np.full((5,3), [.9,.5,.5]))
    assert term and info["outcome"] == "boundary"
    assert info["events"][0]["time"] == pytest.approx(1.5)


def test_pair_contact_occurs_inside_step_with_clear_endpoints():
    base = IdealConfig()
    starts = ((-10., -2., 10.), (-10., 2., 10.)) + base.starts[2:]
    env = IdealEnv(replace(base, task="IK1", dt=2., random_range=0., starts=starts))
    env.reset()
    actions = np.full((5, 3), .5)
    actions[0] = [.5, 1., .5]
    actions[1] = [.5, 0., .5]
    _, _, _, _, info = env.step(actions)
    assert np.linalg.norm(info["actual_states"][0, :3] -
                          info["actual_states"][1, :3]) == pytest.approx(4.)
    assert any(e["kind"] == "contact" and e["time"] == pytest.approx(.875)
               for e in info["events"])
    assert info["minimum_separation"] == pytest.approx(0.)


def test_moving_target_contact_detected_across_reference_breakpoint():
    base = IdealConfig()
    config = replace(base, task="IK2", dt=3.,
                     target_starts=((-12., -12., 10.), base.target_starts[1]),
                     target_ends=((-11., -12., 10.), base.target_ends[1]))
    env = IdealEnv(config)
    env.reset()
    actions = np.full((5, 3), .5)
    actions[0] = [1., .5, .5]
    _, _, terminated, _, info = env.step(actions)
    assert terminated and info["outcome"] == "contact"
    assert any(e["kind"] == "target_contact" and e["time"] == pytest.approx(1.75)
               for e in info["events"])
