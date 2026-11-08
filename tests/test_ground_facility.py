"""Independent geometry and velocity-envelope regression tests for the opt-in ground profile."""
import numpy as np
import pytest

from env.interception import InterceptionConfig, InterceptionEnv
from env.interception.motion import advance, execute_velocity
from env.interception.protected_area import contains_intruder, navigation_point


def ground(**overrides):
    return InterceptionConfig(**overrides)


def single(config, pursuer, evader):
    env = InterceptionEnv(config)
    env.reset(12, {"scenario": "custom", "pursuers": [pursuer], "evaders": [evader]})
    return env


def idle():
    return np.full((5, 3), .5)


def test_ground_facility_geometry_and_navigation():
    config = ground()
    assert navigation_point(config).tolist() == [0., 0., 12.]
    assert contains_intruder([3, 4, 18], config)
    assert contains_intruder([12, 0, 22], config)
    assert not contains_intruder([0, 0, 23], config)
    assert not contains_intruder([13, 0, 18], config)
    assert config.center[2] == 0
    assert config.protected_zone == "ground_cylinder"


@pytest.mark.parametrize("initialization", ["random", "fixed"])
def test_ground_initialization_reproducible_and_translated(initialization):
    config = ground()
    env = InterceptionEnv(config)
    _, info = env.reset(42, {"scenario": initialization})
    _, same = env.reset(42, {"scenario": initialization})
    shift = np.array([20, -15, -2])
    _, shifted = InterceptionEnv(ground(center=shift)).reset(42, {"scenario": initialization})
    for key, mask in (("pursuers", "active_mask"), ("evaders", "evader_mask")):
        np.testing.assert_array_equal(info[key], same[key])
        np.testing.assert_allclose(shifted[key][info[mask], :3], info[key][info[mask], :3]+shift)
        np.testing.assert_allclose(shifted[key][:, 3:], info[key][:, 3:])
    assert 8.-1e-10 <= np.linalg.norm(info["evaders"][0, 3:]) <= 10.+1e-10
    assert 100 <= info["initial_geometry"]["evaders"]["distance"][0] <= 130
    assert all(28-1e-10 <= d <= 42+1e-10 for d in info["initial_geometry"]["pursuers"]["horizontal_distance"])
    assert np.ptp(info["pursuers"][:3, 2]) > 0
    assert -2 <= info["evaders"][0, 5] <= 3


def test_velocity_and_horizontal_acceleration_throughout_held_step():
    config = ground()
    rng = np.random.default_rng(50)
    assert execute_velocity([0,0,99], np.zeros(3), 12, 2.5, config)[2] == 3.
    assert execute_velocity([0,0,-99], np.zeros(3), 12, 2.5, config)[2] == -2.
    for vmax in (12., 13.):
        for request, raw in zip(rng.normal(0,20,(150,3)), rng.normal(0,20,(150,3))):
            initial = execute_velocity(raw, np.zeros(3), vmax, 2.5, config)
            execute = execute_velocity(request, initial, vmax, 2.5, config)
            for time in (0., .001, .05, .1, 10.):
                actual = advance(np.r_[np.zeros(3), initial], execute, 2.5, time)[3:]
                assert np.linalg.norm(actual) <= vmax+1e-12
                assert -2.-1e-12 <= actual[2] <= 3.+1e-12
                assert np.linalg.norm(execute[:2]-actual[:2])/2.5 <= 4.+1e-12


def test_bezier_is_repeatable_and_integrates_commands():
    config = ground(evader_mode="bezier")
    env = InterceptionEnv(config)
    _, before = env.reset(42)
    snapshot = env.snapshot()
    _, _, _, _, info = env.step(idle())
    env.restore(snapshot)
    _, _, _, _, same = env.step(idle())
    np.testing.assert_array_equal(info["commands"], same["commands"])
    np.testing.assert_array_equal(info["evaders"][0],
                                  advance(before["evaders"][0], info["commands"][5], 2.5, .1))
    assert np.linalg.norm(info["initial_accelerations"][5,:2]) <= 4.+1e-12


def test_horizontal_crossing_detected_continuously():
    config = ground(scale="1v1", dt=1., evader_mode="constant_velocity",
                    constant_direction=(1., 0., 0.), evader_speed_fraction=9/13)
    env = single(config, [40, 30, 15, 0, 0, 0], [-13, 0, 12, 9, 0, 0])
    _, _, terminated, truncated, info = env.step(idle())
    assert terminated and not truncated and info["outcome"] == "breach"
    assert info["time"] == pytest.approx(1/9, abs=1e-6)


def test_overflight_above_protected_altitude_is_not_breach():
    config = ground(scale="1v1", dt=1., evader_mode="constant_velocity",
                    constant_direction=(1., 0., 0.), evader_speed_fraction=9/13)
    env = single(config, [40, 30, 15, 0, 0, 0], [-13, 0, 28, 9, 0, 0])
    _, _, terminated, truncated, info = env.step(idle())
    assert not terminated and not truncated
    assert info["outcome"] == "ongoing"


def test_descent_through_roof_triggers_breach():
    config = ground(scale="1v1", dt=1., evader_mode="constant_velocity",
                    constant_direction=(0., 0., -1.))
    env = single(config, [40, 30, 15, 0, 0, 0], [0, 0, 24, 0, 0, -2])
    _, _, terminated, truncated, info = env.step(idle())
    assert terminated and not truncated and info["outcome"] == "breach"
    assert info["time"] == pytest.approx(1., abs=1e-6)


def test_initial_intrusion_and_bad_profiles_fail_early():
    config = ground(scale="1v1")
    with pytest.raises(ValueError, match="initial breach"):
        single(config, [40, 30, 15, 0, 0, 0], [0, 0, 12, 0, 0, 0])
    with pytest.raises(ValueError, match="ground cylinder"):
        ground(center=(0, 0, 15))
    with pytest.raises(ValueError, match="navigation point"):
        ground(navigation_offset=(0, 0, 25))


def test_fractional_legacy_initial_speed_is_not_scaled_twice():
    config = InterceptionConfig(scale="1v1", evader_initial_speed=(0.5, 0.5))
    _, info = InterceptionEnv(config).reset(123, {"scenario": "random"})
    assert np.linalg.norm(info["evaders"][0, 3:]) == pytest.approx(0.5, abs=1e-12)
