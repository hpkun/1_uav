"""Independent geometry and velocity-envelope regression tests for the opt-in ground profile."""
import json
from pathlib import Path

import numpy as np
import pytest

from env.interception import InterceptionConfig, InterceptionEnv
from env.interception.motion import command, project
from env.interception.protected_area import contains_intruder, intrusion_windows, navigation_point


PROFILE = Path(__file__).resolve().parents[1] / "configs" / "interception_ground_facility_v3.json"


def ground(**overrides):
    settings = json.loads(PROFILE.read_text(encoding="utf-8"))
    settings.update(overrides)
    if settings["scale"] == "1v1":
        settings["task"] = "S0"
    return InterceptionConfig(**settings)


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


def test_ground_facility_random_initialization_and_vertical_limits():
    config = ground()
    env = InterceptionEnv(config)
    _, info = env.reset(42, {"scenario": "random"})
    assert info["pursuers"].shape == (5, 6)
    assert info["evaders"].shape == (2, 6)
    assert not contains_intruder(info["evaders"][0, :3], config)
    assert np.max(np.abs(info["pursuers"][:3, 5])) <= config.pursuer_vertical_vmax
    assert np.max(np.abs(info["evaders"][:1, 5])) <= config.evader_vertical_vmax
    _, _, _, _, next_info = env.step(idle())
    assert np.max(np.abs(next_info["commands"][:3, 2])) <= config.pursuer_vertical_vmax
    assert abs(next_info["commands"][5, 2]) <= config.evader_vertical_vmax


def test_velocity_command_caps_only_vertical_component():
    v = command([[1., 1., 1.]], 12., 4.)[0]
    assert np.linalg.norm(v) <= 12. + 1e-12
    assert v[2] == pytest.approx(4.)
    assert project([0, 0, -1], 9., 3.)[2] == pytest.approx(-3.)


def test_horizontal_crossing_detected_continuously():
    config = ground(scale="1v1", dt=1., evader_mode="constant_velocity",
                    constant_direction=(1., 0., 0.))
    env = single(config, [40, 30, 15, 0, 0, 0], [-13, 0, 12, 9, 0, 0])
    _, _, terminated, truncated, info = env.step(idle())
    assert terminated and not truncated and info["outcome"] == "breach"
    assert info["time"] == pytest.approx(1/9, abs=1e-6)


def test_overflight_above_protected_altitude_is_not_breach():
    config = ground(scale="1v1", dt=1., evader_mode="constant_velocity",
                    constant_direction=(1., 0., 0.))
    env = single(config, [40, 30, 15, 0, 0, 0], [-13, 0, 28, 9, 0, 0])
    _, _, terminated, truncated, info = env.step(idle())
    assert not terminated and not truncated
    assert info["outcome"] == "ongoing"


def test_descent_through_roof_triggers_breach():
    config = ground(scale="1v1", dt=1., evader_mode="constant_velocity",
                    constant_direction=(0., 0., -1.))
    env = single(config, [40, 30, 15, 0, 0, 0], [0, 0, 24, 0, 0, -3])
    _, _, terminated, truncated, info = env.step(idle())
    assert terminated and not truncated and info["outcome"] == "breach"
    assert info["time"] == pytest.approx(2/3, abs=1e-6)


def test_initial_intrusion_and_bad_profiles_fail_early():
    config = ground(scale="1v1")
    with pytest.raises(ValueError, match="initial breach"):
        single(config, [40, 30, 15, 0, 0, 0], [0, 0, 12, 0, 0, 0])
    with pytest.raises(ValueError, match="ground cylinder"):
        ground(center=(0, 0, 15))
    with pytest.raises(ValueError, match="vertical"):
        ground(pursuer_vertical_vmax=13.)
