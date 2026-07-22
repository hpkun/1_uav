from __future__ import annotations

import pytest

from conftest import make_state
from uav_env.actions.discrete_15 import DiscreteAction15, get_control
from uav_env.dynamics.propagation import propagate_state
from uav_env.entities.type_profiles import UAVTypeProfile


def test_level_hold_preserves_speed_and_altitude_for_ten_seconds(profile: UAVTypeProfile) -> None:
    state = make_state(profile)
    for _ in range(100):
        state = propagate_state(state, get_control(DiscreteAction15.LEVEL_HOLD), profile, 0.1)
    assert state.speed == pytest.approx(100.0, abs=1.0e-10)
    assert state.z == pytest.approx(1500.0, abs=1.0e-9)


def test_left_and_right_turns_are_mirrored(profile: UAVTypeProfile) -> None:
    left = make_state(profile)
    right = make_state(profile)
    for _ in range(20):
        left = propagate_state(left, get_control(DiscreteAction15.LEFT_HOLD), profile, 0.1)
        right = propagate_state(right, get_control(DiscreteAction15.RIGHT_HOLD), profile, 0.1)
    assert left.x == pytest.approx(right.x, rel=1.0e-12)
    assert left.y == pytest.approx(-right.y, rel=1.0e-12)
    assert left.z == pytest.approx(right.z, abs=1.0e-10)


def test_speed_limits_clip_acceleration_and_deceleration(profile: UAVTypeProfile) -> None:
    fast = make_state(profile, speed=profile.max_speed)
    slow = make_state(profile, speed=profile.min_speed)
    fast = propagate_state(fast, get_control(DiscreteAction15.LEVEL_ACCELERATE), profile, 1.0)
    slow = propagate_state(slow, get_control(DiscreteAction15.LEVEL_DECELERATE), profile, 1.0)
    assert fast.speed == profile.max_speed
    assert slow.speed == profile.min_speed
