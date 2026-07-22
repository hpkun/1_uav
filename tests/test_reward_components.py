from __future__ import annotations

from math import pi

import pytest

from uav_env.rewards.components import angle_reward, dense_reward, height_reward, piecewise_distance_reward, speed_reward


def test_angle_and_speed_rewards() -> None:
    assert angle_reward(0.0, 0.0) == 1.0
    assert angle_reward(pi, 0.0) == 0.0
    assert speed_reward(151.0, 100.0) == 0.1
    assert speed_reward(120.0, 100.0) == 1.0
    assert speed_reward(90.0, 100.0) == pytest.approx(0.5)
    assert speed_reward(70.0, 100.0) == 0.0


@pytest.mark.parametrize("boundary", [40.0, 900.0, 1300.0, 5000.0])
def test_distance_reward_is_continuous_at_boundaries(boundary: float) -> None:
    epsilon = 1.0e-6
    left = piecewise_distance_reward(boundary - epsilon, 40.0, 900.0, 1300.0, 5000.0)
    right = piecewise_distance_reward(boundary + epsilon, 40.0, 900.0, 1300.0, 5000.0)
    assert abs(left - right) < 1.0e-5


def test_height_and_dense_reward_are_finite() -> None:
    assert height_reward(100.0) == 1.0
    assert height_reward(300.0) == 1.0
    assert dense_reward(1.0, 1.0, 1.0, 1.0) == pytest.approx(0.0)
