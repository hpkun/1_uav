from math import pi

import pytest

from uav_env.core.geometry import angle_between, euclidean_distance, normalize_angle


def test_euclidean_distance() -> None:
    assert euclidean_distance([0.0, 0.0, 0.0], [3.0, 4.0, 12.0]) == pytest.approx(13.0)


def test_angle_between_orthogonal_vectors() -> None:
    assert angle_between([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]) == pytest.approx(pi / 2.0)


@pytest.mark.parametrize("value,expected", [(0.0, 0.0), (pi, -pi), (3.0 * pi, -pi), (-3.0 * pi, -pi)])
def test_normalize_angle(value: float, expected: float) -> None:
    assert normalize_angle(value) == pytest.approx(expected)
