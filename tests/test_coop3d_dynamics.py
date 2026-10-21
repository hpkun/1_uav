import math
import numpy as np
import pytest
from coop3d import EnvConfig
from coop3d.dynamics import derivatives, physical_actions, velocity_vector, wrap_angle
from coop3d.integrator import rk4


def integrate(state, control, config=None):
    counters = dict(speed_clips=0, gamma_clips=0, nonfinite_states=0)
    result, finite = rk4(np.array(state, dtype=float), control, config or EnvConfig(), counters)
    assert finite
    return result, counters


def test_level_flight():
    state, counts = integrate([0, 0, 3000, 200, 0, 0], [0, 0, 1])
    np.testing.assert_allclose(state, [50, 0, 3000, 200, 0, 0], atol=1e-12)
    assert not any(counts.values())


@pytest.mark.parametrize('load,expected', [(2, 4.905), (-1, -2.4525)])
def test_acceleration(load, expected):
    result, _ = integrate([0, 0, 3000, 200, 0, 0], [0, load, 1])
    assert result[3] == pytest.approx(200 + expected)


@pytest.mark.parametrize('roll,sign', [(-math.pi / 3, -1), (math.pi / 3, 1)])
def test_turn(roll, sign):
    result, _ = integrate([0, 0, 3000, 200, 0, 0], [roll, 0, 2])
    assert result[5] * sign > 0
    assert abs(result[4]) < 1e-12  # normal*cos(roll)=1
    assert result[1] * sign > 0


@pytest.mark.parametrize('normal,sign', [(2, 1), (0, -1)])
def test_climb_dive(normal, sign):
    result, _ = integrate([0, 0, 3000, 200, 0, 0], [0, 0, normal])
    assert (result[2] - 3000) * sign > 0
    assert result[4] * sign > 0


def test_velocity_vector():
    state = np.array([10, 20, 30, 230, .4, -.8])
    v = velocity_vector(state)
    assert np.linalg.norm(v) == pytest.approx(230)
    np.testing.assert_allclose(v, derivatives(state, [0, 0, 1], EnvConfig())[:3])
    batch = np.broadcast_to(state, (2, 3, 6))
    assert velocity_vector(batch).shape == (2, 3, 3)
    np.testing.assert_allclose(velocity_vector(batch), np.broadcast_to(v, (2, 3, 3)))


@pytest.mark.parametrize('speed,load,expected', [(399.9, 2, 400), (100.1, -1, 100)])
def test_speed_clipping(speed, load, expected):
    result, counters = integrate([0, 0, 3000, speed, 0, 0], [0, load, 1])
    assert result[3] == expected
    assert counters['speed_clips'] == 1


def test_rk4_analytic_circle():
    state = np.array([0., 0., 3000., 200., 0., 0.])
    control = [math.pi / 3, 0, 2]
    dt = 1.0
    result, _ = integrate(state, control, EnvConfig(dt=dt))
    omega = 9.81 * math.sqrt(3) / 200
    expected = [200 / omega * math.sin(omega * dt), 200 / omega * (1 - math.cos(omega * dt))]
    np.testing.assert_allclose(result[:2], expected, atol=4e-6)


def test_angle_wrap():
    result, _ = integrate([0, 0, 3000, 200, 0, math.pi - .001], [math.pi / 3, 0, 2])
    assert -math.pi <= result[5] < -3
    assert wrap_angle(3 * math.pi) == pytest.approx(-math.pi)


def test_singular_derivative_and_gamma_clip():
    c = EnvConfig()
    assert np.isfinite(derivatives([0, 0, 3000, 0, math.pi / 2, 0], [1, 1, 8], c)).all()
    state, counts = integrate([0, 0, 3000, 200, c.gamma_limit, 0], [0, 2, 8])
    assert state[4] == c.gamma_limit
    assert counts['gamma_clips'] == 1


def test_nonfinite_protection():
    counters = dict(speed_clips=0, gamma_clips=0, nonfinite_states=0)
    with pytest.raises(FloatingPointError):
        rk4([0, 0, 3000, np.nan, 0, 0], [0, 0, 1], EnvConfig(), counters)
    assert counters['nonfinite_states'] == 1
    original = np.array([0., 0, 3000, 200, 0, 0])
    result, finite = rk4(original, [0, np.inf, 1], EnvConfig(), counters)
    assert not finite
    np.testing.assert_array_equal(result, original)
    assert counters['nonfinite_states'] == 2


def test_action_mapping():
    np.testing.assert_allclose(physical_actions([[0, 0, 0], [1, 1, 1]]), [[-math.pi, -1, 0], [math.pi, 2, 8]])
