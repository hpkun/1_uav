import numpy as np
import pytest

from uav_env.actions.discrete_15 import DiscreteAction15, get_control
from uav_env.dynamics.point_mass_3d import point_mass_3d_derivative
from uav_env.dynamics.rk4 import rk4_step


def test_level_hold_preserves_altitude_and_speed_briefly() -> None:
    control = get_control(DiscreteAction15.LEVEL_HOLD)
    initial = np.asarray([0.0, 0.0, 1_000.0, 100.0, 0.0, 0.0])

    def derivative(time: float, state: np.ndarray, command: object) -> np.ndarray:
        del time, command
        return point_mass_3d_derivative(state, control)

    final = rk4_step(derivative, 0.0, initial, 0.1, control)
    assert final[2] == pytest.approx(initial[2], abs=1.0e-10)
    assert final[3] == pytest.approx(initial[3], abs=1.0e-10)


@pytest.mark.parametrize("speed,theta", [(100.0, 0.0), (0.0, 0.0), (100.0, np.pi / 2.0)])
def test_derivative_is_finite_with_protected_denominators(speed: float, theta: float) -> None:
    state = np.asarray([0.0, 0.0, 0.0, speed, theta, 0.0])
    result = point_mass_3d_derivative(state, get_control(DiscreteAction15.LEFT_HOLD))
    assert np.all(np.isfinite(result))
