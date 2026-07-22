import numpy as np

from uav_env.dynamics.rk4 import rk4_step


def test_rk4_exponential_growth() -> None:
    def derivative(time: float, state: np.ndarray, rate: float) -> np.ndarray:
        del time
        return rate * state

    result = rk4_step(derivative, 0.0, np.asarray([1.0]), 0.1, 1.0)
    assert result[0] == pytest.approx(np.exp(0.1), rel=1.0e-6)


import pytest
