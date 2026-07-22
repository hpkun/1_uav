"""Generic fourth-order Runge--Kutta integration."""

from __future__ import annotations

from typing import Callable, TypeVar

import numpy as np
from numpy.typing import NDArray

T = TypeVar("T")
Derivative = Callable[[float, NDArray[np.float64], T], NDArray[np.float64]]


def rk4_step(
    derivative: Derivative[T],
    time: float,
    state: NDArray[np.float64],
    dt: float,
    args: T,
) -> NDArray[np.float64]:
    """Advance an ODE by one fixed RK4 step."""

    y = np.asarray(state, dtype=np.float64)
    if not np.all(np.isfinite(y)) or not np.isfinite(time) or not np.isfinite(dt):
        raise ValueError("RK4 inputs must be finite")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    k1 = np.asarray(derivative(time, y, args), dtype=np.float64)
    k2 = np.asarray(derivative(time + dt / 2.0, y + dt * k1 / 2.0, args), dtype=np.float64)
    k3 = np.asarray(derivative(time + dt / 2.0, y + dt * k2 / 2.0, args), dtype=np.float64)
    k4 = np.asarray(derivative(time + dt, y + dt * k3, args), dtype=np.float64)
    for stage in (k1, k2, k3, k4):
        if stage.shape != y.shape or not np.all(np.isfinite(stage)):
            raise ValueError("Derivative must return a finite array matching state shape")
    return y + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
