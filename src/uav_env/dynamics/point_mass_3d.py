"""Three-degree-of-freedom overload point-mass dynamics."""

from __future__ import annotations

from math import cos, sin
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from uav_env.core.constants import GRAVITY_MPS2, NUMERICAL_EPSILON
from uav_env.core.control import ControlInput


def point_mass_3d_derivative(
    state: Sequence[float] | NDArray[np.float64],
    control: ControlInput,
    gravity: float = GRAVITY_MPS2,
    minimum_denominator: float = NUMERICAL_EPSILON,
) -> NDArray[np.float64]:
    """Return derivatives for ``[x, y, z, v, theta, psi]``.

    The speed and ``cos(theta)`` denominators retain their sign while being
    bounded away from zero for numerical safety.
    """

    values = np.asarray(state, dtype=np.float64)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("State must be a finite vector with shape (6,)")
    if not np.isfinite(gravity) or gravity <= 0.0:
        raise ValueError("gravity must be finite and positive")
    if not np.isfinite(minimum_denominator) or minimum_denominator <= 0.0:
        raise ValueError("minimum_denominator must be finite and positive")

    _, _, _, speed, theta, psi = values
    cos_theta = cos(float(theta))
    safe_speed = float(speed) if abs(speed) >= minimum_denominator else (-minimum_denominator if speed < 0 else minimum_denominator)
    safe_cos_theta = cos_theta if abs(cos_theta) >= minimum_denominator else (-minimum_denominator if cos_theta < 0 else minimum_denominator)
    nx = control.tangential_overload
    nz = control.normal_overload
    gamma = control.bank_angle

    derivative = np.asarray(
        [
            speed * cos_theta * cos(float(psi)),
            speed * cos_theta * sin(float(psi)),
            speed * sin(float(theta)),
            gravity * (nx - sin(float(theta))),
            gravity / safe_speed * (nz * cos(gamma) - cos_theta),
            gravity * nz * sin(gamma) / (safe_speed * safe_cos_theta),
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(derivative)):
        raise FloatingPointError("Dynamics produced a non-finite derivative")
    return derivative
