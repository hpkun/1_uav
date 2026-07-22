"""Geometry helpers for three-dimensional flight."""

from __future__ import annotations

from math import atan2, cos, pi, sin
from typing import Sequence, Union

import numpy as np
from numpy.typing import NDArray

VectorLike = Union[Sequence[float], NDArray[np.float64]]


def _vector3(value: VectorLike) -> NDArray[np.float64]:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("Expected a finite three-dimensional vector")
    return vector


def relative_position(source: VectorLike, target: VectorLike) -> NDArray[np.float64]:
    """Return the vector from *source* to *target*."""

    return _vector3(target) - _vector3(source)


def euclidean_distance(first: VectorLike, second: VectorLike) -> float:
    """Return three-dimensional Euclidean distance in metres."""

    return float(np.linalg.norm(relative_position(first, second)))


def velocity_vector(speed: float, flight_path_angle: float, heading_angle: float) -> NDArray[np.float64]:
    """Convert scalar speed and flight angles to a Cartesian velocity."""

    horizontal = speed * cos(flight_path_angle)
    return np.asarray(
        [horizontal * cos(heading_angle), horizontal * sin(heading_angle), speed * sin(flight_path_angle)],
        dtype=np.float64,
    )


def angle_between(first: VectorLike, second: VectorLike) -> float:
    """Return the unsigned angle between two non-zero vectors."""

    a = _vector3(first)
    b = _vector3(second)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 0.0:
        raise ValueError("Angle is undefined for a zero vector")
    cosine = float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))
    return float(np.arccos(cosine))


def attack_angle(attacker_velocity: VectorLike, attacker_to_target: VectorLike) -> float:
    """Return angle between attacker velocity and line of sight to target."""

    return angle_between(attacker_velocity, attacker_to_target)


def escape_angle(target_velocity: VectorLike, attacker_to_target: VectorLike) -> float:
    """Return angle between target velocity and line of sight away from attacker."""

    return angle_between(target_velocity, attacker_to_target)


def normalize_angle(angle: float) -> float:
    """Normalize an angle to the half-open interval ``[-pi, pi)``."""

    if not np.isfinite(angle):
        raise ValueError("Angle must be finite")
    return float((angle + pi) % (2.0 * pi) - pi)


def safe_atan2(y: float, x: float) -> float:
    """Return ``atan2(y, x)``, defining the zero-vector result as zero."""

    if not np.isfinite(y) or not np.isfinite(x):
        raise ValueError("atan2 inputs must be finite")
    if x == 0.0 and y == 0.0:
        return 0.0
    return atan2(y, x)
