"""UAV state definitions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite

import numpy as np
from numpy.typing import NDArray


@dataclass
class UAVState:
    """Physical and bookkeeping state for one UAV.

    Angles are radians, position is metres, and speed is m/s.
    """

    x: float
    y: float
    z: float
    speed: float
    flight_path_angle: float
    heading_angle: float
    health: float
    alive: bool
    team_id: int
    type_id: str

    def __post_init__(self) -> None:
        numeric = (
            self.x,
            self.y,
            self.z,
            self.speed,
            self.flight_path_angle,
            self.heading_angle,
            self.health,
        )
        if not all(isfinite(value) for value in numeric):
            raise ValueError("UAVState numeric values must be finite")
        if not self.type_id:
            raise ValueError("type_id must not be empty")

    def to_kinematic_vector(self) -> NDArray[np.float64]:
        """Return ``[x, y, z, speed, flight_path_angle, heading_angle]``."""

        return np.asarray(
            [
                self.x,
                self.y,
                self.z,
                self.speed,
                self.flight_path_angle,
                self.heading_angle,
            ],
            dtype=np.float64,
        )

    def copy(self) -> "UAVState":
        """Return an independent shallow copy of this scalar data object."""

        return replace(self)

    def with_kinematic_vector(self, vector: NDArray[np.float64]) -> "UAVState":
        """Return a copy whose six kinematic fields come from *vector*."""

        values = np.asarray(vector, dtype=np.float64)
        if values.shape != (6,):
            raise ValueError("Kinematic state must have shape (6,)")
        return replace(
            self,
            x=float(values[0]),
            y=float(values[1]),
            z=float(values[2]),
            speed=float(values[3]),
            flight_path_angle=float(values[4]),
            heading_angle=float(values[5]),
        )
