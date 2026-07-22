"""Control input definitions."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class ControlInput:
    """Dimensionless overload command and bank angle in radians."""

    tangential_overload: float
    normal_overload: float
    bank_angle: float

    def __post_init__(self) -> None:
        values = (self.tangential_overload, self.normal_overload, self.bank_angle)
        if not all(isfinite(value) for value in values):
            raise ValueError("ControlInput values must be finite")

    def to_vector(self) -> list[float]:
        """Return ``[nx, nz, gamma]``."""

        return [self.tangential_overload, self.normal_overload, self.bank_angle]
