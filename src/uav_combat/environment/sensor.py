"""Sensor observation equations (3)-(5)."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..models import AircraftState


@dataclass(frozen=True)
class ObservedState:
    x: float; y: float; z: float; v: float; phi: float; psi: float; theta: float

    def as_aircraft_state(self, alive: bool = True) -> AircraftState:
        return AircraftState(self.x, self.y, self.z, self.v, self.theta, self.psi, alive)


class SensorModel:
    """Implement the shared-noise terms exactly as printed in Eqs. (3)-(5)."""

    def __init__(self, c1: float, c2: float, c3: float, b1: float, b2: float, b3: float, enabled: bool = True) -> None:
        if min(c1, c2, c3, b1, b2, b3) < 0:
            raise ValueError("sensor scales and clipping bounds must be nonnegative")
        self.c1, self.c2, self.c3 = map(float, (c1, c2, c3))
        self.b1, self.b2, self.b3 = map(float, (b1, b2, b3))
        self.enabled = bool(enabled)

    def observe(self, state: AircraftState, phi: float, rng: np.random.Generator) -> ObservedState:
        if not self.enabled:
            return ObservedState(state.x, state.y, state.z, state.v, phi, state.psi, state.theta)
        e1 = float(np.clip(rng.normal(), -self.b1, self.b1))
        e2 = float(np.clip(rng.normal(), -self.b2, self.b2))
        e3 = float(np.clip(rng.normal(), -self.b3, self.b3))
        return ObservedState(
            state.x + self.c1 * e1, state.y + self.c1 * e1, state.z + self.c1 * e1,
            state.v + self.c3 * e3, phi + self.c2 * e2, state.psi + self.c2 * e2,
            state.theta + self.c2 * e2,
        )
