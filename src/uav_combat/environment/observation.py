"""Equation (24) natural 45-dimensional observation expansion."""
from __future__ import annotations

import numpy as np
from .geometry import compute_paper_geometry
from .sensor import ObservedState

OBSERVATION_DIM = 45


def _body_relative(observer: ObservedState, other: ObservedState) -> np.ndarray:
    delta = np.array([other.x - observer.x, other.y - observer.y, other.z - observer.z], dtype=float)
    c, s = np.cos(observer.psi), np.sin(observer.psi)
    return np.array([c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1], delta[2]], dtype=float)


def build_observation(
    observer_index: int,
    red: list[ObservedState],
    blue: list[ObservedState],
    red_alive: list[bool],
    blue_alive: list[bool],
    position_scale: float = 5000.0,
    speed_scale: float = 300.0,
) -> np.ndarray:
    """Build fixed ID-ordered slots; dead aircraft slots are all zero."""
    if len(red) != 4 or len(blue) != 4:
        raise ValueError("Equation (24) expansion requires 4v4")
    own = red[observer_index]
    values: list[float] = []
    if red_alive[observer_index]:
        # Own absolute position is retained for boundary observability; scaling
        # is an explicitly documented reproduction assumption.
        values.extend([own.x / position_scale, own.y / position_scale, own.z / position_scale, own.v / speed_scale, own.phi, own.psi, own.theta])
    else:
        values.extend([0.0] * 7)
    for i, friendly in enumerate(red):
        if i == observer_index:
            continue
        if red_alive[i] and red_alive[observer_index]:
            rel = _body_relative(own, friendly) / position_scale
            values.extend([*rel, friendly.v / speed_scale, friendly.psi, friendly.theta])
        else:
            values.extend([0.0] * 6)
    own_state = own.as_aircraft_state(red_alive[observer_index])
    for j, enemy in enumerate(blue):
        if blue_alive[j] and red_alive[observer_index]:
            geometry = compute_paper_geometry(own_state, enemy.as_aircraft_state(True))
            values.extend([geometry.distance / position_scale, enemy.v / speed_scale, geometry.aa, geometry.ata, geometry.ha])
        else:
            values.extend([0.0] * 5)
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (OBSERVATION_DIM,) or not np.all(np.isfinite(result)):
        raise RuntimeError(f"invalid Equation (24) observation: {result.shape}")
    return result
