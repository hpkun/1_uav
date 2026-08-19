"""Shared pure mathematics for the cylindrical hard/soft arena."""
from __future__ import annotations

import numpy as np

from ..models import AircraftState


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else np.zeros_like(vector)


def horizontal_radius_fraction(state: AircraftState, battlefield: dict) -> float:
    return float(np.hypot(state.x, state.y) / float(battlefield["horizontal_radius"]))


def horizontal_safety_severity(state: AircraftState, battlefield: dict) -> float:
    fraction = horizontal_radius_fraction(state, battlefield)
    soft = float(battlefield["horizontal_soft_fraction"])
    return float(np.clip((fraction - soft) / (1.0 - soft), 0.0, 1.0))


def vertical_safety_severities(
    state: AircraftState, battlefield: dict
) -> tuple[float, float]:
    altitude = state.altitude
    minimum = float(battlefield["altitude_min"])
    maximum = float(battlefield["altitude_max"])
    margin = float(battlefield["vertical_soft_margin"])
    lower = float(np.clip((minimum + margin - altitude) / margin, 0.0, 1.0))
    upper = float(np.clip((altitude - (maximum - margin)) / margin, 0.0, 1.0))
    return lower, upper


def vertical_safety_severity(state: AircraftState, battlefield: dict) -> float:
    return max(vertical_safety_severities(state, battlefield))


def boundary_cost(state: AircraftState, battlefield: dict) -> float:
    horizontal = horizontal_safety_severity(state, battlefield)
    vertical = vertical_safety_severity(state, battlefield)
    return max(horizontal * horizontal, vertical * vertical)


def boundary_cause(state: AircraftState, battlefield: dict) -> str | None:
    if np.hypot(state.x, state.y) > float(battlefield["horizontal_radius"]):
        return "horizontal"
    if state.altitude < float(battlefield["altitude_min"]):
        return "altitude_low"
    if state.altitude > float(battlefield["altitude_max"]):
        return "altitude_high"
    return None


def arena_constrained_direction(
    state: AircraftState, desired_direction: np.ndarray, battlefield: dict
) -> np.ndarray:
    """Apply barrier-style inward correction only in the horizontal soft zone."""
    target_direction = _unit(np.asarray(desired_direction, dtype=float))
    severity = horizontal_safety_severity(state, battlefield)
    if severity <= 0.0:
        return target_direction
    outward = _unit(np.array([state.x, state.y], dtype=float))
    center_direction = -outward
    raw = target_direction + (1.0 + severity) * center_direction
    constrained = _unit(raw)
    return center_direction if np.linalg.norm(constrained) <= 1e-12 else constrained


__all__ = [
    "arena_constrained_direction", "boundary_cause", "boundary_cost",
    "horizontal_radius_fraction", "horizontal_safety_severity",
    "vertical_safety_severities", "vertical_safety_severity",
]
