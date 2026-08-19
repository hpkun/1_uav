"""Deterministic boundary-aware nearest-target pursuit policy."""
from __future__ import annotations

import numpy as np

from ..math_utils import wrap_angle
from ..models import AircraftState
from .arena import (
    arena_constrained_direction, horizontal_safety_severity,
    vertical_safety_severities,
)


class NearestTargetPursuitPolicy:
    """Minimal pursuit heuristic with horizontal and vertical arena safety."""

    def __init__(self, config: dict, battlefield: dict, action_config: dict) -> None:
        self.config = config
        self.battlefield = battlefield
        self.action_config = action_config
        self.radius = float(battlefield["horizontal_radius"])
        self.altitude_min = float(battlefield["altitude_min"])
        self.altitude_max = float(battlefield["altitude_max"])

    @staticmethod
    def nearest_target_index(own: AircraftState, targets: list[AircraftState]) -> int | None:
        candidates = []
        for index, target in enumerate(targets):
            if target.alive:
                distance = (target.x - own.x) ** 2 + (target.y - own.y) ** 2 + (target.z - own.z) ** 2
                candidates.append((distance, index))
        return min(candidates)[1] if candidates else None

    def desired_horizontal_direction(
        self, own: AircraftState, target: AircraftState
    ) -> np.ndarray:
        target_direction = np.array([target.x - own.x, target.y - own.y], dtype=float)
        return arena_constrained_direction(own, target_direction, self.battlefield)

    def recovery_speed(self, own: AircraftState, nominal_speed: float) -> float:
        severity = horizontal_safety_severity(own, self.battlefield)
        recovery = float(self.config["recovery_speed"])
        return float((1.0 - severity) * nominal_speed + severity * recovery)

    def action_toward(
        self,
        own: AircraftState,
        desired_heading: float,
        desired_elevation: float,
        desired_speed: float,
    ) -> np.ndarray:
        """Produce a normalized maneuver action; physical mapping remains in control.py."""
        cfg = self.config
        desired_delta_nz = cfg["pitch_load_gain"] * (desired_elevation - own.theta)
        return np.clip(np.array([
            (desired_speed - own.v) / cfg["speed_error_scale"],
            desired_delta_nz / self.action_config["nz_delta_scale"],
            cfg["heading_gain"] * wrap_angle(desired_heading - own.psi)
            / self.action_config["phi_max"],
        ], dtype=np.float32), -1.0, 1.0)

    def safe_action_toward(
        self,
        own: AircraftState,
        desired_heading: float,
        desired_elevation: float,
        desired_speed: float,
    ) -> np.ndarray:
        """Apply shared arena safety, then use the common normalized maneuver helper."""
        requested = np.array([np.cos(desired_heading), np.sin(desired_heading)], dtype=float)
        direction = arena_constrained_direction(own, requested, self.battlefield)
        safe_heading = float(np.arctan2(direction[1], direction[0]))
        lower, upper = vertical_safety_severities(own, self.battlefield)
        if lower > 0.0:
            desired_elevation = max(desired_elevation, 0.0)
        if upper > 0.0:
            desired_elevation = min(desired_elevation, 0.0)
        safe_speed = self.recovery_speed(own, desired_speed)
        return self.action_toward(
            own, safe_heading, desired_elevation, safe_speed
        )

    def action(self, own: AircraftState, targets: list[AircraftState]) -> np.ndarray:
        target_index = self.nearest_target_index(own, targets) if own.alive else None
        if target_index is None:
            return np.zeros(3, dtype=np.float32)
        target = targets[target_index]
        target_direction = np.array([target.x - own.x, target.y - own.y], dtype=float)
        desired_heading = np.arctan2(target_direction[1], target_direction[0])
        horizontal_distance = np.hypot(target.x - own.x, target.y - own.y)
        desired_elevation = float(np.arctan2(own.z - target.z, horizontal_distance))
        return self.safe_action_toward(
            own, desired_heading, desired_elevation, float(self.config["desired_speed"])
        )

    def team_actions(self, team: list[AircraftState], targets: list[AircraftState]) -> np.ndarray:
        return np.stack([self.action(state, targets) for state in team])


__all__ = ["NearestTargetPursuitPolicy"]
