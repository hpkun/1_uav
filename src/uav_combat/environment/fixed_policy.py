"""Deterministic boundary-aware nearest-target pursuit policy."""
from __future__ import annotations

import numpy as np

from ..math_utils import wrap_angle
from ..models import AircraftState


class NearestTargetPursuitPolicy:
    """Minimal pursuit heuristic with horizontal and vertical arena safety."""

    def __init__(self, config: dict, battlefield: dict) -> None:
        self.config = config
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

    @staticmethod
    def _unit(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1e-12 else np.zeros_like(vector)

    def desired_horizontal_direction(
        self, own: AircraftState, target: AircraftState
    ) -> np.ndarray:
        target_direction = self._unit(np.array([target.x - own.x, target.y - own.y], dtype=float))
        horizontal_radius = float(np.hypot(own.x, own.y))
        start = float(self.config["boundary_recovery_start_fraction"])
        if horizontal_radius <= start * self.radius:
            return target_direction
        center_direction = self._unit(np.array([-own.x, -own.y], dtype=float))
        weight = float(np.clip(
            (horizontal_radius / self.radius - start) / (1.0 - start), 0.0, 1.0
        ))
        blended = (1.0 - weight) * target_direction + weight * center_direction
        direction = self._unit(blended)
        if np.linalg.norm(direction) <= 1e-12:
            direction = center_direction if weight >= 0.5 else target_direction
        return direction

    def action(self, own: AircraftState, targets: list[AircraftState]) -> np.ndarray:
        target_index = self.nearest_target_index(own, targets) if own.alive else None
        if target_index is None:
            return np.zeros(3, dtype=np.float32)
        target = targets[target_index]
        direction = self.desired_horizontal_direction(own, target)
        desired_heading = np.arctan2(direction[1], direction[0])
        horizontal_distance = np.hypot(target.x - own.x, target.y - own.y)
        desired_elevation = float(np.arctan2(own.z - target.z, horizontal_distance))
        margin = float(self.config["vertical_safety_margin"])
        if own.altitude <= self.altitude_min + margin:
            desired_elevation = max(desired_elevation, 0.0)
        if own.altitude >= self.altitude_max - margin:
            desired_elevation = min(desired_elevation, 0.0)
        cfg = self.config
        return np.clip(np.array([
            (cfg["desired_speed"] - own.v) / cfg["speed_error_scale"],
            cfg["elevation_gain"] * (desired_elevation - own.theta) / cfg["elevation_action_scale"],
            cfg["heading_gain"] * wrap_angle(desired_heading - own.psi) / (np.pi / 3.0),
        ], dtype=np.float32), -1.0, 1.0)

    def team_actions(self, team: list[AircraftState], targets: list[AircraftState]) -> np.ndarray:
        return np.stack([self.action(state, targets) for state in team])


__all__ = ["NearestTargetPursuitPolicy"]
