"""Deterministic nearest-alive-target pure-pursuit policy."""
from __future__ import annotations

import numpy as np

from ..math_utils import wrap_angle
from ..models import AircraftState


class NearestTargetPursuitPolicy:
    def __init__(self, config: dict) -> None:
        self.config = config

    @staticmethod
    def nearest_target_index(own: AircraftState, targets: list[AircraftState]) -> int | None:
        candidates = []
        for index, target in enumerate(targets):
            if target.alive:
                distance = (target.x - own.x) ** 2 + (target.y - own.y) ** 2 + (target.z - own.z) ** 2
                candidates.append((distance, index))
        return min(candidates)[1] if candidates else None

    def action(self, own: AircraftState, targets: list[AircraftState]) -> np.ndarray:
        target_index = self.nearest_target_index(own, targets) if own.alive else None
        if target_index is None:
            return np.zeros(3, dtype=np.float32)
        target = targets[target_index]
        dx, dy, dz = target.x - own.x, target.y - own.y, target.z - own.z
        desired_heading = np.arctan2(dy, dx)
        horizontal_distance = np.hypot(dx, dy)
        desired_elevation = np.arctan2(-dz, horizontal_distance)
        cfg = self.config
        return np.clip(np.array([
            (cfg["desired_speed"] - own.v) / cfg["speed_error_scale"],
            cfg["elevation_gain"] * (desired_elevation - own.theta) / cfg["elevation_action_scale"],
            cfg["heading_gain"] * wrap_angle(desired_heading - own.psi) / (np.pi / 3.0),
        ], dtype=np.float32), -1.0, 1.0)

    def team_actions(self, team: list[AircraftState], targets: list[AircraftState]) -> np.ndarray:
        return np.stack([self.action(state, targets) for state in team])


__all__ = ["NearestTargetPursuitPolicy"]
