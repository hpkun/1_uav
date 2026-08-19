"""Deterministic nearest-target pure-pursuit policy."""
from __future__ import annotations

import numpy as np

from ..math_utils import wrap_angle
from ..models import AircraftState


class NearestTargetPursuitPolicy:
    """Minimal nearest-alive-target pure pursuit without map-management logic."""

    def __init__(self, config: dict, action_config: dict) -> None:
        self.config = config
        self.action_config = action_config

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
        norm = float(np.linalg.norm(target_direction))
        return target_direction / norm if norm > 1e-12 else np.zeros(2, dtype=float)

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

    def action(self, own: AircraftState, targets: list[AircraftState]) -> np.ndarray:
        target_index = self.nearest_target_index(own, targets) if own.alive else None
        if target_index is None:
            return np.zeros(3, dtype=np.float32)
        target = targets[target_index]
        direction = self.desired_horizontal_direction(own, target)
        desired_heading = np.arctan2(direction[1], direction[0])
        horizontal_distance = np.hypot(target.x - own.x, target.y - own.y)
        desired_elevation = float(np.arctan2(own.z - target.z, horizontal_distance))
        return self.action_toward(
            own, desired_heading, desired_elevation, float(self.config["desired_speed"])
        )

    def team_actions(self, team: list[AircraftState], targets: list[AircraftState]) -> np.ndarray:
        return np.stack([self.action(state, targets) for state in team])


__all__ = ["NearestTargetPursuitPolicy"]
