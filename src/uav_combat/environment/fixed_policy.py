"""Deterministic nearest-target pursuit in the V2 maneuver action space."""
from __future__ import annotations

import numpy as np

from ..math_utils import wrap_angle
from ..models import AircraftState


class NearestTargetPursuitPolicy:
    """A stable baseline with no planning, coordination or learned behavior."""

    def __init__(self, config: dict, action_config: dict) -> None:
        self.config = config
        self.command_config = action_config["command"]

    @staticmethod
    def nearest_target_index(
        own: AircraftState, targets: list[AircraftState]
    ) -> int | None:
        candidates = []
        for index, target in enumerate(targets):
            if target.alive:
                distance = (
                    (target.x - own.x) ** 2
                    + (target.y - own.y) ** 2
                    + (target.z - own.z) ** 2
                )
                candidates.append((distance, index))
        return min(candidates)[1] if candidates else None

    def action_toward(
        self,
        own: AircraftState,
        desired_heading: float,
        desired_elevation: float,
        desired_speed: float,
    ) -> np.ndarray:
        cfg = self.command_config
        speed_center = 0.5 * (
            float(cfg["speed_command_min"]) + float(cfg["speed_command_max"])
        )
        speed_half_range = 0.5 * (
            float(cfg["speed_command_max"]) - float(cfg["speed_command_min"])
        )
        return np.clip(np.array([
            wrap_angle(desired_heading - own.psi) / float(cfg["heading_delta_max"]),
            desired_elevation / float(cfg["pitch_command_max"]),
            (desired_speed - speed_center) / speed_half_range,
        ], dtype=np.float32), -1.0, 1.0)

    def action(self, own: AircraftState, targets: list[AircraftState]) -> np.ndarray:
        target_index = self.nearest_target_index(own, targets) if own.alive else None
        if target_index is None:
            return np.zeros(3, dtype=np.float32)
        target = targets[target_index]
        desired_heading = float(np.arctan2(target.y - own.y, target.x - own.x))
        horizontal_distance = float(np.hypot(target.x - own.x, target.y - own.y))
        desired_elevation = float(np.arctan2(own.z - target.z, horizontal_distance))
        return self.action_toward(
            own,
            desired_heading,
            desired_elevation,
            float(self.config["desired_speed"]),
        )

    def team_actions(
        self, team: list[AircraftState], targets: list[AircraftState]
    ) -> np.ndarray:
        return np.stack([self.action(state, targets) for state in team])


__all__ = ["NearestTargetPursuitPolicy"]
