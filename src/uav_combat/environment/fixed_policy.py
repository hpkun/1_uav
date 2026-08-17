"""Section 2.5 nearest-alive-target pure pursuit."""
from __future__ import annotations

import numpy as np
from ..math_utils import angle_difference
from ..models import AircraftState


class NearestTargetPursuitPolicy:
    def __init__(self, delta_psi_max: float = np.pi, delta_theta_max: float = np.pi / 3, delta_v_max: float = 50.0, desired_speed: float = 225.0) -> None:
        self.scales = np.array([delta_psi_max, delta_theta_max, delta_v_max], dtype=float)
        self.desired_speed = float(desired_speed)

    @staticmethod
    def nearest_target_index(own: AircraftState, targets: list[AircraftState]) -> int | None:
        alive = [(i, t) for i, t in enumerate(targets) if t.alive]
        if not alive:
            return None
        p = own.as_array()[:3]
        return min(alive, key=lambda it: (float(np.linalg.norm(it[1].as_array()[:3] - p)), it[0]))[0]

    def action(self, own: AircraftState, targets: list[AircraftState]) -> tuple[np.ndarray, int | None]:
        index = self.nearest_target_index(own, targets)
        if index is None or not own.alive:
            return np.zeros(3, dtype=np.float32), index
        rel = targets[index].as_array()[:3] - own.as_array()[:3]
        horizontal = max(float(np.hypot(rel[0], rel[1])), 1e-8)
        desired_psi = float(np.arctan2(rel[1], rel[0]))
        desired_theta = float(np.arctan2(-rel[2], horizontal))
        physical = np.array([angle_difference(desired_psi, own.psi), desired_theta - own.theta, self.desired_speed - own.v])
        return np.clip(physical / self.scales, -1.0, 1.0).astype(np.float32), index
