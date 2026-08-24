"""Signed air-combat geometry, including true 3-D weapon boresight error."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..math_utils import wrap_angle
from ..models import AircraftState


@dataclass(frozen=True)
class EngagementGeometry:
    distance: float
    horizontal_distance: float
    line_of_sight: float
    ata: float
    aa: float
    ha: float
    hca: float
    off_boresight: float

    @property
    def attack_angle(self) -> float:
        return self.off_boresight

    @property
    def target_aspect(self) -> float:
        return abs(self.aa)


def engagement_geometry(
    attacker: AircraftState, target: AircraftState
) -> EngagementGeometry:
    dx = float(target.x - attacker.x)
    dy = float(target.y - attacker.y)
    dz = float(target.z - attacker.z)
    horizontal = float(np.hypot(dx, dy))
    distance = float(np.sqrt(dx * dx + dy * dy + dz * dz))
    line_of_sight = float(np.arctan2(dy, dx)) if horizontal > 0.0 else attacker.psi
    if distance > 0.0:
        los_unit = np.asarray([dx, dy, dz], dtype=float) / distance
        ct = float(np.cos(attacker.theta))
        forward_unit = np.asarray([
            ct * np.cos(attacker.psi),
            ct * np.sin(attacker.psi),
            -np.sin(attacker.theta),
        ], dtype=float)
        cosine = float(np.clip(np.dot(forward_unit, los_unit), -1.0, 1.0))
        off_boresight = float(np.arccos(cosine))
    else:
        # Preserve the existing zero-range boundary semantics: a coincident
        # target has no defined LOS direction and is treated as aligned.
        off_boresight = 0.0
    return EngagementGeometry(
        distance=distance,
        horizontal_distance=horizontal,
        line_of_sight=line_of_sight,
        ata=float(wrap_angle(line_of_sight - attacker.psi)),
        aa=float(wrap_angle(target.psi - line_of_sight)),
        ha=float(np.arctan2(-dz, horizontal)),
        hca=float(wrap_angle(target.psi - attacker.psi)),
        off_boresight=off_boresight,
    )


__all__ = ["EngagementGeometry", "engagement_geometry"]
