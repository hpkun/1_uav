"""Canonical signed horizontal air-combat geometry for Eq. (6)."""
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

    @property
    def attack_angle(self) -> float:
        return abs(self.ata)

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
    line_of_sight = float(np.arctan2(dy, dx)) if horizontal > 0.0 else attacker.psi
    return EngagementGeometry(
        distance=float(np.sqrt(dx * dx + dy * dy + dz * dz)),
        horizontal_distance=horizontal,
        line_of_sight=line_of_sight,
        ata=float(wrap_angle(line_of_sight - attacker.psi)),
        aa=float(wrap_angle(target.psi - line_of_sight)),
        ha=float(np.arctan2(-dz, horizontal)),
        hca=float(wrap_angle(target.psi - attacker.psi)),
    )


__all__ = ["EngagementGeometry", "engagement_geometry"]
