"""Full-3D directed engagement geometry."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..models import AircraftState


@dataclass(frozen=True)
class EngagementGeometry:
    distance: float
    attack_angle: float
    target_aspect: float


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-12:
        return 0.0
    cosine = float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))
    return float(np.arccos(cosine))


def engagement_geometry(attacker: AircraftState, target: AircraftState) -> EngagementGeometry:
    displacement = np.array(
        [target.x - attacker.x, target.y - attacker.y, target.z - attacker.z],
        dtype=float,
    )
    return EngagementGeometry(
        distance=float(np.linalg.norm(displacement)),
        attack_angle=_angle(attacker.velocity_vector(), displacement),
        target_aspect=_angle(target.velocity_vector(), displacement),
    )


__all__ = ["EngagementGeometry", "engagement_geometry"]
