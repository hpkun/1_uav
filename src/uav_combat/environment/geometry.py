"""Public full-3D engagement geometry."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..models import AircraftState


@dataclass(frozen=True)
class EngagementGeometry:
    distance: float
    attack_angle: float
    escape_angle: float


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-12:
        return 0.0
    cosine = float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))
    return float(np.arccos(cosine))


def engagement_geometry(attacker: AircraftState, target: AircraftState) -> EngagementGeometry:
    displacement = np.array(
        [target.x - attacker.x, target.y - attacker.y, target.z - attacker.z], dtype=float
    )
    return EngagementGeometry(
        distance=float(np.linalg.norm(displacement)),
        attack_angle=_angle(attacker.velocity_vector(), displacement),
        escape_angle=_angle(target.velocity_vector(), displacement),
    )


def engagement_score(geometry: EngagementGeometry, battlefield_radius: float) -> float:
    range_score = float(np.clip(1.0 - geometry.distance / (2.0 * battlefield_radius), 0.0, 1.0))
    attack_score = (1.0 + np.cos(geometry.attack_angle)) / 2.0
    escape_score = (1.0 + np.cos(geometry.escape_angle)) / 2.0
    return float(range_score * attack_score * escape_score)


__all__ = ["EngagementGeometry", "engagement_geometry", "engagement_score"]
