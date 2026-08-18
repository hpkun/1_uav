"""Deterministic attack envelope and continuous lock state."""
from __future__ import annotations

from dataclasses import dataclass

from .geometry import EngagementGeometry


@dataclass
class LockState:
    current_lock_target: int = -1
    lock_steps: int = 0

    def reset(self) -> None:
        self.current_lock_target = -1
        self.lock_steps = 0


@dataclass(frozen=True)
class WeaponEnvelope:
    attack_distance_max: float
    attack_angle_max: float
    escape_angle_max: float
    lock_steps_required: int

    def attackable(self, geometry: EngagementGeometry) -> bool:
        return bool(
            geometry.distance <= self.attack_distance_max
            and geometry.attack_angle <= self.attack_angle_max
            and geometry.escape_angle <= self.escape_angle_max
        )


__all__ = ["LockState", "WeaponEnvelope"]
