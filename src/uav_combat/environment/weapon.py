"""Deterministic, dwell-based firing-window model."""
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
    range_min: float
    range_max: float
    attack_angle_max: float
    target_aspect_max: float
    lock_steps_required: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.range_min < self.range_max:
            raise ValueError("weapon range must satisfy 0 <= range_min < range_max")
        if min(self.attack_angle_max, self.target_aspect_max) <= 0.0:
            raise ValueError("weapon angle limits must be positive")
        if self.lock_steps_required <= 0:
            raise ValueError("lock_steps_required must be positive")

    def in_fire_window(self, geometry: EngagementGeometry) -> bool:
        return bool(
            self.range_min <= geometry.distance <= self.range_max
            and geometry.attack_angle <= self.attack_angle_max
            and geometry.target_aspect <= self.target_aspect_max
        )


__all__ = ["LockState", "WeaponEnvelope"]
