"""Paper Eq. (7)-(8) entry-triggered probabilistic attack model."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .geometry import EngagementGeometry


@dataclass
class FireState:
    armed: bool = True


@dataclass(frozen=True)
class WeaponEnvelope:
    range_min: float
    range_max: float
    attack_angle_max: float
    height_angle_max: float
    effective_hit_distance: float
    attack_noise_scale: float
    height_noise_scale: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.range_min < self.range_max:
            raise ValueError("weapon range must satisfy 0 <= range_min < range_max")
        if min(
            self.attack_angle_max, self.height_angle_max,
            self.effective_hit_distance,
        ) <= 0.0:
            raise ValueError("weapon angle and hit-distance parameters must be positive")
        if min(self.attack_noise_scale, self.height_noise_scale) < 0.0:
            raise ValueError("weapon noise scales must be non-negative")

    def in_fire_window(self, geometry: EngagementGeometry) -> bool:
        return bool(
            self.range_min <= geometry.distance <= self.range_max
            and abs(geometry.ata) <= self.attack_angle_max
            and abs(geometry.ha) <= self.height_angle_max
        )

    def hit_threshold(self, distance: float) -> float:
        return float(np.pi * np.exp(-distance / self.effective_hit_distance))

    def attempt_hit(
        self, geometry: EngagementGeometry, rng: np.random.Generator
    ) -> bool:
        threshold = self.hit_threshold(geometry.distance)
        attack_noise = float(rng.normal())
        height_noise = float(rng.normal())
        return bool(
            abs(geometry.ata + self.attack_noise_scale * attack_noise) <= threshold
            and abs(geometry.ha + self.height_noise_scale * height_noise) <= threshold
        )


__all__ = ["FireState", "WeaponEnvelope"]
