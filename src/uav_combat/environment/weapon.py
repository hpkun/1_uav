"""Paper weapon launch Equation (7) and probabilistic hit Equation (8)."""
from __future__ import annotations

import numpy as np
from .geometry import PaperAirCombatGeometry


class WeaponModel:
    def __init__(self, distance_min: float, distance_max: float, ata_max: float, ha_max: float, d_hit: float, c4: float, c5: float) -> None:
        if distance_min < 0 or distance_max < distance_min or d_hit <= 0 or min(c4, c5) < 0:
            raise ValueError("invalid weapon assumptions")
        self.distance_min, self.distance_max = float(distance_min), float(distance_max)
        self.ata_max, self.ha_max = float(ata_max), float(ha_max)
        self.d_hit, self.c4, self.c5 = float(d_hit), float(c4), float(c5)

    def can_fire(self, geometry: PaperAirCombatGeometry) -> bool:
        return bool(abs(geometry.ata) <= self.ata_max and abs(geometry.ha) <= self.ha_max and self.distance_min <= geometry.distance <= self.distance_max)

    def sample_hit(self, geometry: PaperAirCombatGeometry, rng: np.random.Generator) -> bool:
        """Sample the shared epsilon_fire printed in Equation (8)."""
        epsilon_fire = float(rng.normal())
        threshold = float(np.pi * np.exp(-geometry.distance / self.d_hit))
        return bool(abs(geometry.ata + self.c4 * epsilon_fire) <= threshold and abs(geometry.ha + self.c5 * epsilon_fire) <= threshold)
