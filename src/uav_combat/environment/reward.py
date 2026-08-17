"""Segmented reward from Equation (25)."""
from __future__ import annotations

import numpy as np
from .geometry import PaperAirCombatGeometry

DEG5, DEG15, DEG30 = np.deg2rad([5.0, 15.0, 30.0])
DISTANCE_THRESHOLD = 4000.0


def _tier(ata: float, ha: float, rewards: tuple[float, float, float]) -> float:
    a, h = abs(ata), abs(ha)
    if a <= DEG5 and h <= DEG5:
        return rewards[2]
    if a <= DEG15 and h <= DEG15:
        return rewards[1]
    if a <= DEG30 and h <= DEG30:
        return rewards[0]
    return 0.0


def equation25_reward(
    red_geometry: PaperAirCombatGeometry | None,
    blue_geometry: PaperAirCombatGeometry | None,
    destroyed_blue: int = 0,
    red_destroyed: bool = False,
    red_boundary_loss: bool = False,
) -> float:
    r1 = 10.0 * destroyed_blue - (10.0 if red_destroyed else 0.0)
    r2 = -10.0 if red_boundary_loss else 0.0
    r3 = 0.0
    r4 = 0.0
    if red_geometry is not None:
        if abs(red_geometry.ata) <= DEG30 and abs(red_geometry.ha) <= DEG30 and red_geometry.distance >= DISTANCE_THRESHOLD:
            r3 = 0.001
        if abs(red_geometry.aa) <= DEG30 and red_geometry.distance <= DISTANCE_THRESHOLD:
            r4 += _tier(red_geometry.ata, red_geometry.ha, (0.01, 0.02, 0.10))
    if blue_geometry is not None and abs(blue_geometry.aa) <= DEG30 and blue_geometry.distance <= DISTANCE_THRESHOLD:
        r4 += _tier(blue_geometry.ata, blue_geometry.ha, (-0.015, -0.025, -0.150))
    return float(r1 + r2 + r3 + r4)
