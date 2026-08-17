"""Segmented reward from Equation (25), split by simulation timing."""
from __future__ import annotations

import numpy as np
from .geometry import PaperAirCombatGeometry

DEG5, DEG15, DEG30 = np.deg2rad([5.0, 15.0, 30.0])
DISTANCE_THRESHOLD = 4000.0


def _tier(ata: float, ha: float, rewards: tuple[float, float, float]) -> float:
    a, h = abs(ata), abs(ha)
    if a <= DEG5 and h <= DEG5: return rewards[2]
    if a <= DEG15 and h <= DEG15: return rewards[1]
    if a <= DEG30 and h <= DEG30: return rewards[0]
    return 0.0


def equation25_geometric_reward(red_geometry: PaperAirCombatGeometry | None, blue_geometry: PaperAirCombatGeometry | None) -> float:
    """Compute R3+R4 from the immutable pre-attack snapshot."""
    reward = 0.0
    if red_geometry is not None:
        if abs(red_geometry.ata) <= DEG30 and abs(red_geometry.ha) <= DEG30 and red_geometry.distance >= DISTANCE_THRESHOLD:
            reward += 0.001
        if abs(red_geometry.aa) <= DEG30 and red_geometry.distance <= DISTANCE_THRESHOLD:
            reward += _tier(red_geometry.ata, red_geometry.ha, (0.01, 0.02, 0.10))
    if blue_geometry is not None and abs(blue_geometry.aa) <= DEG30 and blue_geometry.distance <= DISTANCE_THRESHOLD:
        reward += _tier(blue_geometry.ata, blue_geometry.ha, (-0.015, -0.025, -0.150))
    return float(reward)


def equation25_event_reward(destroyed_blue: int = 0, red_attack_death: bool = False, red_boundary_death: bool = False) -> float:
    """Compute post-event R1 plus boundary R2."""
    return float(10.0 * destroyed_blue - 10.0 * red_attack_death - 10.0 * red_boundary_death)


def equation25_reward(red_geometry=None, blue_geometry=None, destroyed_blue=0, red_destroyed=False, red_boundary_loss=False) -> float:
    """Compatibility composition equal to R1+R2+R3+R4."""
    return equation25_geometric_reward(red_geometry, blue_geometry) + equation25_event_reward(destroyed_blue, red_destroyed, red_boundary_loss)
