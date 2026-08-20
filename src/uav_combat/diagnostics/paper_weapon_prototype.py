"""Diagnostic-only Li et al. (2023) Eq. (7)-(8) attack prototype."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models import AircraftState


FIRE_DISTANCE_MIN = 0.0
FIRE_DISTANCE_MAX = 4000.0
FIRE_ANGLE_MAX = np.pi / 6.0
D_HIT = FIRE_DISTANCE_MAX / np.log(6.0)
C4 = C5 = 1.0


@dataclass(frozen=True)
class PaperWeaponGeometry:
    distance: float
    ata: float
    ha: float


def wrap_angle(value: float) -> float:
    return float((value + np.pi) % (2.0 * np.pi) - np.pi)


def paper_weapon_geometry(own: AircraftState, target: AircraftState) -> PaperWeaponGeometry:
    dx, dy, dz = target.x - own.x, target.y - own.y, target.z - own.z
    horizontal = float(np.hypot(dx, dy))
    los = float(np.arctan2(dy, dx)) if horizontal > 1e-12 else own.psi
    return PaperWeaponGeometry(
        distance=float(np.sqrt(dx * dx + dy * dy + dz * dz)),
        ata=wrap_angle(los - own.psi),
        ha=float(np.arctan2(-dz, horizontal)),
    )


def fire_gate(geometry: PaperWeaponGeometry) -> bool:
    """Paper Eq. (7), with reconstructed zero minimum range and no AA/lock gate."""
    return bool(
        FIRE_DISTANCE_MIN <= geometry.distance <= FIRE_DISTANCE_MAX
        and abs(geometry.ata) <= FIRE_ANGLE_MAX
        and abs(geometry.ha) <= FIRE_ANGLE_MAX
    )


def hit_threshold(distance: float, d_hit: float = D_HIT) -> float:
    if distance < 0.0 or d_hit <= 0.0:
        raise ValueError("distance must be non-negative and D_hit positive")
    return float(np.pi * np.exp(-distance / d_hit))


def hit_samples(
    geometry: PaperWeaponGeometry, rng: np.random.Generator, samples: int,
    noise_semantics: str, c4: float = C4, c5: float = C5, d_hit: float = D_HIT,
) -> np.ndarray:
    if samples <= 0:
        raise ValueError("samples must be positive")
    threshold = hit_threshold(geometry.distance, d_hit)
    if noise_semantics == "shared":
        epsilon = rng.standard_normal(samples)
        ata_noise, ha_noise = epsilon, epsilon
    elif noise_semantics == "independent":
        ata_noise = rng.standard_normal(samples)
        ha_noise = rng.standard_normal(samples)
    else:
        raise ValueError("noise_semantics must be shared or independent")
    return (
        np.abs(geometry.ata + c4 * ata_noise) <= threshold
    ) & (
        np.abs(geometry.ha + c5 * ha_noise) <= threshold
    )


class EntryTriggeredAttempt:
    """One attempt on each rising edge of a continuous Eq. (7) firing window."""

    def __init__(self) -> None:
        self._inside = False

    def update(self, inside_fire_gate: bool) -> bool:
        attempt = bool(inside_fire_gate and not self._inside)
        self._inside = bool(inside_fire_gate)
        return attempt

    def reset(self) -> None:
        self._inside = False


EVIDENCE = {
    "eq7_structure": "PAPER",
    "D_firemax_4000m": "PAPER Table 1",
    "ATA_HA_max_30deg": "PAPER Table 1",
    "D_firemin_zero": "PREDECESSOR-supported RECONSTRUCTION requested for prototype",
    "eq8_structure": "PAPER",
    "D_hit_4000_over_ln6": "DERIVED by setting 4-km nominal threshold to 30 deg",
    "c4_c5_one": "PREDECESSOR-supported RECONSTRUCTION",
    "shared_epsilon": "PAPER-literal and PREDECESSOR-supported",
    "independent_epsilon": "RECONSTRUCTION tested for sign symmetry",
    "per_step_resample": "RECONSTRUCTION cadence candidate",
    "entry_triggered": "RECONSTRUCTION cadence candidate",
}


__all__ = [
    "C4", "C5", "D_HIT", "EVIDENCE", "EntryTriggeredAttempt",
    "FIRE_ANGLE_MAX", "FIRE_DISTANCE_MAX", "PaperWeaponGeometry", "fire_gate",
    "hit_samples", "hit_threshold", "paper_weapon_geometry",
]
