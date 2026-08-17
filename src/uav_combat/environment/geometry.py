"""Air-combat geometry from Figure 2 and Equation (6)."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..math_utils import wrap_angle
from ..models import AircraftState


@dataclass(frozen=True)
class PaperAirCombatGeometry:
    distance: float
    ata: float
    aa: float
    ha: float
    hca: float


def compute_paper_geometry(own: AircraftState, target: AircraftState, eps: float = 1e-8) -> PaperAirCombatGeometry:
    """Return signed ATA/AA/HA/HCA, with ATA and AA in the xy projection.

    ATA is own-heading to horizontal LOS. AA is target-heading to the LOS from
    target back to own. HA is the elevation LOS in NED (positive target above).
    HCA is target heading minus own heading. Absolute values are used by the
    paper's launch and reward inequalities.
    """
    rel = target.as_array()[:3] - own.as_array()[:3]
    distance = float(np.linalg.norm(rel))
    horizontal = float(np.hypot(rel[0], rel[1]))
    if distance < eps:
        return PaperAirCombatGeometry(0.0, 0.0, 0.0, 0.0, wrap_angle(target.psi - own.psi))
    los = float(np.arctan2(rel[1], rel[0])) if horizontal >= eps else own.psi
    return PaperAirCombatGeometry(
        distance=distance,
        ata=wrap_angle(los - own.psi),
        aa=wrap_angle((los + np.pi) - target.psi),
        ha=float(np.arctan2(-rel[2], max(horizontal, eps))),
        hca=wrap_angle(target.psi - own.psi),
    )
