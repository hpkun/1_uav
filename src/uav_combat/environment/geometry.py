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
    """Evaluate Figure 2 / Equation (6) from ``own`` (o1) to ``target`` (o2).

    ``ATA`` is the signed own-heading-to-LOS angle. ``AA`` is the signed angle
    from that same o1->o3 horizontal LOS to the target velocity.  In
    particular, AA does *not* use the reverse LOS: an attacker directly behind
    a same-heading target has ATA=AA=0, as required by Figure 2's pursuit
    geometry.  ``HA`` is positive when the target is above in NED coordinates.
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
        aa=wrap_angle(target.psi - los),
        ha=float(np.arctan2(-rel[2], max(horizontal, eps))),
        hca=wrap_angle(target.psi - own.psi),
    )
