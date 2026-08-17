"""Paper 4v4 random-diameter initialization."""
from __future__ import annotations

import numpy as np
from ..math_utils import wrap_angle
from ..models import AircraftState


def random_diameter_states(
    rng: np.random.Generator,
    team_size: int = 4,
    center_distance: float = 4000.0,
    formation_spacing: float = 150.0,
    altitude: float = 3000.0,
    speed: float = 225.0,
) -> tuple[list[AircraftState], list[AircraftState], float]:
    """Place symmetric formations at opposite points of a random diameter.

    Center distance and formation spacing are reproduction assumptions because
    the paper does not publish within-formation geometry.
    """
    if team_size != 4:
        raise ValueError("the paper scenario has exactly four UAVs per side")
    angle = float(rng.uniform(-np.pi, np.pi))
    radial = np.array([np.cos(angle), np.sin(angle)])
    tangent = np.array([-radial[1], radial[0]])
    offsets = (np.arange(team_size, dtype=float) - 1.5) * formation_spacing
    red_center, blue_center = -center_distance * radial, center_distance * radial
    red, blue = [], []
    for offset in offsets:
        rp, bp = red_center + offset * tangent, blue_center + offset * tangent
        red.append(AircraftState(rp[0], rp[1], -altitude, speed, 0.0, wrap_angle(angle), True))
        blue.append(AircraftState(bp[0], bp[1], -altitude, speed, 0.0, wrap_angle(angle + np.pi), True))
    return red, blue, angle
