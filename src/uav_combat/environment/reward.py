"""Potential-based dense reward helpers."""
from __future__ import annotations

import numpy as np

from ..models import AircraftState
from .geometry import engagement_geometry, engagement_score


def team_potentials(
    team: list[AircraftState], opponents: list[AircraftState], battlefield_radius: float
) -> np.ndarray:
    values = np.zeros(len(team), dtype=np.float32)
    for index, own in enumerate(team):
        if not own.alive:
            continue
        attack = [
            engagement_score(engagement_geometry(own, target), battlefield_radius)
            for target in opponents if target.alive
        ]
        threat = [
            engagement_score(engagement_geometry(target, own), battlefield_radius)
            for target in opponents if target.alive
        ]
        values[index] = (max(attack, default=0.0) - max(threat, default=0.0))
    return values


__all__ = ["team_potentials"]
