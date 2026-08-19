"""Potential-based dense reward helpers."""
from __future__ import annotations

import numpy as np

from ..models import AircraftState
from .arena import boundary_cost as state_boundary_cost
from .geometry import engagement_geometry, engagement_score


def tactical_potentials(
    team: list[AircraftState], opponents: list[AircraftState], engagement_distance_scale: float
) -> np.ndarray:
    values = np.zeros(len(team), dtype=np.float32)
    for index, own in enumerate(team):
        if not own.alive:
            continue
        attack = [
            engagement_score(engagement_geometry(own, target), engagement_distance_scale)
            for target in opponents if target.alive
        ]
        threat = [
            engagement_score(engagement_geometry(target, own), engagement_distance_scale)
            for target in opponents if target.alive
        ]
        values[index] = (max(attack, default=0.0) - max(threat, default=0.0))
    return values


def boundary_costs(team: list[AircraftState], battlefield: dict) -> np.ndarray:
    """Return arena costs, including a newly boundary-dead state's terminal position."""
    return np.asarray(
        [state_boundary_cost(state, battlefield) for state in team], dtype=np.float32
    )


def combined_potentials(
    team: list[AircraftState], opponents: list[AircraftState],
    engagement_distance_scale: float, battlefield: dict, boundary_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tactical = tactical_potentials(team, opponents, engagement_distance_scale)
    boundary = boundary_costs(team, battlefield)
    combined = tactical - float(boundary_weight) * boundary
    return tactical, boundary, combined.astype(np.float32)


# Backward-compatible public name for the unchanged tactical definition.
team_potentials = tactical_potentials


__all__ = [
    "boundary_costs", "combined_potentials", "tactical_potentials", "team_potentials",
]
