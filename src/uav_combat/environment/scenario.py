"""Seeded 4v4 line-abreast initial-condition generator."""
from __future__ import annotations

import numpy as np

from ..math_utils import wrap_angle
from ..models import AircraftState


def random_line_abreast_states(
    rng: np.random.Generator,
    center_radius: float,
    formation_offsets: list[float],
    altitude: float,
    speed: float,
    heading_perturbation_max: float,
    speed_perturbation_max: float,
    altitude_perturbation_max: float,
    team_size: int = 4,
) -> tuple[list[AircraftState], list[AircraftState], float]:
    if team_size != 4 or len(formation_offsets) != team_size:
        raise ValueError("this benchmark requires four aircraft and four formation offsets")
    radial_angle = float(rng.uniform(-np.pi, np.pi))
    radial = np.array([np.cos(radial_angle), np.sin(radial_angle)])
    lateral = np.array([-radial[1], radial[0]])

    def team(side: float) -> list[AircraftState]:
        center = side * center_radius * radial
        nominal_heading = radial_angle if side < 0.0 else wrap_angle(radial_angle + np.pi)
        states = []
        for offset in formation_offsets:
            position = center + float(offset) * lateral
            states.append(AircraftState(
                x=float(position[0]),
                y=float(position[1]),
                z=-float(altitude + rng.uniform(-altitude_perturbation_max, altitude_perturbation_max)),
                v=float(speed + rng.uniform(-speed_perturbation_max, speed_perturbation_max)),
                theta=0.0,
                psi=float(wrap_angle(nominal_heading + rng.uniform(-heading_perturbation_max, heading_perturbation_max))),
            ))
        return states

    return team(-1.0), team(1.0), radial_angle


__all__ = ["random_line_abreast_states"]
