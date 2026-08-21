"""Stationary mixture of representative pre-merge combat geometries."""
from __future__ import annotations

import numpy as np

from ..math_utils import wrap_angle
from ..models import AircraftState


def random_combat_states(
    rng: np.random.Generator,
    modes: list[str],
    mode_probabilities: list[float],
    center_separation_min: float,
    center_separation_max: float,
    formation_offsets: list[float],
    altitude_min: float,
    altitude_max: float,
    altitude_perturbation_max: float,
    speed_min: float,
    speed_max: float,
    heading_perturbation_max: float,
    offset_angle_min: float,
    offset_angle_max: float,
    team_size: int = 4,
) -> tuple[list[AircraftState], list[AircraftState], float, str]:
    """Sample head-on, offset or flank initial states without curriculum."""
    if team_size != 4 or len(formation_offsets) != team_size:
        raise ValueError("this benchmark requires four aircraft and four formation offsets")
    if modes != ["head_on", "offset", "flank"]:
        raise ValueError("scenario modes must be head_on, offset and flank")
    probabilities = np.asarray(mode_probabilities, dtype=float)
    if probabilities.shape != (3,) or np.any(probabilities < 0.0):
        raise ValueError("mode_probabilities must contain three non-negative values")
    probabilities /= probabilities.sum()

    mode = str(rng.choice(modes, p=probabilities))
    radial_angle = float(rng.uniform(-np.pi, np.pi))
    separation = float(rng.uniform(center_separation_min, center_separation_max))
    radial = np.array([np.cos(radial_angle), np.sin(radial_angle)])
    lateral = np.array([-radial[1], radial[0]])
    red_heading = radial_angle
    blue_heading = wrap_angle(radial_angle + np.pi)

    if mode == "offset":
        offset = float(rng.uniform(offset_angle_min, offset_angle_max))
        sign = float(rng.choice([-1.0, 1.0]))
        red_heading = wrap_angle(red_heading + sign * offset)
        blue_heading = wrap_angle(blue_heading - sign * offset)
    elif mode == "flank":
        crossing = float(rng.choice([-1.0, 1.0])) * np.pi / 2.0
        if rng.random() < 0.5:
            blue_heading = wrap_angle(radial_angle + crossing)
        else:
            red_heading = wrap_angle(radial_angle + np.pi + crossing)

    base_altitude = float(rng.uniform(altitude_min, altitude_max))

    def team(side: float, nominal_heading: float) -> list[AircraftState]:
        center = side * 0.5 * separation * radial
        states = []
        for offset in formation_offsets:
            position = center + float(offset) * lateral
            states.append(AircraftState(
                x=float(position[0]),
                y=float(position[1]),
                z=-float(base_altitude + rng.uniform(
                    -altitude_perturbation_max, altitude_perturbation_max
                )),
                v=float(rng.uniform(speed_min, speed_max)),
                theta=0.0,
                psi=float(wrap_angle(nominal_heading + rng.uniform(
                    -heading_perturbation_max, heading_perturbation_max
                ))),
            ))
        return states

    return (
        team(-1.0, red_heading),
        team(1.0, blue_heading),
        radial_angle,
        mode,
    )


__all__ = ["random_combat_states"]
