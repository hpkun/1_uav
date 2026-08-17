"""Equation (24) natural 45-dimensional observation expansion."""
from __future__ import annotations

import numpy as np
from .geometry import compute_paper_geometry
from .sensor import ObservedState

OBSERVATION_DIM = 45


def earth_to_body_relative(observer: ObservedState, other: ObservedState) -> np.ndarray:
    """Transform an Earth-fixed NED displacement into observer body axes.

    The 3-2-1 direction-cosine matrix is the full ``F_g -> F_b`` transform
    implied by Fig. 1 and explicitly named for ``p_b`` after Eq. (17) of the
    2022 predecessor.  In particular this is not a yaw-only horizontal turn.
    """
    delta = np.array([other.x - observer.x, other.y - observer.y, other.z - observer.z], dtype=float)
    c_phi, s_phi = np.cos(observer.phi), np.sin(observer.phi)
    c_theta, s_theta = np.cos(observer.theta), np.sin(observer.theta)
    c_psi, s_psi = np.cos(observer.psi), np.sin(observer.psi)
    earth_to_body = np.array([
        [c_theta * c_psi, c_theta * s_psi, -s_theta],
        [s_phi * s_theta * c_psi - c_phi * s_psi,
         s_phi * s_theta * s_psi + c_phi * c_psi,
         s_phi * c_theta],
        [c_phi * s_theta * c_psi + s_phi * s_psi,
         c_phi * s_theta * s_psi - s_phi * c_psi,
         c_phi * c_theta],
    ])
    return earth_to_body @ delta


def build_observation(
    observer_index: int,
    red: list[ObservedState],
    blue: list[ObservedState],
    red_alive: list[bool],
    blue_alive: list[bool],
    position_scale: float = 5000.0,
    speed_scale: float = 300.0,
) -> np.ndarray:
    """Build fixed ID-ordered slots; dead aircraft slots are all zero."""
    if len(red) != 4 or len(blue) != 4:
        raise ValueError("Equation (24) expansion requires 4v4")
    own = red[observer_index]
    values: list[float] = []
    if red_alive[observer_index]:
        # Own absolute position is retained for boundary observability; scaling
        # is an explicitly documented reproduction assumption.
        values.extend([own.x / position_scale, own.y / position_scale, own.z / position_scale, own.v / speed_scale, own.phi, own.psi, own.theta])
    else:
        values.extend([0.0] * 7)
    for i, friendly in enumerate(red):
        if i == observer_index:
            continue
        if red_alive[i] and red_alive[observer_index]:
            rel = earth_to_body_relative(own, friendly) / position_scale
            values.extend([*rel, friendly.v / speed_scale, friendly.psi, friendly.theta])
        else:
            values.extend([0.0] * 6)
    own_state = own.as_aircraft_state(red_alive[observer_index])
    for j, enemy in enumerate(blue):
        if blue_alive[j] and red_alive[observer_index]:
            geometry = compute_paper_geometry(own_state, enemy.as_aircraft_state(True))
            values.extend([geometry.distance / position_scale, enemy.v / speed_scale, geometry.aa, geometry.ata, geometry.ha])
        else:
            values.extend([0.0] * 5)
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (OBSERVATION_DIM,) or not np.all(np.isfinite(result)):
        raise RuntimeError(f"invalid Equation (24) observation: {result.shape}")
    return result


def build_team_observations(
    own_team: list[ObservedState],
    opponent_team: list[ObservedState],
    own_alive: list[bool],
    opponent_alive: list[bool],
    position_scale: float = 5000.0,
    speed_scale: float = 300.0,
) -> np.ndarray:
    """Shared mirror-safe encoder for either red or blue perspective."""
    return np.stack([
        build_observation(i, own_team, opponent_team, own_alive, opponent_alive, position_scale, speed_scale)
        for i in range(4)
    ])
