"""Bounded, invariant 52-dimensional local state observations."""
from __future__ import annotations

import numpy as np

from ..models import AircraftState

OBSERVATION_DIM = 52


def flight_path_frame(state: AircraftState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = state.velocity_vector()
    forward = forward / max(float(np.linalg.norm(forward)), 1e-12)
    right = np.array([-np.sin(state.psi), np.cos(state.psi), 0.0], dtype=float)
    up = np.cross(right, forward)
    up = up / max(float(np.linalg.norm(up)), 1e-12)
    return forward, right, up


def _relative_slot(
    own: AircraftState,
    other: AircraftState,
    frame: tuple[np.ndarray, np.ndarray, np.ndarray],
    cfg: dict,
) -> np.ndarray:
    if not other.alive:
        return np.zeros(7, dtype=np.float32)
    forward, right, up = frame
    displacement = np.array([other.x - own.x, other.y - own.y, other.z - own.z], dtype=float)
    relative_velocity = other.velocity_vector() - own.velocity_vector()
    position_scale = float(cfg["relative_position_scale"])
    position = np.array([
        np.dot(displacement, forward) / position_scale,
        np.dot(displacement, right) / position_scale,
        np.dot(displacement, up) / position_scale,
    ])
    velocity = np.array([
        np.dot(relative_velocity, forward),
        np.dot(relative_velocity, right),
        np.dot(relative_velocity, up),
    ]) / cfg["relative_velocity_scale"]
    values = np.concatenate((position, velocity))
    return np.concatenate((np.clip(values, -1.0, 1.0), [1.0])).astype(np.float32)


def build_team_observations(
    team: list[AircraftState], opponents: list[AircraftState], cfg: dict,
    flight_envelope: dict,
) -> np.ndarray:
    observations = []
    for own_index, own in enumerate(team):
        if not own.alive:
            observations.append(np.zeros(OBSERVATION_DIM, dtype=np.float32))
            continue
        frame = flight_path_frame(own)
        altitude_min = float(flight_envelope["altitude_min"])
        altitude_max = float(flight_envelope["altitude_max"])
        self_features = np.clip(np.array([
            (own.v - cfg["speed_center"]) / cfg["speed_scale"],
            own.theta / (np.pi / 3.0),
            2.0 * (own.altitude - altitude_min) / (
                altitude_max - altitude_min
            ) - 1.0,
        ], dtype=np.float32), -1.0, 1.0)
        allies = [state for index, state in enumerate(team) if index != own_index]
        slots = [
            _relative_slot(own, state, frame, cfg)
            for state in allies + list(opponents)
        ]
        observations.append(np.concatenate([self_features, *slots]).astype(np.float32))
    result = np.stack(observations)
    if result.shape != (4, OBSERVATION_DIM) or not np.all(np.isfinite(result)):
        raise FloatingPointError("invalid combat observation")
    return result


__all__ = ["OBSERVATION_DIM", "build_team_observations", "flight_path_frame"]
