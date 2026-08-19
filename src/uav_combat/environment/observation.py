"""Rotation-invariant 54-dimensional local observations."""
from __future__ import annotations

import numpy as np

from ..models import AircraftState

OBSERVATION_DIM = 54


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
    battlefield: dict,
) -> np.ndarray:
    if not other.alive:
        return np.zeros(7, dtype=np.float32)
    forward, right, up = frame
    displacement = np.array([other.x - own.x, other.y - own.y, other.z - own.z], dtype=float)
    relative_velocity = other.velocity_vector() - own.velocity_vector()
    horizontal_scale = 2.0 * float(battlefield["horizontal_radius"])
    vertical_scale = float(battlefield["altitude_max"] - battlefield["altitude_min"])
    position = np.array([
        np.dot(displacement, forward) / horizontal_scale,
        np.dot(displacement, right) / horizontal_scale,
        np.dot(displacement, up) / vertical_scale,
    ])
    velocity = np.array([
        np.dot(relative_velocity, forward),
        np.dot(relative_velocity, right),
        np.dot(relative_velocity, up),
    ]) / cfg["relative_velocity_scale"]
    return np.concatenate((position, velocity, [1.0])).astype(np.float32)


def build_team_observations(
    team: list[AircraftState], opponents: list[AircraftState], cfg: dict,
    battlefield: dict,
) -> np.ndarray:
    observations = []
    for own_index, own in enumerate(team):
        if not own.alive:
            observations.append(np.zeros(OBSERVATION_DIM, dtype=np.float32))
            continue
        frame = flight_path_frame(own)
        horizontal_forward = np.array([np.cos(own.psi), np.sin(own.psi), 0.0])
        center_vector = np.array([-own.x, -own.y, 0.0])
        altitude_min = float(battlefield["altitude_min"])
        altitude_max = float(battlefield["altitude_max"])
        center_scale = float(battlefield["horizontal_radius"])
        self_features = np.array([
            (own.v - cfg["speed_center"]) / cfg["speed_scale"],
            own.theta / (np.pi / 3.0),
            2.0 * (own.altitude - altitude_min) / (
                altitude_max - altitude_min
            ) - 1.0,
            np.dot(center_vector, horizontal_forward) / center_scale,
            np.dot(center_vector, frame[1]) / center_scale,
        ], dtype=np.float32)
        allies = [state for index, state in enumerate(team) if index != own_index]
        slots = [
            _relative_slot(own, state, frame, cfg, battlefield)
            for state in allies + list(opponents)
        ]
        observations.append(np.concatenate([self_features, *slots]).astype(np.float32))
    result = np.stack(observations)
    if result.shape != (4, OBSERVATION_DIM) or not np.all(np.isfinite(result)):
        raise FloatingPointError("invalid combat observation")
    return result


__all__ = ["OBSERVATION_DIM", "build_team_observations", "flight_path_frame"]
