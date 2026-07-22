"""Paper-oriented 1v1 actor observation and critic state."""

from __future__ import annotations

from math import atan2, pi

import numpy as np
from numpy.typing import NDArray

from uav_env.core.geometry import normalize_angle
from uav_env.core.state import UAVState
from uav_env.observations.normalization import NormalizationConfig, normalize_features


def _relative_angles(own_state: UAVState, enemy_state: UAVState) -> tuple[float, float]:
    displacement = enemy_state.position_vector() - own_state.position_vector()
    horizontal = float(np.hypot(displacement[0], displacement[1]))
    pitch = atan2(float(displacement[2]), horizontal)
    yaw = atan2(float(displacement[1]), float(displacement[0]))
    return pitch, yaw


def _safe_vector_angle(first: NDArray[np.float64], second: NDArray[np.float64]) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1.0e-12:
        return 0.0
    cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
    return float(np.arccos(cosine))


def actor_observation_raw_1v1(own_state: UAVState, enemy_state: UAVState) -> NDArray[np.float64]:
    """Return the fixed 11-dimensional unnormalized actor feature order."""

    own_position = own_state.position_vector()
    enemy_position = enemy_state.position_vector()
    displacement_to_enemy = enemy_position - own_position
    distance = float(np.linalg.norm(displacement_to_enemy))
    relative_pitch, relative_yaw = _relative_angles(own_state, enemy_state)
    velocity_difference = own_state.velocity_vector() - enemy_state.velocity_vector()
    enemy_escape_angle = _safe_vector_angle(enemy_state.velocity_vector(), displacement_to_enemy)
    self_attack_angle = _safe_vector_angle(own_state.velocity_vector(), displacement_to_enemy)
    return np.asarray(
        [
            own_state.x - enemy_state.x,
            own_state.y - enemy_state.y,
            own_state.z,
            distance,
            relative_pitch,
            relative_yaw,
            velocity_difference[0],
            velocity_difference[1],
            velocity_difference[2],
            enemy_escape_angle,
            self_attack_angle,
        ],
        dtype=np.float64,
    )


def build_actor_observation_1v1(
    own_state: UAVState,
    enemy_state: UAVState,
    normalization_config: NormalizationConfig,
) -> NDArray[np.float64]:
    """Build and normalize the 2023 11-dimensional actor observation."""

    raw = actor_observation_raw_1v1(own_state, enemy_state)
    references = np.asarray(
        [
            normalization_config.horizontal_reference,
            normalization_config.horizontal_reference,
            normalization_config.altitude_reference,
            normalization_config.horizontal_reference,
            normalization_config.angle_reference,
            normalization_config.angle_reference,
            normalization_config.speed_difference_reference,
            normalization_config.speed_difference_reference,
            normalization_config.speed_difference_reference,
            normalization_config.angle_reference,
            normalization_config.angle_reference,
        ]
    )
    return normalize_features(raw, references, normalization_config)[0]


def critic_state_raw_1v1(own_state: UAVState, enemy_state: UAVState) -> NDArray[np.float64]:
    """Return the fixed 10-dimensional unnormalized critic state order."""

    displacement = enemy_state.position_vector() - own_state.position_vector()
    distance = float(np.linalg.norm(displacement))
    _, relative_yaw = _relative_angles(own_state, enemy_state)
    self_attack_angle = _safe_vector_angle(own_state.velocity_vector(), displacement)
    enemy_escape_angle = _safe_vector_angle(enemy_state.velocity_vector(), displacement)
    return np.asarray(
        [
            distance,
            self_attack_angle,
            enemy_escape_angle,
            enemy_state.z - own_state.z,
            relative_yaw,
            normalize_angle(enemy_state.heading_angle - own_state.heading_angle),
            normalize_angle(enemy_state.flight_path_angle - own_state.flight_path_angle),
            own_state.speed - enemy_state.speed,
            own_state.health,
            own_state.health - enemy_state.health,
        ],
        dtype=np.float64,
    )


def build_critic_state_1v1(
    own_state: UAVState,
    enemy_state: UAVState,
    normalization_config: NormalizationConfig,
) -> NDArray[np.float64]:
    """Build and normalize the 10-dimensional centralized critic state."""

    raw = critic_state_raw_1v1(own_state, enemy_state)
    references = np.asarray(
        [
            normalization_config.horizontal_reference,
            pi,
            pi,
            normalization_config.altitude_reference,
            pi,
            pi,
            pi,
            normalization_config.speed_difference_reference,
            normalization_config.health_reference,
            normalization_config.health_reference,
        ]
    )
    return normalize_features(raw, references, normalization_config)[0]


def build_single_observation(
    own_state: UAVState,
    enemy_state: UAVState,
    normalization_config: NormalizationConfig,
) -> NDArray[np.float64]:
    """Alias for the actor observation used by earlier callers."""

    return build_actor_observation_1v1(own_state, enemy_state, normalization_config)
