"""Named 1v1 actor observation and centralized critic state."""

from __future__ import annotations

from math import atan2, pi

import numpy as np
from numpy.typing import NDArray

from uav_env.core.geometry import normalize_angle
from uav_env.core.state import UAVState
from uav_env.observations.normalization import FeatureSpec, NormalizationConfig, NormalizationResult, normalize_by_specs


ACTOR_OBSERVATION_FEATURE_NAMES = [
    "dx_self_minus_enemy", "dy_self_minus_enemy", "altitude_self", "distance",
    "relative_pitch", "relative_yaw", "dvx", "dvy", "dvz", "enemy_escape_angle", "self_attack_angle",
]
CRITIC_STATE_FEATURE_NAMES = [
    "distance", "self_attack_angle", "enemy_escape_angle", "dz_enemy_minus_self", "relative_yaw",
    "heading_difference", "flight_path_angle_difference", "speed_difference", "health_self", "health_difference",
]


def relative_angles(source: UAVState, target: UAVState) -> tuple[float, float]:
    """Return signed pitch and yaw of the source-to-target displacement."""

    displacement = target.position_vector() - source.position_vector()
    return atan2(float(displacement[2]), float(np.hypot(displacement[0], displacement[1]))), atan2(float(displacement[1]), float(displacement[0]))


def safe_vector_angle(first: NDArray[np.float64], second: NDArray[np.float64]) -> float:
    """Return a finite zero-protected unsigned vector angle."""

    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1.0e-12:
        return 0.0
    return float(np.arccos(float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))))


def actor_feature_specs(config: NormalizationConfig) -> list[FeatureSpec]:
    """Return the explicit Actor feature type table."""

    h, z, a, v = config.horizontal_reference, config.altitude_reference, config.angle_reference, config.speed_difference_reference
    kinds = ["signed", "signed", "nonnegative", "nonnegative", "signed", "yaw", "signed", "signed", "signed", "nonnegative", "nonnegative"]
    refs = [h, h, z, h, a, config.heading_reference, v, v, v, a, a]
    return [FeatureSpec(name, ref, kind) for name, ref, kind in zip(ACTOR_OBSERVATION_FEATURE_NAMES, refs, kinds)]  # type: ignore[arg-type]


def critic_feature_specs(config: NormalizationConfig) -> list[FeatureSpec]:
    """Return the explicit Critic feature type table."""

    refs = [config.horizontal_reference, pi, pi, config.altitude_reference, config.heading_reference, pi, pi, config.speed_difference_reference, config.health_reference, config.health_reference]
    kinds = ["nonnegative", "nonnegative", "nonnegative", "signed", "yaw", "signed", "signed", "signed", "nonnegative", "signed"]
    return [FeatureSpec(name, ref, kind) for name, ref, kind in zip(CRITIC_STATE_FEATURE_NAMES, refs, kinds)]  # type: ignore[arg-type]


def actor_observation_raw_1v1(own: UAVState, enemy: UAVState) -> NDArray[np.float64]:
    """Return the fixed 11-dimensional raw Actor vector."""

    displacement = enemy.position_vector() - own.position_vector()
    pitch, yaw = relative_angles(own, enemy)
    dv = own.velocity_vector() - enemy.velocity_vector()
    return np.asarray([
        own.x - enemy.x, own.y - enemy.y, own.z, np.linalg.norm(displacement), pitch, yaw,
        dv[0], dv[1], dv[2], safe_vector_angle(enemy.velocity_vector(), displacement),
        safe_vector_angle(own.velocity_vector(), displacement),
    ], dtype=np.float64)


def normalize_actor_observation_1v1(own: UAVState, enemy: UAVState, config: NormalizationConfig) -> NormalizationResult:
    """Return normalized Actor values and saturation diagnostics."""

    return normalize_by_specs(actor_observation_raw_1v1(own, enemy), actor_feature_specs(config), config)


def build_actor_observation_1v1(own_state: UAVState, enemy_state: UAVState, normalization_config: NormalizationConfig) -> NDArray[np.float64]:
    """Build the selected-mode 11-dimensional Actor observation."""

    return normalize_actor_observation_1v1(own_state, enemy_state, normalization_config).values


def critic_state_raw_1v1(own: UAVState, enemy: UAVState) -> NDArray[np.float64]:
    """Return the fixed 10-dimensional raw Critic vector."""

    displacement = enemy.position_vector() - own.position_vector()
    _, yaw = relative_angles(own, enemy)
    return np.asarray([
        np.linalg.norm(displacement), safe_vector_angle(own.velocity_vector(), displacement),
        safe_vector_angle(enemy.velocity_vector(), displacement), enemy.z - own.z, yaw,
        normalize_angle(enemy.heading_angle - own.heading_angle), normalize_angle(enemy.flight_path_angle - own.flight_path_angle),
        own.speed - enemy.speed, own.health, own.health - enemy.health,
    ], dtype=np.float64)


def build_critic_state_1v1(own_state: UAVState, enemy_state: UAVState, normalization_config: NormalizationConfig) -> NDArray[np.float64]:
    """Build the selected-mode 10-dimensional centralized Critic state."""

    return normalize_by_specs(critic_state_raw_1v1(own_state, enemy_state), critic_feature_specs(normalization_config), normalization_config).values


def build_single_observation(own_state: UAVState, enemy_state: UAVState, normalization_config: NormalizationConfig) -> NDArray[np.float64]:
    """Alias for the Actor observation."""

    return build_actor_observation_1v1(own_state, enemy_state, normalization_config)
