"""Fixed homogeneous 2v2 local observations adapted from 2024 entity blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from uav_env.entities.uav import UAV
from uav_env.observations.normalization import FeatureSpec, NormalizationConfig, normalize_by_specs
from uav_env.observations.single_observation import relative_angles, safe_vector_angle


ALLY_FEATURES = ["distance", "relative_pitch", "relative_yaw", "dvx", "dvy", "dvz"]
ENEMY_FEATURES = ["dx", "dy", "altitude_self", "distance", "relative_pitch", "relative_yaw", "dvx", "dvy", "dvz", "attack_angle", "escape_angle"]
MULTI_OBSERVATION_FEATURE_NAMES = [f"ally_0_{name}" for name in ALLY_FEATURES] + [f"enemy_{slot}_{name}" for slot in range(2) for name in ENEMY_FEATURES]


@dataclass(frozen=True)
class MultiObservationResult:
    """Raw/normalized local observations and fixed entity masks."""

    raw: NDArray[np.float64]
    normalized: NDArray[np.float64]
    ally_alive_masks: NDArray[np.int8]
    enemy_alive_masks: NDArray[np.int8]
    own_alive_mask: NDArray[np.int8]
    feature_names: list[str]


def multi_observation_specs(config: NormalizationConfig) -> list[FeatureSpec]:
    """Return the 28 explicit feature semantics."""

    h, a, v = config.horizontal_reference, config.angle_reference, config.speed_difference_reference
    ally = [
        FeatureSpec("distance", h, "nonnegative"), FeatureSpec("relative_pitch", a, "signed"),
        FeatureSpec("relative_yaw", config.heading_reference, "yaw"),
        *[FeatureSpec(name, v, "signed") for name in ("dvx", "dvy", "dvz")],
    ]
    enemy = [
        FeatureSpec("dx", h, "signed"), FeatureSpec("dy", h, "signed"),
        FeatureSpec("altitude_self", config.altitude_reference, "nonnegative"), FeatureSpec("distance", h, "nonnegative"),
        FeatureSpec("relative_pitch", a, "signed"), FeatureSpec("relative_yaw", config.heading_reference, "yaw"),
        *[FeatureSpec(name, v, "signed") for name in ("dvx", "dvy", "dvz")],
        FeatureSpec("attack_angle", a, "nonnegative"), FeatureSpec("escape_angle", a, "nonnegative"),
    ]
    return ally + enemy + enemy


def _ally_block(own: UAV, ally: UAV) -> NDArray[np.float64]:
    displacement = ally.state.position_vector() - own.state.position_vector()
    pitch, yaw = relative_angles(own.state, ally.state)
    dv = own.state.velocity_vector() - ally.state.velocity_vector()
    return np.asarray([np.linalg.norm(displacement), pitch, yaw, *dv], dtype=np.float64)


def _enemy_block(own: UAV, enemy: UAV) -> NDArray[np.float64]:
    displacement = enemy.state.position_vector() - own.state.position_vector()
    pitch, yaw = relative_angles(own.state, enemy.state)
    dv = own.state.velocity_vector() - enemy.state.velocity_vector()
    return np.asarray([
        own.state.x - enemy.state.x, own.state.y - enemy.state.y, own.state.z, np.linalg.norm(displacement), pitch, yaw,
        *dv, safe_vector_angle(own.state.velocity_vector(), displacement), safe_vector_angle(enemy.state.velocity_vector(), displacement),
    ], dtype=np.float64)


def build_multi_observations(red_aircraft: Sequence[UAV], blue_aircraft: Sequence[UAV], config: NormalizationConfig) -> MultiObservationResult:
    """Build two red-agent rows, each containing one ally and two enemy blocks."""

    reds = sorted(red_aircraft, key=lambda u: u.uav_id)
    blues = sorted(blue_aircraft, key=lambda u: u.uav_id)
    if len(reds) != 2 or len(blues) != 2:
        raise ValueError("The current multi observation supports exactly homogeneous 2v2")
    raw_rows: list[NDArray[np.float64]] = []
    norm_rows: list[NDArray[np.float64]] = []
    ally_masks: list[list[int]] = []
    enemy_masks: list[list[int]] = []
    specs = multi_observation_specs(config)
    for own in reds:
        ally = next(u for u in reds if u.uav_id != own.uav_id)
        ranked_blues = sorted(blues, key=lambda u: (not u.is_alive, float(np.linalg.norm(u.state.position_vector() - own.state.position_vector())), u.uav_id))
        ally_mask = int(ally.is_alive)
        enemy_mask = [int(u.is_alive) for u in ranked_blues]
        if not own.is_alive:
            raw = np.zeros(28, dtype=np.float64)
            normalized = raw.copy()
        else:
            raw = np.concatenate([
                _ally_block(own, ally) if ally.is_alive else np.zeros(6),
                *[_enemy_block(own, enemy) if enemy.is_alive else np.zeros(11) for enemy in ranked_blues],
            ]).astype(np.float64)
            normalized = normalize_by_specs(raw, specs, config).values
            if not ally.is_alive:
                normalized[:6] = 0.0
            for slot, enemy in enumerate(ranked_blues):
                if not enemy.is_alive:
                    normalized[6 + slot * 11: 6 + (slot + 1) * 11] = 0.0
        raw_rows.append(raw)
        norm_rows.append(normalized)
        ally_masks.append([ally_mask])
        enemy_masks.append(enemy_mask)
    return MultiObservationResult(
        np.stack(raw_rows), np.stack(norm_rows), np.asarray(ally_masks, dtype=np.int8),
        np.asarray(enemy_masks, dtype=np.int8), np.asarray([int(u.is_alive) for u in reds], dtype=np.int8),
        list(MULTI_OBSERVATION_FEATURE_NAMES),
    )


def build_multi_observation(observer: UAV, aircraft: Sequence[UAV]) -> NDArray[np.float64]:
    """Legacy single-row helper using the symmetric default configuration."""

    allies = [u for u in aircraft if u.team == observer.team]
    enemies = [u for u in aircraft if u.team != observer.team]
    result = build_multi_observations(allies, enemies, NormalizationConfig())
    index = sorted(allies, key=lambda u: u.uav_id).index(observer)
    return result.normalized[index]
