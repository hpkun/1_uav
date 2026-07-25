"""Fixed homogeneous 2v2/3v3 local observations using stable entity blocks."""

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


def multi_observation_feature_names(ally_count: int, enemy_count: int) -> list[str]:
    """Return stable slot-major feature names for a fixed team size."""

    return [f"ally_{slot}_{name}" for slot in range(ally_count) for name in ALLY_FEATURES] + [
        f"enemy_{slot}_{name}" for slot in range(enemy_count) for name in ENEMY_FEATURES
    ]


@dataclass(frozen=True)
class MultiObservationResult:
    """Raw/normalized local observations and fixed entity masks."""

    raw: NDArray[np.float64]
    normalized: NDArray[np.float64]
    ally_alive_masks: NDArray[np.int8]
    enemy_alive_masks: NDArray[np.int8]
    own_alive_mask: NDArray[np.int8]
    feature_names: list[str]
    saturation_count: NDArray[np.int64]
    saturation_ratio: NDArray[np.float64]
    saturated_feature_masks: NDArray[np.bool_]
    saturated_feature_names: list[list[str]]


def multi_observation_specs(config: NormalizationConfig, ally_count: int = 1, enemy_count: int = 2) -> list[FeatureSpec]:
    """Return explicit repeated entity-block normalization semantics."""

    h, a, v = config.horizontal_reference, config.angle_reference, config.speed_difference_reference
    ally = [
        FeatureSpec("distance", h, "nonnegative"), FeatureSpec("relative_pitch", a, "signed"),
        FeatureSpec("relative_yaw", config.heading_reference, "yaw", np.pi),
        *[FeatureSpec(name, v, "signed") for name in ("dvx", "dvy", "dvz")],
    ]
    enemy = [
        FeatureSpec("dx", h, "signed"), FeatureSpec("dy", h, "signed"),
        FeatureSpec("altitude_self", config.altitude_reference, "nonnegative"), FeatureSpec("distance", h, "nonnegative"),
        FeatureSpec("relative_pitch", a, "signed"), FeatureSpec("relative_yaw", config.heading_reference, "yaw", np.pi),
        *[FeatureSpec(name, v, "signed") for name in ("dvx", "dvy", "dvz")],
        FeatureSpec("attack_angle", a, "nonnegative"), FeatureSpec("escape_angle", a, "nonnegative"),
    ]
    return ally * ally_count + enemy * enemy_count


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
    """Build fixed red-agent rows for homogeneous 2v2 or 3v3 combat."""

    reds = sorted(red_aircraft, key=lambda u: u.uav_id)
    blues = sorted(blue_aircraft, key=lambda u: u.uav_id)
    if len(reds) != len(blues) or len(reds) not in {2, 3}:
        raise ValueError("Multi observations support equal homogeneous team sizes of 2 or 3")
    ally_count, enemy_count = len(reds) - 1, len(blues)
    observation_dim = ally_count * len(ALLY_FEATURES) + enemy_count * len(ENEMY_FEATURES)
    feature_names = multi_observation_feature_names(ally_count, enemy_count)
    raw_rows: list[NDArray[np.float64]] = []
    norm_rows: list[NDArray[np.float64]] = []
    ally_masks: list[list[int]] = []
    enemy_masks: list[list[int]] = []
    saturation_counts: list[int] = []
    saturation_ratios: list[float] = []
    saturation_masks: list[NDArray[np.bool_]] = []
    specs = multi_observation_specs(config, ally_count, enemy_count)
    for own in reds:
        ranked_allies = sorted(
            (u for u in reds if u.uav_id != own.uav_id),
            key=lambda u: (not u.is_alive, float(np.linalg.norm(u.state.position_vector() - own.state.position_vector())), u.uav_id),
        )
        ranked_blues = sorted(blues, key=lambda u: (not u.is_alive, float(np.linalg.norm(u.state.position_vector() - own.state.position_vector())), u.uav_id))
        ally_mask = [int(u.is_alive) for u in ranked_allies]
        enemy_mask = [int(u.is_alive) for u in ranked_blues]
        if not own.is_alive:
            raw = np.zeros(observation_dim, dtype=np.float64)
            normalized = raw.copy()
            saturated = np.zeros(observation_dim, dtype=bool)
            saturation_count, saturation_ratio = 0, 0.0
        else:
            raw = np.concatenate([
                *[_ally_block(own, ally) if ally.is_alive else np.zeros(6) for ally in ranked_allies],
                *[_enemy_block(own, enemy) if enemy.is_alive else np.zeros(11) for enemy in ranked_blues],
            ]).astype(np.float64)
            result = normalize_by_specs(raw, specs, config)
            normalized = result.values
            saturated = result.saturated_mask.copy()
            for slot, ally in enumerate(ranked_allies):
                if not ally.is_alive:
                    normalized[slot * 6: (slot + 1) * 6] = 0.0
                    saturated[slot * 6: (slot + 1) * 6] = False
            enemy_offset = ally_count * 6
            for slot, enemy in enumerate(ranked_blues):
                if not enemy.is_alive:
                    normalized[enemy_offset + slot * 11: enemy_offset + (slot + 1) * 11] = 0.0
                    saturated[enemy_offset + slot * 11: enemy_offset + (slot + 1) * 11] = False
            saturation_count = int(np.count_nonzero(saturated))
            saturation_ratio = saturation_count / float(observation_dim)
        raw_rows.append(raw)
        norm_rows.append(normalized)
        ally_masks.append(ally_mask)
        enemy_masks.append(enemy_mask)
        saturation_counts.append(saturation_count)
        saturation_ratios.append(saturation_ratio)
        saturation_masks.append(saturated)
    return MultiObservationResult(
        np.stack(raw_rows), np.stack(norm_rows), np.asarray(ally_masks, dtype=np.int8),
        np.asarray(enemy_masks, dtype=np.int8), np.asarray([int(u.is_alive) for u in reds], dtype=np.int8),
        feature_names, np.asarray(saturation_counts, dtype=np.int64),
        np.asarray(saturation_ratios, dtype=np.float64), np.stack(saturation_masks),
        [[name for name, flag in zip(feature_names, mask) if flag] for mask in saturation_masks],
    )


def build_multi_observation(observer: UAV, aircraft: Sequence[UAV]) -> NDArray[np.float64]:
    """Legacy single-row helper using the symmetric default configuration."""

    allies = [u for u in aircraft if u.team == observer.team]
    enemies = [u for u in aircraft if u.team != observer.team]
    result = build_multi_observations(allies, enemies, NormalizationConfig())
    index = sorted(allies, key=lambda u: u.uav_id).index(observer)
    return result.normalized[index]
