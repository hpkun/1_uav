"""Fixed homogeneous 2v2/3v3 local observations using stable entity blocks."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, cos, pi, sin
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from uav_env.entities.uav import UAV
from uav_env.core.geometry import normalize_angle
from uav_env.combat.attack_geometry import compute_combat_geometry
from uav_env.observations.normalization import FeatureSpec, NormalizationConfig, normalize_by_specs
from uav_env.observations.single_observation import relative_angles, safe_vector_angle


ALLY_FEATURES = ["distance", "relative_pitch", "relative_yaw", "dvx", "dvy", "dvz"]
ENEMY_FEATURES = ["dx", "dy", "altitude_self", "distance", "relative_pitch", "relative_yaw", "dvx", "dvy", "dvz", "attack_angle", "escape_angle"]
MULTI_OBSERVATION_FEATURE_NAMES = [f"ally_0_{name}" for name in ALLY_FEATURES] + [f"enemy_{slot}_{name}" for slot in range(2) for name in ENEMY_FEATURES]
V2_OWN_FEATURES = ["own_altitude", "own_speed", "own_flight_path_angle", "own_heading_sin", "own_heading_cos", "own_health_ratio", "own_last_action", "episode_progress"]
V2_ALLY_FEATURES = ["alive_flag", "body_relative_x", "body_relative_y", "relative_z", "body_relative_vx", "body_relative_vy", "relative_vz", "health_ratio"]
V2_ENEMY_FEATURES = [
    "alive_flag", "body_relative_x", "body_relative_y", "relative_z", "body_relative_vx", "body_relative_vy", "relative_vz",
    "distance", "body_relative_bearing", "body_relative_elevation", "attack_angle", "escape_angle", "health_ratio",
]
V2_OWN_SIZE = len(V2_OWN_FEATURES)
V2_ALLY_SIZE = len(V2_ALLY_FEATURES)
V2_ENEMY_SIZE = len(V2_ENEMY_FEATURES)
V2_LOCAL_OBSERVATION_DIM = V2_OWN_SIZE + 2 * V2_ALLY_SIZE + 3 * V2_ENEMY_SIZE


def multi_observation_feature_names(ally_count: int, enemy_count: int) -> list[str]:
    """Return stable slot-major feature names for a fixed team size."""

    return [f"ally_{slot}_{name}" for slot in range(ally_count) for name in ALLY_FEATURES] + [
        f"enemy_{slot}_{name}" for slot in range(enemy_count) for name in ENEMY_FEATURES
    ]


def multi_observation_feature_names_v2() -> list[str]:
    """Return the fixed 63D homogeneous 3v3 time-aware V2 local-observation names."""

    return [
        *V2_OWN_FEATURES,
        *[f"ally_{slot}_{name}" for slot in range(2) for name in V2_ALLY_FEATURES],
        *[f"blue_{slot}_{name}" for slot in range(3) for name in V2_ENEMY_FEATURES],
    ]


def multi_observation_feature_names_v2_for_agent(red_id: str) -> list[str]:
    """Return row-specific 63D names with true fixed entity IDs for audit/debug."""

    red_ids = ["red_0", "red_1", "red_2"]
    if red_id not in red_ids:
        raise ValueError(f"Unknown fixed 3v3 red id: {red_id!r}")
    ally_ids = [entity_id for entity_id in red_ids if entity_id != red_id]
    return [
        *[f"{red_id}_{name}" for name in V2_OWN_FEATURES],
        *[f"{ally_id}_{name}" for ally_id in ally_ids for name in V2_ALLY_FEATURES],
        *[f"blue_{slot}_{name}" for slot in range(3) for name in V2_ENEMY_FEATURES],
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
    feature_names_by_agent: dict[str, list[str]] = field(default_factory=dict)


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


def _body_relative_kinematics(own: UAV, target: UAV) -> tuple[float, float, float, float, float, float, float, float, float]:
    dx = target.state.x - own.state.x
    dy = target.state.y - own.state.y
    dz = target.state.z - own.state.z
    c, s = cos(own.state.heading_angle), sin(own.state.heading_angle)
    body_x = c * dx + s * dy
    body_y = -s * dx + c * dy
    own_v = own.state.velocity_vector()
    target_v = target.state.velocity_vector()
    dv = target_v - own_v
    body_vx = c * float(dv[0]) + s * float(dv[1])
    body_vy = -s * float(dv[0]) + c * float(dv[1])
    distance = float(np.linalg.norm(target.state.position_vector() - own.state.position_vector()))
    bearing = normalize_angle(atan2(body_y, body_x))
    elevation = normalize_angle(atan2(dz, float(np.hypot(body_x, body_y))) - own.state.flight_path_angle)
    return body_x, body_y, dz, body_vx, body_vy, float(dv[2]), distance, bearing, elevation


def _normalize_v2(raw: NDArray[np.float64], names: list[str], config: dict[str, object]) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    max_altitude = float(config["max_altitude"])
    min_speed = float(config["min_speed"])
    max_speed = float(config["max_speed"])
    speed_span = max(max_speed - min_speed, 1.0e-12)
    local_reference = float(config.get("local_position_reference", config["desired_distance_max"]))
    speed_reference = float(config.get("speed_difference_reference", speed_span))
    max_theta = max(abs(float(config["min_flight_path_angle"])), abs(float(config["max_flight_path_angle"])), 1.0e-12)
    normalized = np.zeros_like(raw)
    for index, (name, value) in enumerate(zip(names, raw)):
        if name.endswith("alive_flag") or name in {"own_heading_sin", "own_heading_cos"}:
            transformed = value
        elif name.endswith("health_ratio") or name == "own_health_ratio":
            transformed = 2.0 * value - 1.0
        elif name == "own_last_action" or name.endswith("last_action"):
            transformed = 2.0 * value / 14.0 - 1.0
        elif name == "episode_progress":
            transformed = 2.0 * float(np.clip(value, 0.0, 1.0)) - 1.0
        elif name == "own_speed":
            transformed = 2.0 * (value - min_speed) / speed_span - 1.0
        elif name == "own_altitude":
            transformed = 2.0 * value / max_altitude - 1.0
        elif name == "own_flight_path_angle":
            transformed = value / max_theta
        elif name.endswith("body_relative_x") or name.endswith("body_relative_y"):
            transformed = value / local_reference
        elif name.endswith("relative_z"):
            transformed = value / max_altitude
        elif name.endswith("body_relative_vx") or name.endswith("body_relative_vy") or name.endswith("relative_vz"):
            transformed = value / speed_reference
        elif name.endswith("body_relative_bearing") or name.endswith("body_relative_elevation"):
            transformed = value / pi
        elif name.endswith("attack_angle") or name.endswith("escape_angle"):
            transformed = 2.0 * value / pi - 1.0
        elif name.endswith("distance"):
            transformed = 2.0 * value / local_reference - 1.0
        else:
            raise ValueError(f"Unknown V2 local observation feature: {name}")
        normalized[index] = transformed
    saturated = np.abs(normalized) > 1.0
    return np.clip(normalized, -1.0, 1.0), saturated


def _clear_dead_v2_slots(
    normalized: NDArray[np.float64],
    saturated: NDArray[np.bool_],
    allies: Sequence[UAV],
    enemies: Sequence[UAV],
) -> None:
    """Ensure dead entity slots normalize to alive=-1 and all other fields zero."""

    for slot, ally in enumerate(allies):
        if not ally.is_alive:
            start = V2_OWN_SIZE + slot * V2_ALLY_SIZE
            end = start + V2_ALLY_SIZE
            normalized[start] = -1.0
            normalized[start + 1:end] = 0.0
            saturated[start:end] = False
    enemy_offset = V2_OWN_SIZE + 2 * V2_ALLY_SIZE
    for slot, enemy in enumerate(enemies):
        if not enemy.is_alive:
            start = enemy_offset + slot * V2_ENEMY_SIZE
            end = start + V2_ENEMY_SIZE
            normalized[start] = -1.0
            normalized[start + 1:end] = 0.0
            saturated[start:end] = False


def _ally_block_v2(own: UAV, ally: UAV, config: dict[str, object]) -> NDArray[np.float64]:
    if not ally.is_alive:
        return np.asarray([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    body_x, body_y, dz, body_vx, body_vy, relative_vz, _, _, _ = _body_relative_kinematics(own, ally)
    return np.asarray([1.0, body_x, body_y, dz, body_vx, body_vy, relative_vz, ally.state.health / float(config["initial_health"])], dtype=np.float64)


def _enemy_block_v2(own: UAV, enemy: UAV, attack_config: object, config: dict[str, object]) -> NDArray[np.float64]:
    if not enemy.is_alive:
        return np.asarray([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    body_x, body_y, dz, body_vx, body_vy, relative_vz, distance, bearing, elevation = _body_relative_kinematics(own, enemy)
    geometry = compute_combat_geometry(own.state, enemy.state, attack_config)  # type: ignore[arg-type]
    return np.asarray([
        1.0, body_x, body_y, dz, body_vx, body_vy, relative_vz, distance, bearing, elevation,
        geometry.attacker_attack_angle, geometry.target_escape_angle, enemy.state.health / float(config["initial_health"]),
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


def build_multi_observations_v2(
    red_aircraft: Sequence[UAV],
    blue_aircraft: Sequence[UAV],
    config: dict[str, object],
    attack_config: object,
    episode_progress: float,
) -> MultiObservationResult:
    """Build fixed-ID body-frame 63D observations for homogeneous 3v3 time-aware V2."""

    reds = sorted(red_aircraft, key=lambda u: u.uav_id)
    blues = sorted(blue_aircraft, key=lambda u: u.uav_id)
    if len(reds) != 3 or len(blues) != 3:
        raise ValueError("V2 local observations require fixed homogeneous 3v3")
    names = multi_observation_feature_names_v2()
    raw_rows: list[NDArray[np.float64]] = []
    norm_rows: list[NDArray[np.float64]] = []
    saturated_masks: list[NDArray[np.bool_]] = []
    ally_masks: list[list[int]] = []
    enemy_masks: list[list[int]] = []
    for own in reds:
        if not own.is_alive:
            raw = np.zeros(V2_LOCAL_OBSERVATION_DIM, dtype=np.float64)
            normalized = raw.copy()
            saturated = np.zeros(V2_LOCAL_OBSERVATION_DIM, dtype=bool)
        else:
            own_action = float(own.state.last_action if own.state.last_action is not None else 0.0)
            own_block = np.asarray([
                own.state.z, own.state.speed, own.state.flight_path_angle, sin(own.state.heading_angle),
                cos(own.state.heading_angle), own.state.health / float(config["initial_health"]), own_action, episode_progress,
            ], dtype=np.float64)
            allies = [ally for ally in reds if ally.uav_id != own.uav_id]
            raw = np.concatenate([
                own_block,
                *[_ally_block_v2(own, ally, config) for ally in allies],
                *[_enemy_block_v2(own, enemy, attack_config, config) for enemy in blues],
            ]).astype(np.float64)
            normalized, saturated = _normalize_v2(raw, names, config)
            _clear_dead_v2_slots(normalized, saturated, allies, blues)
        raw_rows.append(raw)
        norm_rows.append(normalized)
        saturated_masks.append(saturated)
        ally_masks.append([int(ally.is_alive) for ally in reds if ally.uav_id != own.uav_id])
        enemy_masks.append([int(enemy.is_alive) for enemy in blues])
    saturation_counts = np.asarray([int(np.count_nonzero(mask)) for mask in saturated_masks], dtype=np.int64)
    return MultiObservationResult(
        np.stack(raw_rows), np.stack(norm_rows), np.asarray(ally_masks, dtype=np.int8),
        np.asarray(enemy_masks, dtype=np.int8), np.asarray([int(u.is_alive) for u in reds], dtype=np.int8),
        names, saturation_counts, saturation_counts.astype(np.float64) / float(V2_LOCAL_OBSERVATION_DIM), np.stack(saturated_masks),
        [[name for name, flag in zip(names, mask) if flag] for mask in saturated_masks],
        {red.uav_id: multi_observation_feature_names_v2_for_agent(red.uav_id) for red in reds},
    )


def build_multi_observation(observer: UAV, aircraft: Sequence[UAV]) -> NDArray[np.float64]:
    """Legacy single-row helper using the symmetric default configuration."""

    allies = [u for u in aircraft if u.team == observer.team]
    enemies = [u for u in aircraft if u.team != observer.team]
    result = build_multi_observations(allies, enemies, NormalizationConfig())
    index = sorted(allies, key=lambda u: u.uav_id).index(observer)
    return result.normalized[index]
