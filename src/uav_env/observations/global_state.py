"""Fixed centralized state for homogeneous 2v2 and 3v3 combat."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from uav_env.actions.discrete_15 import DiscreteAction15
from uav_env.entities.uav import UAV
from uav_env.observations.normalization import FeatureSpec, NormalizationConfig, normalize_by_specs
from uav_env.observations.single_observation import relative_angles, safe_vector_angle


PAIR_FEATURES = ["distance", "relative_pitch", "relative_yaw", "dvx", "dvy", "dvz", "velocity_vector_angle", "attack_angle", "escape_angle"]
GLOBAL_STATE_FEATURE_NAMES = ["red_0_failure", "red_1_failure"] + [
    f"red_{r}_blue_{b}_{name}" for r in range(2) for b in range(2) for name in PAIR_FEATURES
] + ["red_0_last_action", "red_1_last_action"]
V2_ENTITY_FEATURES = [
    "alive_flag", "health_ratio", "absolute_x", "absolute_y", "absolute_z", "speed",
    "flight_path_angle", "heading_sin", "heading_cos", "last_action",
]


def global_state_feature_names(team_size: int) -> list[str]:
    """Return stable red-major centralized-state feature names."""

    return [f"red_{index}_failure" for index in range(team_size)] + [
        f"red_{red}_blue_{blue}_{name}"
        for red in range(team_size)
        for blue in range(team_size)
        for name in PAIR_FEATURES
    ] + [f"red_{index}_last_action" for index in range(team_size)]


def global_state_feature_names_v2() -> list[str]:
    """Return the fixed 60D full-entity V2 global-state names."""

    return [f"{prefix}_{index}_{name}" for prefix in ("red", "blue") for index in range(3) for name in V2_ENTITY_FEATURES]


@dataclass(frozen=True)
class GlobalStateResult:
    """Raw and normalized 2v2 centralized state."""

    raw: NDArray[np.float64]
    normalized: NDArray[np.float64]
    feature_names: list[str]
    saturation_count: int
    saturation_ratio: float
    saturated_feature_mask: NDArray[np.bool_]
    saturated_feature_names: list[str]


def _pair_block(red: UAV, blue: UAV) -> NDArray[np.float64]:
    displacement = blue.state.position_vector() - red.state.position_vector()
    pitch, yaw = relative_angles(red.state, blue.state)
    red_velocity, blue_velocity = red.state.velocity_vector(), blue.state.velocity_vector()
    dv = red_velocity - blue_velocity
    return np.asarray([
        np.linalg.norm(displacement), pitch, yaw, *dv, safe_vector_angle(red_velocity, blue_velocity),
        safe_vector_angle(red_velocity, displacement), safe_vector_angle(blue_velocity, displacement),
    ], dtype=np.float64)


def build_global_state(red_aircraft: Sequence[UAV], blue_aircraft: Sequence[UAV], config: NormalizationConfig, epsilon: float = 1.0) -> GlobalStateResult:
    """Build fixed red-major pair ordering for equal team sizes of 2 or 3."""

    reds, blues = sorted(red_aircraft, key=lambda u: u.uav_id), sorted(blue_aircraft, key=lambda u: u.uav_id)
    if len(reds) != len(blues) or len(reds) not in {2, 3}:
        raise ValueError("Global state supports equal homogeneous team sizes of 2 or 3")
    team_size = len(reds)
    failures = [epsilon * ((-1.0) ** int(u.state.damaged)) for u in reds]
    pairs = np.concatenate([_pair_block(red, blue) for red in reds for blue in blues])
    actions = [float(u.state.last_action if u.state.last_action is not None else int(DiscreteAction15.LEVEL_HOLD)) for u in reds]
    raw = np.asarray([*failures, *pairs, *actions], dtype=np.float64)
    pair_specs = [
        FeatureSpec("distance", config.horizontal_reference, "nonnegative"), FeatureSpec("relative_pitch", np.pi, "signed"),
        FeatureSpec("relative_yaw", config.heading_reference, "yaw", np.pi),
        *[FeatureSpec(name, config.speed_difference_reference, "signed") for name in ("dvx", "dvy", "dvz")],
        *[FeatureSpec(name, np.pi, "nonnegative") for name in ("velocity_vector_angle", "attack_angle", "escape_angle")],
    ]
    specs = [FeatureSpec("red_failure", 1.0, "failure") for _ in range(team_size)] + pair_specs * (team_size * team_size) + [FeatureSpec("last_action", 14.0, "action") for _ in range(team_size)]
    result = normalize_by_specs(raw, specs, config)
    names = global_state_feature_names(team_size)
    return GlobalStateResult(raw, result.values, names, result.saturation_count, result.saturation_ratio,
                             result.saturated_mask, [name for name, flag in zip(names, result.saturated_mask) if flag])


def _normalize_v2_global(raw: NDArray[np.float64], names: list[str], config: dict[str, object]) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    global_xy_reference = float(config.get("global_xy_reference", float(config["max_speed"]) * float(config["max_episode_seconds"])))
    max_altitude = float(config["max_altitude"])
    min_speed = float(config["min_speed"])
    max_speed = float(config["max_speed"])
    speed_span = max(max_speed - min_speed, 1.0e-12)
    max_theta = max(abs(float(config["min_flight_path_angle"])), abs(float(config["max_flight_path_angle"])), 1.0e-12)
    output = np.zeros_like(raw)
    for index, (name, value) in enumerate(zip(names, raw)):
        if name.endswith("alive_flag") or name.endswith("heading_sin") or name.endswith("heading_cos"):
            transformed = value
        elif name.endswith("health_ratio"):
            transformed = 2.0 * value - 1.0
        elif name.endswith("absolute_x") or name.endswith("absolute_y"):
            transformed = value / global_xy_reference
        elif name.endswith("absolute_z"):
            transformed = 2.0 * value / max_altitude - 1.0
        elif name.endswith("speed"):
            transformed = 2.0 * (value - min_speed) / speed_span - 1.0
        elif name.endswith("flight_path_angle"):
            transformed = value / max_theta
        elif name.endswith("last_action"):
            transformed = 2.0 * value / 14.0 - 1.0
        else:
            raise ValueError(f"Unknown V2 global-state feature: {name}")
        output[index] = transformed
    saturated = np.abs(output) > 1.0
    return np.clip(output, -1.0, 1.0), saturated


def _entity_block_v2(aircraft: UAV, config: dict[str, object]) -> NDArray[np.float64]:
    if not aircraft.is_alive:
        return np.asarray([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(DiscreteAction15.LEVEL_HOLD)], dtype=np.float64)
    last_action = float(aircraft.state.last_action if aircraft.state.last_action is not None else int(DiscreteAction15.LEVEL_HOLD))
    return np.asarray([
        1.0, aircraft.state.health / float(config["initial_health"]), aircraft.state.x, aircraft.state.y, aircraft.state.z,
        aircraft.state.speed, aircraft.state.flight_path_angle, sin(aircraft.state.heading_angle), cos(aircraft.state.heading_angle),
        last_action,
    ], dtype=np.float64)


def build_global_state_v2(red_aircraft: Sequence[UAV], blue_aircraft: Sequence[UAV], config: dict[str, object]) -> GlobalStateResult:
    """Build fixed-ID 60D full-entity global state for homogeneous 3v3 V2."""

    reds, blues = sorted(red_aircraft, key=lambda u: u.uav_id), sorted(blue_aircraft, key=lambda u: u.uav_id)
    if len(reds) != 3 or len(blues) != 3:
        raise ValueError("V2 global state requires fixed homogeneous 3v3")
    names = global_state_feature_names_v2()
    raw = np.concatenate([*[_entity_block_v2(u, config) for u in reds], *[_entity_block_v2(u, config) for u in blues]]).astype(np.float64)
    normalized, saturated = _normalize_v2_global(raw, names, config)
    saturation_count = int(np.count_nonzero(saturated))
    return GlobalStateResult(
        raw, normalized, names, saturation_count, saturation_count / 60.0,
        saturated, [name for name, flag in zip(names, saturated) if flag],
    )


def build_global_state_2v2(red_aircraft: Sequence[UAV], blue_aircraft: Sequence[UAV], config: NormalizationConfig, epsilon: float = 1.0) -> GlobalStateResult:
    """Backward-compatible 2v2 entry point."""

    if len(red_aircraft) != 2 or len(blue_aircraft) != 2:
        raise ValueError("build_global_state_2v2 requires exactly 2v2")
    return build_global_state(red_aircraft, blue_aircraft, config, epsilon)
