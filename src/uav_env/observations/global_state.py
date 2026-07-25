"""Fixed centralized state for homogeneous 2v2 and 3v3 combat."""

from __future__ import annotations

from dataclasses import dataclass
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


def global_state_feature_names(team_size: int) -> list[str]:
    """Return stable red-major centralized-state feature names."""

    return [f"red_{index}_failure" for index in range(team_size)] + [
        f"red_{red}_blue_{blue}_{name}"
        for red in range(team_size)
        for blue in range(team_size)
        for name in PAIR_FEATURES
    ] + [f"red_{index}_last_action" for index in range(team_size)]


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


def build_global_state_2v2(red_aircraft: Sequence[UAV], blue_aircraft: Sequence[UAV], config: NormalizationConfig, epsilon: float = 1.0) -> GlobalStateResult:
    """Backward-compatible 2v2 entry point."""

    if len(red_aircraft) != 2 or len(blue_aircraft) != 2:
        raise ValueError("build_global_state_2v2 requires exactly 2v2")
    return build_global_state(red_aircraft, blue_aircraft, config, epsilon)
