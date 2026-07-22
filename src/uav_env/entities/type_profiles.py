"""Static UAV type profiles."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi
from typing import Any


@dataclass(frozen=True)
class UAVTypeProfile:
    """Physical and sensing limits for a UAV type."""

    type_id: str
    min_speed: float
    max_speed: float
    min_tangential_overload: float
    max_tangential_overload: float
    min_normal_overload: float
    max_normal_overload: float
    min_flight_path_angle: float
    max_flight_path_angle: float
    detection_range: float
    attack_range: float
    initial_health: float

    def __post_init__(self) -> None:
        if not self.type_id:
            raise ValueError("type_id must not be empty")
        numeric = (
            self.min_speed,
            self.max_speed,
            self.min_tangential_overload,
            self.max_tangential_overload,
            self.min_normal_overload,
            self.max_normal_overload,
            self.min_flight_path_angle,
            self.max_flight_path_angle,
            self.detection_range,
            self.attack_range,
            self.initial_health,
        )
        if not all(isfinite(value) for value in numeric):
            raise ValueError("Profile values must be finite")
        if self.min_speed <= 0.0 or self.max_speed <= self.min_speed:
            raise ValueError("Speed limits are invalid")
        if self.min_flight_path_angle >= self.max_flight_path_angle:
            raise ValueError("Flight-path-angle limits are invalid")
        if self.detection_range <= 0.0 or self.attack_range <= 0.0:
            raise ValueError("Range limits must be positive")
        if self.initial_health <= 0.0:
            raise ValueError("initial_health must be positive")


def homogeneous_baseline() -> UAVTypeProfile:
    """Return the sole baseline type profile for the current project phase."""

    return UAVTypeProfile(
        type_id="homogeneous_baseline",
        min_speed=30.0,
        max_speed=150.0,
        min_tangential_overload=-1.0,
        max_tangential_overload=2.5,
        min_normal_overload=-4.0,
        max_normal_overload=4.0,
        min_flight_path_angle=-pi / 4.0,
        max_flight_path_angle=pi / 4.0,
        detection_range=20_000.0,
        attack_range=2_000.0,
        initial_health=300.0,
    )


def homogeneous_2023_profile() -> UAVTypeProfile:
    """Return the homogeneous platform parameters reported for the 2023 setup."""

    return homogeneous_baseline()


def homogeneous_2024_profile() -> UAVTypeProfile:
    """Return the homogeneous 2024 profile with the specified 180 m/s limit."""

    profile = homogeneous_baseline()
    return UAVTypeProfile(**{**profile.__dict__, "type_id": "homogeneous_2024", "max_speed": 180.0})


def profile_from_config(config: dict[str, Any], type_id: str | None = None) -> UAVTypeProfile:
    """Construct a homogeneous profile from a validated experiment config."""

    return UAVTypeProfile(
        type_id=type_id or str(config.get("profile_name", "homogeneous_2024")),
        min_speed=float(config["min_speed"]),
        max_speed=float(config["max_speed"]),
        min_tangential_overload=float(config["min_tangential_overload"]),
        max_tangential_overload=float(config["max_tangential_overload"]),
        min_normal_overload=float(config["min_normal_overload"]),
        max_normal_overload=float(config["max_normal_overload"]),
        min_flight_path_angle=float(config["min_flight_path_angle"]),
        max_flight_path_angle=float(config["max_flight_path_angle"]),
        detection_range=float(config.get("detection_range", config["desired_distance_max"])),
        attack_range=float(config["attack_distance_max"]),
        initial_health=float(config["initial_health"]),
    )
