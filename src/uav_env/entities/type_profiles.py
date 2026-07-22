"""Static UAV type profiles."""

from __future__ import annotations

from dataclasses import dataclass


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
    detection_range: float
    attack_range: float
    initial_health: float

    def __post_init__(self) -> None:
        if not self.type_id:
            raise ValueError("type_id must not be empty")
        if self.min_speed <= 0.0 or self.max_speed <= self.min_speed:
            raise ValueError("Speed limits are invalid")
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
        detection_range=20_000.0,
        attack_range=2_000.0,
        initial_health=300.0,
    )
