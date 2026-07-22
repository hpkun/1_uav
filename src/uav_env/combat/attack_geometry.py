"""Consistent red/blue attack and advantage geometry."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np
from numpy.typing import NDArray

from uav_env.core.geometry import angle_between
from uav_env.core.state import UAVState


@dataclass(frozen=True)
class AttackZoneConfig:
    """Distance and angular definitions of attack and advantage regions."""

    attack_distance_min: float
    attack_distance_max: float
    attack_angle_max: float
    escape_angle_max: float
    attack_area_angle_max: float
    advantage_distance_min: float
    advantage_distance_max: float
    advantage_escape_angle_max: float

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in self.__dict__.values())
        if not all(isfinite(value) for value in values):
            raise ValueError("Attack-zone parameters must be finite")
        if not 0.0 <= self.attack_distance_min < self.attack_distance_max:
            raise ValueError("Attack-distance interval is invalid")
        if not 0.0 <= self.advantage_distance_min < self.advantage_distance_max:
            raise ValueError("Advantage-distance interval is invalid")

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AttackZoneConfig":
        """Build attack geometry settings from an experiment config."""

        return cls(**{key: float(config[key]) for key in cls.__dataclass_fields__})


@dataclass(frozen=True)
class CombatGeometry:
    """Geometry from one attacker toward one target."""

    distance: float
    line_of_sight: NDArray[np.float64]
    attacker_attack_angle: float
    target_escape_angle: float
    in_attack_area: bool
    in_advantage_area: bool
    can_attack: bool

    def __eq__(self, other: object) -> bool:
        """Compare array-bearing geometry records without ambiguous NumPy truth values."""

        if not isinstance(other, CombatGeometry):
            return NotImplemented
        return (
            self.distance == other.distance
            and np.array_equal(self.line_of_sight, other.line_of_sight)
            and self.attacker_attack_angle == other.attacker_attack_angle
            and self.target_escape_angle == other.target_escape_angle
            and self.in_attack_area == other.in_attack_area
            and self.in_advantage_area == other.in_advantage_area
            and self.can_attack == other.can_attack
        )


def _protected_angle(first: NDArray[np.float64], second: NDArray[np.float64]) -> float:
    if np.linalg.norm(first) <= 1.0e-12 or np.linalg.norm(second) <= 1.0e-12:
        return 0.0
    return angle_between(first, second)


def compute_combat_geometry(
    attacker: UAVState,
    target: UAVState,
    config: AttackZoneConfig,
) -> CombatGeometry:
    """Compute attack geometry using the line of sight from attacker to target."""

    displacement = target.position_vector() - attacker.position_vector()
    distance = float(np.linalg.norm(displacement))
    line_of_sight = displacement / distance if distance > 1.0e-12 else np.zeros(3, dtype=np.float64)
    attack_angle = _protected_angle(attacker.velocity_vector(), displacement)
    escape_angle = _protected_angle(target.velocity_vector(), displacement)
    in_attack_distance = config.attack_distance_min <= distance <= config.attack_distance_max
    in_attack_area = in_attack_distance and attack_angle <= config.attack_area_angle_max
    in_advantage_area = (
        config.advantage_distance_min <= distance <= config.advantage_distance_max
        and escape_angle <= config.advantage_escape_angle_max
    )
    can_attack = in_attack_distance and attack_angle <= config.attack_angle_max and escape_angle <= config.escape_angle_max
    return CombatGeometry(
        distance=distance,
        line_of_sight=line_of_sight,
        attacker_attack_angle=attack_angle,
        target_escape_angle=escape_angle,
        in_attack_area=in_attack_area,
        in_advantage_area=in_advantage_area,
        can_attack=can_attack,
    )


def is_in_attack_zone(attacker: UAVState, target: UAVState, config: AttackZoneConfig) -> bool:
    """Return whether the complete attack conditions are satisfied."""

    return compute_combat_geometry(attacker, target, config).can_attack
