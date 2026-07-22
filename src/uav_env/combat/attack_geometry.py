"""Attack-zone configuration and future judgement interface."""

from __future__ import annotations

from dataclasses import dataclass

from uav_env.core.state import UAVState


@dataclass(frozen=True)
class AttackZoneConfig:
    """Distance and angle constraints for an attack zone."""

    max_range: float
    max_attack_angle: float
    max_escape_angle: float


def is_in_attack_zone(attacker: UAVState, target: UAVState, config: AttackZoneConfig) -> bool:
    """Determine whether *target* lies in *attacker*'s attack zone."""

    raise NotImplementedError("Attack-zone judgement is not implemented")
