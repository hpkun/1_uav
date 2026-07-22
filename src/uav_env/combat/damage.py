"""Damage configuration and future probability interface."""

from __future__ import annotations

from dataclasses import dataclass

from uav_env.core.state import UAVState


@dataclass(frozen=True)
class DamageConfig:
    """Bounds for a future probabilistic health-damage model."""

    min_damage: float
    max_damage: float
    hit_probability: float


def apply_damage(target: UAVState, config: DamageConfig, random_value: float) -> UAVState:
    """Return a damaged state using a future probabilistic model."""

    raise NotImplementedError("Probabilistic damage is not implemented")
