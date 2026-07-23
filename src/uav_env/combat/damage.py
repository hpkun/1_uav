"""Reproducible piecewise nominal and effective health damage."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from typing import Any, Sequence

import numpy as np

from uav_env.core.state import UAVState


@dataclass(frozen=True)
class DamageConfig:
    """Cumulative probability thresholds and corresponding nominal damage."""

    probability_thresholds: tuple[float, float, float] = (0.1, 0.4, 0.8)
    damage_values: tuple[float, float, float, float] = (51.0, 21.0, 11.0, 0.0)

    def __post_init__(self) -> None:
        if len(self.probability_thresholds) != 3 or len(self.damage_values) != 4:
            raise ValueError("Damage model requires three thresholds and four values")
        if not 0.0 < self.probability_thresholds[0] < self.probability_thresholds[1] < self.probability_thresholds[2] < 1.0:
            raise ValueError("Damage probability thresholds must increase within (0, 1)")
        if not all(isfinite(value) and value >= 0.0 for value in self.damage_values):
            raise ValueError("Damage values must be finite and non-negative")

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DamageConfig":
        """Build the damage model from an experiment config."""

        thresholds: Sequence[float] = config["damage_probability_thresholds"]
        values: Sequence[float] = config["damage_values"]
        return cls(tuple(float(v) for v in thresholds), tuple(float(v) for v in values))  # type: ignore[arg-type]


@dataclass(frozen=True)
class DamageResult:
    """One attack result distinguishing sampled, effective, and excess damage."""

    attempted: bool
    random_value: float | None
    nominal_damage: float
    effective_damage: float
    overkill_damage: float
    health_before: float
    health_after: float
    hit: bool
    destroyed: bool

    @property
    def damage(self) -> float:
        """Backward-compatible alias for effective damage."""

        return self.effective_damage


def damage_for_random_value(random_value: float, config: DamageConfig) -> float:
    """Map one sample in ``[0, 1)`` to nominal damage."""

    if not isfinite(random_value) or not 0.0 <= random_value < 1.0:
        raise ValueError("random_value must be finite and in [0, 1)")
    index = int(np.searchsorted(config.probability_thresholds, random_value, side="right"))
    return config.damage_values[index]


def apply_damage(
    target: UAVState,
    config: DamageConfig,
    random_value: float | None,
    attempted: bool = True,
) -> tuple[UAVState, DamageResult]:
    """Apply a supplied sample while preserving failure-state semantics."""

    before = target.health
    if not attempted:
        return target.copy(), DamageResult(False, None, 0.0, 0.0, 0.0, before, before, False, False)
    if random_value is None:
        raise ValueError("An attempted attack requires random_value")
    nominal = damage_for_random_value(random_value, config)
    after = max(0.0, before - nominal)
    effective = before - after
    overkill = nominal - effective
    hit = effective > 0.0
    destroyed = hit and after <= 0.0
    updated = replace(
        target,
        health=after,
        alive=target.alive and not destroyed,
        damaged=target.damaged or destroyed,
        ever_hit=target.ever_hit or hit,
    )
    return updated, DamageResult(True, random_value, nominal, effective, overkill, before, after, hit, destroyed)


def sample_damage(
    target: UAVState,
    config: DamageConfig,
    rng: np.random.Generator,
    attempted: bool = True,
) -> tuple[UAVState, DamageResult]:
    """Sample exclusively through the supplied generator."""

    return apply_damage(target, config, float(rng.random()) if attempted else None, attempted)
