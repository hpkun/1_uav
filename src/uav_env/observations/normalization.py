"""Explicit linear observation normalization."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class NormalizationConfig:
    """References and linear coefficients used by 1v1 observations."""

    horizontal_reference: float = 5000.0
    altitude_reference: float = 5000.0
    angle_reference: float = np.pi
    heading_reference: float = 2.0 * np.pi
    speed_difference_reference: float = 150.0
    health_reference: float = 300.0
    a: float = 2.0
    b: float = 1.0
    clip_observation: bool = True

    def __post_init__(self) -> None:
        references = (
            self.horizontal_reference,
            self.altitude_reference,
            self.angle_reference,
            self.heading_reference,
            self.speed_difference_reference,
            self.health_reference,
        )
        if not all(isfinite(value) and value > 0.0 for value in references):
            raise ValueError("Normalization references must be finite and positive")
        if not isfinite(self.a) or not isfinite(self.b):
            raise ValueError("Normalization coefficients must be finite")

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "NormalizationConfig":
        """Build normalization settings from an experiment config."""

        return cls(
            horizontal_reference=float(config["desired_distance_max"]),
            altitude_reference=float(config["max_altitude"]),
            speed_difference_reference=float(config["max_speed"] - config["min_speed"]),
            health_reference=float(config["initial_health"]),
            a=float(config.get("normalization_a", 2.0)),
            b=float(config.get("normalization_b", 1.0)),
            clip_observation=bool(config.get("clip_observation", True)),
        )


def normalize_features(
    values: Sequence[float] | NDArray[np.float64],
    references: Sequence[float] | NDArray[np.float64],
    config: NormalizationConfig,
) -> tuple[NDArray[np.float64], float]:
    """Apply ``a * value / reference - b`` and optionally clip the result."""

    vector = np.asarray(values, dtype=np.float64)
    scales = np.asarray(references, dtype=np.float64)
    if vector.shape != scales.shape or not np.all(np.isfinite(vector)) or not np.all(np.isfinite(scales)):
        raise ValueError("Normalization inputs must be finite arrays with matching shapes")
    if np.any(scales <= 0.0):
        raise ValueError("Normalization references must be positive")
    unbounded = config.a * vector / scales - config.b
    preclip_max_abs = float(np.max(np.abs(unbounded))) if unbounded.size else 0.0
    normalized = np.clip(unbounded, -1.0, 1.0) if config.clip_observation else unbounded
    return normalized.astype(np.float64), preclip_max_abs


def normalize_observation(observation: NDArray[np.float64]) -> NDArray[np.float64]:
    """Backward-compatible identity-reference linear normalization."""

    result, _ = normalize_features(observation, np.ones_like(observation), NormalizationConfig())
    return result
