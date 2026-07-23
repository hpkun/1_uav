"""Feature-semantic paper and training normalization modes."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi
from typing import Any, Literal, Sequence

import numpy as np
from numpy.typing import NDArray


FeatureKind = Literal["signed", "nonnegative", "yaw", "action", "failure"]


@dataclass(frozen=True)
class FeatureSpec:
    """Name, reference scale, and semantic normalization type."""

    name: str
    reference: float
    kind: FeatureKind


@dataclass(frozen=True)
class NormalizationResult:
    """Normalized values plus saturation diagnostics."""

    values: NDArray[np.float64]
    saturation_count: int
    saturation_ratio: float
    saturated_mask: NDArray[np.bool_]


@dataclass(frozen=True)
class NormalizationConfig:
    """Two explicit modes: paper reproduction and symmetric training."""

    mode: str = "symmetric_training"
    horizontal_reference: float = 5000.0
    altitude_reference: float = 5000.0
    angle_reference: float = pi
    heading_reference: float = 2.0 * pi
    speed_difference_reference: float = 150.0
    health_reference: float = 300.0
    a: float = 2.0
    b: float = 1.0
    clip_observation: bool | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"paper_linear", "symmetric_training"}:
            raise ValueError("normalization mode must be paper_linear or symmetric_training")
        if self.clip_observation is None:
            object.__setattr__(self, "clip_observation", self.mode == "symmetric_training")
        references = (
            self.horizontal_reference, self.altitude_reference, self.angle_reference,
            self.heading_reference, self.speed_difference_reference, self.health_reference,
        )
        if not all(isfinite(v) and v > 0.0 for v in references):
            raise ValueError("Normalization references must be finite and positive")

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "NormalizationConfig":
        """Build the selected normalization mode from experiment config."""

        mode = str(config.get("normalization_mode", "symmetric_training"))
        return cls(
            mode=mode,
            horizontal_reference=float(config["desired_distance_max"]),
            altitude_reference=float(config["max_altitude"]),
            speed_difference_reference=float(config["max_speed"] - config["min_speed"]),
            health_reference=float(config["initial_health"]),
            a=float(config.get("normalization_a", 2.0)),
            b=float(config.get("normalization_b", 1.0)),
            clip_observation=False if mode == "paper_linear" else True,
        )


def normalize_by_specs(
    values: Sequence[float] | NDArray[np.float64],
    specs: Sequence[FeatureSpec],
    config: NormalizationConfig,
) -> NormalizationResult:
    """Normalize every feature according to its explicit semantic type."""

    raw = np.asarray(values, dtype=np.float64)
    if raw.shape != (len(specs),) or not np.all(np.isfinite(raw)):
        raise ValueError("Values must be a finite vector matching feature specs")
    unbounded = np.empty_like(raw)
    for index, (value, spec) in enumerate(zip(raw, specs)):
        if spec.reference <= 0.0 or not isfinite(spec.reference):
            raise ValueError(f"Invalid reference for {spec.name}")
        transformed = float(value)
        if config.mode == "paper_linear":
            if spec.kind == "yaw":
                transformed %= 2.0 * pi
            if spec.kind in {"action", "failure"}:
                unbounded[index] = transformed
            else:
                unbounded[index] = config.a * transformed / spec.reference - config.b
        else:
            if spec.kind in {"signed", "yaw"}:
                unbounded[index] = transformed / spec.reference
            elif spec.kind == "action":
                unbounded[index] = 2.0 * transformed / spec.reference - 1.0
            elif spec.kind == "failure":
                unbounded[index] = transformed
            else:
                unbounded[index] = 2.0 * transformed / spec.reference - 1.0
    saturated_mask = np.abs(unbounded) > 1.0
    saturation_count = int(np.count_nonzero(saturated_mask))
    output = np.clip(unbounded, -1.0, 1.0) if bool(config.clip_observation) else unbounded
    return NormalizationResult(output.astype(np.float64), saturation_count, saturation_count / len(specs) if specs else 0.0, saturated_mask)


def normalize_features(
    values: Sequence[float] | NDArray[np.float64],
    references: Sequence[float] | NDArray[np.float64],
    config: NormalizationConfig,
) -> tuple[NDArray[np.float64], float]:
    """Compatibility helper treating all supplied features as nonnegative."""

    specs = [FeatureSpec(f"feature_{i}", float(r), "nonnegative") for i, r in enumerate(references)]
    result = normalize_by_specs(values, specs, config)
    return result.values, float(result.saturation_count)


def normalize_observation(observation: NDArray[np.float64]) -> NDArray[np.float64]:
    """Backward-compatible symmetric normalization with unit references."""

    specs = [FeatureSpec(f"feature_{i}", 1.0, "signed") for i in range(observation.size)]
    return normalize_by_specs(observation, specs, NormalizationConfig()).values
