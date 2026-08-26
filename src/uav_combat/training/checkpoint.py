"""Shared environment-compatibility checks for training checkpoints."""
from __future__ import annotations

from typing import Any

from ..config import ENVIRONMENT_VERSION


def validate_checkpoint_environment(
    state: dict[str, Any], env_config: dict[str, Any]
) -> None:
    extra = state.get("extra", {})
    version = extra.get("environment_version")
    if version != ENVIRONMENT_VERSION:
        raise RuntimeError(
            "checkpoint environment_version mismatch: expected "
            f"{ENVIRONMENT_VERSION}, got {version!r}; environment semantics "
            "are incompatible"
        )
    expected_variant = env_config.get("environment_variant", "direct_v2_3")
    checkpoint_variant = extra.get("environment_variant", "direct_v2_3")
    if checkpoint_variant != expected_variant:
        raise RuntimeError(
            "checkpoint environment_variant mismatch: expected "
            f"{expected_variant!r}, got {checkpoint_variant!r}"
        )


def evaluation_selection_key(
    record: dict[str, Any], environment_variant: str
) -> tuple[float, ...]:
    """Return the variant-specific lexicographic best-checkpoint key."""
    if environment_variant == "persistent_wave_v1":
        waves_cleared = record.get(
            "average_waves_cleared", record.get("mean_waves_cleared", 0.0)
        )
        final_clear = record.get(
            "clear_wave_3_probability", record.get("win_rate", 0.0)
        )
        return (
            float(waves_cleared),
            float(final_clear),
            float(record["average_return"]),
            -float(record["average_red_loss"]),
        )
    return (
        float(record["win_rate"]),
        float(record["average_return"]),
        -float(record["average_red_loss"]),
    )


__all__ = ["evaluation_selection_key", "validate_checkpoint_environment"]
