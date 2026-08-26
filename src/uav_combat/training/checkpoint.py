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


__all__ = ["validate_checkpoint_environment"]
