"""Shared reward-diagnostic helpers for runner bookkeeping.

These helpers deliberately cover diagnostic counters and accumulators only.
They do not relax model, optimizer, network-shape or schema metadata checks.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


TERMINAL_TIMEOUT_SCHEMAS = {
    "homogeneous_3v3_v2_timeaware",
    "functional_heterogeneous_3v3_v1",
}


def allows_truncation_bootstrap(schema: str, truncated: bool, termination_reason: str) -> bool:
    """Return whether a truncated step should bootstrap from terminal value.

    Terminal-timeout schemas treat timeout as a complete episode terminal state
    with explicit terminal reward allocation, so critic bootstrap is disabled.
    Legacy schemas keep their existing bootstrap behavior.
    """

    if not truncated:
        return False
    if schema in TERMINAL_TIMEOUT_SCHEMAS:
        if termination_reason != "timeout":
            raise RuntimeError(f"{schema} truncated step must be timeout, got {termination_reason!r}")
        return False
    return True


def restore_reward_component_accumulators(
    state: Any,
    component_names: tuple[str, ...],
    expected_shape: tuple[int, ...],
    *,
    error_prefix: str,
) -> dict[str, np.ndarray]:
    """Restore reward-component accumulators with backward-compatible diagnostics.

    Missing current diagnostic fields are initialized to zero so older
    checkpoints remain resumable after logging-only fields are added.  Existing
    fields still require valid numeric arrays with the exact expected shape.
    Extra deprecated diagnostic fields from checkpoints are ignored.
    """

    if state is None:
        return {name: np.zeros(expected_shape, dtype=np.float64) for name in component_names}
    if not isinstance(state, Mapping):
        raise ValueError(f"{error_prefix} must be a mapping, got {type(state).__name__}")

    restored: dict[str, np.ndarray] = {}
    for name in component_names:
        if name not in state:
            restored[name] = np.zeros(expected_shape, dtype=np.float64)
            continue
        try:
            values = np.asarray(state[name], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{error_prefix}[{name}] must be numeric") from exc
        if values.shape != expected_shape:
            raise ValueError(f"{error_prefix}[{name}] shape mismatch: checkpoint={values.shape}, expected={expected_shape}")
        restored[name] = values.copy()
    return restored
