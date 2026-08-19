"""Pure functions for action/control stability diagnosis."""
from __future__ import annotations

import numpy as np
import torch

from ..madsac.actor import SharedSquashedGaussianActor
from ..models import AircraftState, ControlCommand
from ..environment.control import trim_normal_load


def trim_a1(theta: float | np.ndarray, phi: float | np.ndarray) -> np.ndarray:
    """Historical V1.1 raw-action coordinate of the bank-trim load."""
    return (trim_normal_load(theta, phi) - 1.0) / 4.0


def vertical_balance(theta: float | np.ndarray, a1: float | np.ndarray, a2: float | np.ndarray):
    """Vertical balance under the active V1.2+ trim-relative semantics."""
    phi = (np.pi / 3.0) * np.asarray(a2)
    nz = trim_normal_load(theta, phi) + 2.0 * np.asarray(a1)
    return nz * np.cos(phi) - np.cos(theta)


def bank_compensated_actions(
    states: list[AircraftState], base_actions: np.ndarray
) -> np.ndarray:
    """Diagnostic-only historical V1.1 bank-compensation reproducer."""
    result = np.asarray(base_actions, dtype=np.float32).copy()
    for index, state in enumerate(states):
        if not state.alive:
            result[index] = 0.0
            continue
        phi = (np.pi / 3.0) * float(np.clip(result[index, 2], -1.0, 1.0))
        elevation_correction = 4.0 * float(result[index, 1])
        nz_desired = float(trim_normal_load(state.theta, phi)) + elevation_correction
        result[index, 1] = np.clip((nz_desired - 1.0) / 4.0, -1.0, 1.0)
    return result


def trim_relative_control(state: AircraftState, action: np.ndarray, k: float = 2.0) -> ControlCommand:
    """Diagnostic-only explicit control equivalent of the adopted V1.2+ mapping."""
    a0, a1, a2 = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
    phi = (np.pi / 3.0) * a2
    nz = float(trim_normal_load(state.theta, phi) + k * a1)
    return ControlCommand(nx=2.0 * a0, nz=nz, phi=phi)


def fresh_actor(seed: int, hidden_dim: int = 256) -> SharedSquashedGaussianActor:
    """Construct a never-trained actor; this function does not load state or create optimizers."""
    torch.manual_seed(int(seed))
    actor = SharedSquashedGaussianActor(
        observation_dim=54, action_dim=3, hidden_dim=hidden_dim,
        log_std_min=-5.0, log_std_max=2.0, activation="relu",
    )
    actor.eval()
    return actor


__all__ = [
    "bank_compensated_actions", "fresh_actor", "trim_a1", "trim_normal_load",
    "trim_relative_control", "vertical_balance",
]
