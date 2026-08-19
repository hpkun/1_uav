"""Canonical normalized-action to point-mass control mapping."""
from __future__ import annotations

import numpy as np

from ..models import AircraftState, ControlCommand


def trim_normal_load(
    theta: float | np.ndarray, phi: float | np.ndarray, eps: float = 1e-8
) -> float | np.ndarray:
    """Return the load that makes flight-path angle rate zero at ``theta, phi``."""
    cosine_phi = np.cos(phi)
    safe_cosine_phi = np.where(
        np.abs(cosine_phi) < eps,
        np.where(cosine_phi < 0.0, -eps, eps),
        cosine_phi,
    )
    result = np.cos(theta) / safe_cosine_phi
    return float(result) if np.ndim(result) == 0 else result


def action_to_control(
    state: AircraftState, action: np.ndarray, config: dict
) -> ControlCommand:
    """Map normalized ``[acceleration, trim-relative vertical, bank]`` to controls."""
    a0, a1, a2 = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
    phi = float(config["phi_max"] * a2)
    return ControlCommand(
        nx=float(config["nx_scale"] * a0),
        nz=float(trim_normal_load(state.theta, phi) + config["nz_delta_scale"] * a1),
        phi=phi,
    )


__all__ = ["action_to_control", "trim_normal_load"]
