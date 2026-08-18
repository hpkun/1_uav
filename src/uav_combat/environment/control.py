"""Direct normalized-action to point-mass control mapping."""
from __future__ import annotations

import numpy as np

from ..models import ControlCommand


def action_to_control(action: np.ndarray, config: dict) -> ControlCommand:
    """Map ``[a0,a1,a2]`` directly to ``[nx,nz,phi]`` after clipping."""
    a0, a1, a2 = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
    return ControlCommand(
        nx=float(config["nx_scale"] * a0),
        nz=float(config["nz_trim"] + config["nz_scale"] * a1),
        phi=float(config["phi_max"] * a2),
    )


__all__ = ["action_to_control"]
