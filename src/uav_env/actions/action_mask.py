"""Action-mask interface placeholder."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from uav_env.core.state import UAVState


def build_action_mask(state: UAVState) -> NDArray[np.bool_]:
    """Build a state-dependent action mask.

    TODO: Define constraints after environment boundary behaviour is specified.
    """

    raise NotImplementedError("State-dependent action masking is not implemented")
