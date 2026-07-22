"""Single-agent observation placeholder."""

import numpy as np
from numpy.typing import NDArray

from uav_env.entities.uav import UAV


def build_single_observation(ownship: UAV, opponent: UAV) -> NDArray[np.float64]:
    """Build a future normalized 1v1 observation."""

    raise NotImplementedError("Single-agent observations are not implemented")
