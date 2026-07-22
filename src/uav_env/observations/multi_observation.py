"""Multi-agent observation placeholder."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from uav_env.entities.uav import UAV


def build_multi_observation(observer: UAV, aircraft: Sequence[UAV]) -> NDArray[np.float64]:
    """Build a future padded multi-aircraft observation."""

    raise NotImplementedError("Multi-agent observations are not implemented")
