"""Observation-mask placeholder."""

import numpy as np
from numpy.typing import NDArray


def observation_mask(entity_count: int, maximum_entities: int) -> NDArray[np.bool_]:
    """Build a future padding mask for multi-agent observations."""

    raise NotImplementedError("Observation masking is not implemented")
