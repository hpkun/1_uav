"""Deterministic nearest-target association for active point-mass members."""
import numpy as np


def nearest_distance_assignment(defender_positions, intruder_positions,
                                defender_mask, intruder_mask):
    """Map each active defender to its nearest active intruder in 3D.

    Multiple defenders may select one intruder. Exact ties choose the lower
    intruder index, without randomness or order-dependent containers.
    """
    defenders = np.asarray(defender_positions, dtype=np.float64)
    intruders = np.asarray(intruder_positions, dtype=np.float64)
    own = np.asarray(defender_mask)
    target = np.asarray(intruder_mask)
    if (defenders.ndim != 2 or defenders.shape[1] != 3 or
            intruders.ndim != 2 or intruders.shape[1] != 3 or
            own.shape != (len(defenders),) or target.shape != (len(intruders),) or
            own.dtype.kind != "b" or target.dtype.kind != "b" or
            not np.isfinite(defenders).all() or not np.isfinite(intruders).all()):
        raise ValueError("invalid assignment positions or active masks")
    active_targets = np.flatnonzero(target)
    if not len(active_targets):
        return {}
    result = {}
    for i in np.flatnonzero(own):
        distance = np.linalg.norm(intruders[active_targets]-defenders[i], axis=1)
        result[int(i)] = int(active_targets[np.argmin(distance)])
    return result
