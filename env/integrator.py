"""Historical Coop3D RK4 and projections, retained for old trajectory compatibility.

The gamma projection is not a constraint published in Xu's paper.
"""
import numpy as np
from .dynamics import derivatives, wrap_angle


def rk4(state, control, config, counters):
    y = np.asarray(state, dtype=np.float64)
    if not np.isfinite(y).all():
        counters['nonfinite_states'] += 1
        raise FloatingPointError('input flight state is nonfinite')
    h = config.dt
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        k1 = derivatives(y, control, config)
        k2 = derivatives(y + h * k1 / 2, control, config)
        k3 = derivatives(y + h * k2 / 2, control, config)
        k4 = derivatives(y + h * k3, control, config)
        result = y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    if not np.isfinite(result).all():
        counters['nonfinite_states'] += 1
        return y.copy(), False  # Retain last finite state; environment deactivates UAV.
    counters['speed_clips'] += int(not config.v_min <= result[3] <= config.v_max)
    counters['gamma_clips'] += int(abs(result[4]) > config.gamma_limit)
    result[3] = np.clip(result[3], config.v_min, config.v_max)
    result[4] = np.clip(result[4], -config.gamma_limit, config.gamma_limit)
    result[5] = wrap_angle(result[5])
    return result, True
