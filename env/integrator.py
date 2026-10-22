"""Shared unprojected RK4, then engineering speed projection/heading wrapping.

Neither RK4 nor endpoint speed projection is specified by Xu. No angle clipping.
"""
import numpy as np
from .dynamics import wrap_angle
from .paper_dynamics import DynamicsDomainError, rk4_step


def rk4(state, control, config, counters, *, diagnostics=None):
    y = np.asarray(state, dtype=np.float64)
    if not np.isfinite(y).all():
        counters['nonfinite_states'] += 1
        raise FloatingPointError('input flight state is nonfinite')
    try:
        step = rk4_step(y, control, gravity=config.gravity, dt=config.dt,
                        epsilon=config.denominator_epsilon)
        reason = step.reason
    except DynamicsDomainError as exc:
        step, reason = None, str(exc)
    if step is None or not step.accepted:
        counters['integration_failures'] = counters.get('integration_failures', 0) + 1
        if not np.isfinite(control).all():
            counters['nonfinite_states'] += 1
        if diagnostics is not None:
            diagnostics['reason'] = reason
        return y.copy(), False
    result = step.state.copy()
    counters['speed_clips'] += int(not config.v_min <= result[3] <= config.v_max)
    result[3] = np.clip(result[3], config.v_min, config.v_max)
    result[5] = wrap_angle(result[5])
    return result, True
