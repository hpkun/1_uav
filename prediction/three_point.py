"""Three-position quadratic extrapolation in three spatial dimensions."""
import math
import numpy as np


def predict_two_steps(history_positions, dt):
    """Return positions at t+dt and t+2dt from oldest-to-newest observations."""
    positions = np.asarray(history_positions, dtype=np.float64)
    if positions.shape != (3, 3) or not np.isfinite(positions).all():
        raise ValueError("history_positions must be a finite (3, 3) array")
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    old, middle, current = positions
    a = (current - 2 * middle + old) / (2 * dt * dt)
    b = (3 * current - 4 * middle + old) / (2 * dt)
    first = a * dt * dt + b * dt + current
    second = a * (2 * dt)**2 + b * (2 * dt) + current
    result = np.stack((first, second))
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite quadratic prediction")
    return result
