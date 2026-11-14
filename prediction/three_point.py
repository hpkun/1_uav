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


def predict_at(history_positions, dt, query_times):
    """Quadratic extrapolation at positive offsets from the latest observation."""
    positions = np.asarray(history_positions, dtype=np.float64)
    if positions.shape != (3, 3) or not np.isfinite(positions).all():
        raise ValueError("history_positions must be a finite (3, 3) array")
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    times = np.asarray(query_times, dtype=np.float64)
    if times.ndim != 1 or not np.isfinite(times).all() or np.any(times <= 0) or np.any(np.diff(times) <= 0):
        raise ValueError("query_times must be strictly increasing positive offsets")
    old, middle, current = positions
    a = (current - 2 * middle + old) / (2 * dt * dt)
    b = (3 * current - 4 * middle + old) / (2 * dt)
    result = current[None, :] + times[:, None] * b[None, :] + times[:, None]**2 * a[None, :]
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite quadratic prediction")
    return result


def recommended_horizon(history_positions, dt, *, eta, minimum, maximum,
                        epsilon_acceleration):
    """Exploratory quadratic extrapolation cap, in seconds; does not alter prediction.

    epsilon_acceleration has units m/s². All tuning values are caller supplied.
    """
    import numpy as np
    values = np.asarray(history_positions, dtype=np.float64)
    parameters = (dt, eta, minimum, maximum, epsilon_acceleration)
    if (values.shape != (3, 3) or not np.isfinite(values).all() or
            not all(np.isfinite(x) for x in parameters) or dt <= 0 or eta <= 0 or
            minimum <= 0 or maximum < minimum or epsilon_acceleration <= 0):
        raise ValueError("invalid history or horizon cap parameters")
    old, middle, current = values
    velocity = (3*current-4*middle+old)/(2*dt)
    acceleration = (current-2*middle+old)/(dt*dt)
    raw = eta*np.linalg.norm(velocity)/(np.linalg.norm(acceleration)+epsilon_acceleration)
    return float(np.clip(raw, minimum, maximum))
