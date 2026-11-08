"""Exact zero-order hold motion in SI units."""
import numpy as np


def project(vector, limit, vertical_limit=None):
    """Project a velocity into the total-speed ball and optionally limit climb/descent."""
    vector = np.asarray(vector, dtype=float)
    projected = limit * vector / np.maximum(1., np.linalg.norm(vector, axis=-1, keepdims=True))
    if vertical_limit is not None:
        projected = np.array(projected, copy=True)
        projected[..., 2] = np.clip(projected[..., 2], -vertical_limit, vertical_limit)
    return projected


def command(actions, limit, vertical_limit=None):
    return project(2. * np.asarray(actions, dtype=float) - 1., limit, vertical_limit)


def advance(state, target, tau, duration):
    state = np.asarray(state, dtype=float)
    target = np.asarray(target, dtype=float)
    fraction = -np.expm1(-duration / tau)
    # Blend the initial and commanded velocities. Series avoids cancellation
    # in the displacement from acceleration when duration/tau is tiny.
    ratio = duration / tau
    lag = tau * fraction
    drive = (duration * ratio * (.5 - ratio / 6 + ratio**2 / 24)
             if ratio < 1e-4 else duration - lag)
    position = state[..., :3] + lag * state[..., 3:] + drive * target
    velocity = np.exp(-ratio) * state[..., 3:] + fraction * target
    return np.concatenate((position, velocity), axis=-1)
