"""Exact zero-order hold motion in SI units."""
import numpy as np


def project(vector, limit):
    vector = np.asarray(vector, dtype=float)
    return limit * vector / np.maximum(1., np.linalg.norm(vector, axis=-1, keepdims=True))


def command(actions, limit):
    return project(2. * np.asarray(actions, dtype=float) - 1., limit)


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
