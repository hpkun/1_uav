"""Exact piecewise linear motion and continuous segment events."""
import numpy as np


def direction(action, previous):
    raw = np.asarray(action, dtype=float) - 0.5
    length = np.linalg.norm(raw)
    return raw / length if length > 1e-12 else np.asarray(previous, dtype=float)


def proximity(p, v, q, w, radius, duration):
    relative = p - q
    velocity = v - w
    a = float(velocity @ velocity)
    b = float(2 * relative @ velocity)
    c = float(relative @ relative - radius * radius)
    if c <= 0:
        return 0.
    if a <= 0:
        return None
    discriminant = b*b - 4*a*c
    if discriminant < 0:
        return None
    time = (-b - np.sqrt(discriminant)) / (2*a)
    return float(time) if 0 <= time <= duration else None


def closest(p, v, q, w, duration):
    relative, velocity = p - q, v - w
    squared = float(velocity @ velocity)
    time = np.clip(-float(relative @ velocity) / squared, 0., duration) if squared else 0.
    return float(np.linalg.norm(relative + time * velocity))


def boundary(p, v, lower, upper, duration):
    candidates = []
    for axis in range(3):
        if v[axis] > 0:
            time = (upper[axis] - p[axis]) / v[axis]
        elif v[axis] < 0:
            time = (lower[axis] - p[axis]) / v[axis]
        else:
            continue
        if 0 <= time <= duration:
            candidates.append(float(time))
    return min(candidates) if candidates else None
