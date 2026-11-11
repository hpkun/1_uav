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


def inside_intervals(p, v, q, w, radius, duration):
    """Closed intervals whose relative linear motion lies within a sphere."""
    r, u = np.asarray(p)-np.asarray(q), np.asarray(v)-np.asarray(w)
    a, b, c = float(u@u), float(2*r@u), float(r@r-radius*radius)
    if a <= 1e-24:
        return [(0., duration)] if c <= 0 else []
    discriminant = b*b-4*a*c
    if discriminant < 0:
        return []
    root = np.sqrt(max(0., discriminant))
    left, right = (-b-root)/(2*a), (-b+root)/(2*a)
    lo, hi = max(0., left), min(duration, right)
    return [(float(lo), float(hi))] if hi > lo else []


def merge_intervals(intervals, tolerance=1e-10):
    merged = []
    for left, right in sorted(intervals):
        if merged and left <= merged[-1][1]+tolerance:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
        else:
            merged.append((left, right))
    return merged


def subtract_intervals(outer, inner):
    result = []
    for left, right in outer:
        cursor = left
        for a, b in inner:
            if b <= cursor or a >= right:
                continue
            if a > cursor:
                result.append((cursor, min(a, right)))
            cursor = max(cursor, b)
        if cursor < right:
            result.append((cursor, right))
    return result


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
