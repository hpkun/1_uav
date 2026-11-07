"""Continuous threshold windows on analytic trajectories.

For each interval the vector curve lies within M*h*h/8 of its chord,
where M bounds its second derivative. Only ambiguous intervals subdivide.
"""
import numpy as np


def _chord(first, last):
    delta = last - first
    square = float(delta @ delta)
    fraction = float(np.clip(-(first @ delta) / square, 0, 1)) if square else 0.
    return delta, square, float(np.linalg.norm(first + fraction * delta))


def windows(curve, bound, radius, start, stop, tolerance):
    intervals = []

    def append(left, right):
        if intervals and left <= intervals[-1][1] + 1e-13:
            intervals[-1] = (intervals[-1][0], max(right, intervals[-1][1]))
        else:
            intervals.append((left, right))

    def visit(left, right, first, last):
        width = right - left
        delta, square, minimum = _chord(first, last)
        error = bound * width**2 / 8
        maximum = max(np.linalg.norm(first), np.linalg.norm(last))
        roundoff = 32 * np.finfo(float).eps * max(1., maximum, radius)
        if minimum > radius + error + roundoff:
            return
        if maximum + error <= radius:
            append(left, right)
            return
        if width <= tolerance or error <= 1e-12:
            # Resolve the chord against the sphere. The small envelope gives
            # conservative detection of tangencies within spatial resolution.
            limit = radius + error
            if square == 0:
                append(left, right)
                return
            center = -float(first @ delta) / square
            offset = first + center * delta
            extent = np.sqrt(max(0., (limit**2 - float(offset @ offset)) / square))
            append(left + width * float(np.clip(center - extent, 0., 1.)),
                   left + width * float(np.clip(center + extent, 0., 1.)))
            return
        mid = (left + right) / 2
        middle = curve(mid)
        visit(left, mid, first, middle)
        visit(mid, right, middle, last)

    visit(start, stop, curve(start), curve(stop))
    return intervals


def intersection(left, right):
    i = j = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if start <= end:
            yield start, end
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1


def minimum_distance(curve, bound, end, tolerance=1e-3):
    """Continuous minimum to a certified distance tolerance, in metres."""
    best = float("inf")

    def visit(left, right, first, last):
        nonlocal best
        delta, square, lower = _chord(first, last)
        error = bound * (right - left)**2 / 8
        if lower - error >= best - tolerance:
            return
        fraction = float(np.clip(-(first @ delta) / square, 0, 1)) if square else 0.
        best = min(best, float(np.linalg.norm(curve(left + fraction * (right - left)))))
        if 2 * error > tolerance:
            mid = (left + right) / 2
            middle = curve(mid)
            visit(left, mid, first, middle)
            visit(mid, right, middle, last)

    visit(0., end, curve(0.), curve(end))
    return best
