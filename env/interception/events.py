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


def windows(curve, bound, radius, start, stop, tolerance, *, outside=False):
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
        if outside:
            if minimum - error >= radius - roundoff:
                append(left, right)
                return
            if maximum + error < radius - roundoff:
                return
        else:
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
                if not outside or minimum >= limit - roundoff:
                    append(left, right)
                return
            center = -float(first @ delta) / square
            offset = first + center * delta
            extent = np.sqrt(max(0., (limit**2 - float(offset @ offset)) / square))
            enter = left + width * float(np.clip(center - extent, 0., 1.))
            leave = left + width * float(np.clip(center + extent, 0., 1.))
            if outside:
                if enter > left:
                    append(left, enter)
                if leave < right:
                    append(leave, right)
            else:
                append(enter, leave)
            return
        mid = (left + right) / 2
        middle = curve(mid)
        visit(left, mid, first, middle)
        visit(mid, right, middle, last)

    visit(start, stop, curve(start), curve(stop))
    return intervals


def boundary_contact(state, target, tau, at, lower, upper, duration, tolerance):
    """Split each analytic component at its unique possible turning point."""
    first = None
    for axis in range(3):
        initial, command = state[3+axis], target[axis]
        cuts = [0., duration]
        turning = None
        if initial*command < 0.:
            turning = tau*np.log1p(abs(initial/command))
            if 0 < turning < duration:
                cuts.insert(1, turning)
        for start, stop in zip(cuts, cuts[1:]):
            left, right = at(start)[axis], at(stop)[axis]
            speed = 0. if stop == turning else at(stop)[3+axis]
            upper_exit = right > upper[axis] or (right == upper[axis] and speed > 0)
            lower_exit = right < lower[axis] or (right == lower[axis] and speed < 0)
            if not (upper_exit or lower_exit):
                continue
            limit = upper[axis] if upper_exit else lower[axis]
            if left == limit:
                contact = start
            else:
                low, high = start, stop
                while high-low > tolerance/4:
                    mid = (low+high)/2
                    if (at(mid)[axis]-limit)*(1 if upper_exit else -1) >= 0:
                        high = mid
                    else:
                        low = mid
                contact = high
            first = contact if first is None else min(first, contact)
            break
    return first


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


def minimum_distance(curve, bound, end, tolerance=1e-3, *, with_time=False):
    """Continuous minimum to a certified distance tolerance, in metres."""
    best, best_time = float("inf"), 0.

    def visit(left, right, first, last):
        nonlocal best, best_time
        delta, square, lower = _chord(first, last)
        error = bound * (right - left)**2 / 8
        if lower - error >= best - tolerance:
            return
        fraction = float(np.clip(-(first @ delta) / square, 0, 1)) if square else 0.
        t = left + fraction * (right - left)
        distance = float(np.linalg.norm(curve(t)))
        if distance < best:
            best, best_time = distance, t
        if 2 * error > tolerance:
            mid = (left + right) / 2
            middle = curve(mid)
            visit(left, mid, first, middle)
            visit(mid, right, middle, last)

    visit(0., end, curve(0.), curve(end))
    return (best, best_time) if with_time else best
