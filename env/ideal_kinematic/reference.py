"""Constant arc-length speed along a piecewise linear Bézier approximation."""
import numpy as np


class Reference:
    def __init__(self, start, end, speed, mode):
        start, end = np.asarray(start, float), np.asarray(end, float)
        if mode == "straight":
            points = np.stack((start, end))
        else:
            span = end - start
            offset = np.array((-span[1], span[0], 0.))
            if np.linalg.norm(offset) < 1e-12:
                offset = np.array((0., 1., 0.))
            offset = 0.25 * np.linalg.norm(span) * offset / np.linalg.norm(offset)
            control1, control2 = start + span/3 + offset, start + 2*span/3 - offset
            t = np.linspace(0., 1., 1025)[:, None]
            points = (1-t)**3*start + 3*(1-t)**2*t*control1 + 3*(1-t)*t*t*control2 + t**3*end
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        if np.any(lengths <= 0):
            raise ValueError("reference curve has zero-length segments")
        self.points = points
        self.lengths = lengths
        self.distance = np.r_[0., np.cumsum(lengths)]
        self.speed = speed
        self.duration = self.distance[-1] / speed

    def sample(self, time):
        distance = min(max(time, 0.) * self.speed, self.distance[-1])
        index = min(np.searchsorted(self.distance, distance, side="right") - 1, len(self.lengths)-1)
        fraction = (distance - self.distance[index]) / self.lengths[index]
        position = self.points[index] + fraction * (self.points[index+1] - self.points[index])
        velocity = (self.points[index+1] - self.points[index]) / self.lengths[index] * self.speed
        return position, velocity if time < self.duration else np.zeros(3)
