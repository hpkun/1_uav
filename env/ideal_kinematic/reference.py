"""Piecewise linear arc-length reference; velocity is right-continuous at knots."""
import math
import numpy as np


class Reference:
    def __init__(self, start, end, speed, mode, end_behavior="stop"):
        start, end = np.asarray(start, float), np.asarray(end, float)
        if (start.shape != (3,) or end.shape != (3,) or not np.isfinite(start).all()
                or not np.isfinite(end).all() or not math.isfinite(speed) or speed <= 0
                or mode not in ("straight", "bezier") or end_behavior not in ("stop", "ping_pong")):
            raise ValueError("invalid reference geometry, speed or mode")
        if mode == "straight":
            points = np.stack((start, end))
        else:
            span = end - start
            offset = np.array((-span[1], span[0], 0.))
            if np.linalg.norm(offset) < 1e-12:
                offset = np.array((0., 1., 0.))
            offset = .25 * np.linalg.norm(span) * offset / np.linalg.norm(offset)
            a, b = start + span/3 + offset, start + 2*span/3 - offset
            t = np.linspace(0., 1., 1025)[:, None]
            points = (1-t)**3*start + 3*(1-t)**2*t*a + 3*(1-t)*t*t*b + t**3*end
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        if not np.isfinite(points).all() or np.any(lengths <= 0):
            raise ValueError("reference path must have finite, nonzero segments")
        self.points, self.lengths = points, lengths
        self.distance = np.r_[0., np.cumsum(lengths)]
        self.speed, self.end_behavior = speed, end_behavior
        self.duration = self.distance[-1] / speed

    def sample(self, time):
        if not math.isfinite(time) or time < 0:
            raise ValueError("invalid reference time")
        if self.end_behavior == "stop" and time >= self.duration:
            return self.points[-1].copy(), np.zeros(3)
        leg = math.floor(time / self.duration)
        local = time - leg*self.duration
        if self.end_behavior == "stop":
            leg = 0
        reverse = self.end_behavior == "ping_pong" and leg % 2 == 1
        distance = self.distance[-1] - local*self.speed if reverse else local*self.speed
        index = min(max(np.searchsorted(self.distance, distance, side="left" if reverse else "right")-1, 0),
                    len(self.lengths)-1)
        fraction = (distance-self.distance[index])/self.lengths[index]
        position = self.points[index]+fraction*(self.points[index+1]-self.points[index])
        velocity = (self.points[index+1]-self.points[index])/self.lengths[index]*self.speed
        return position, -velocity if reverse else velocity

    def segments(self, start, end):
        """Yield exact linear pieces over [start,end]; no segment crosses a knot."""
        if end < start:
            raise ValueError("reversed reference interval")
        knots = {float(start), float(end)}
        if self.end_behavior == "stop" and start < self.duration < end:
            knots.add(self.duration)
        count = int(end/self.duration)+1 if self.end_behavior == "ping_pong" else 1
        for leg in range(count+1):
            base = leg*self.duration
            if base > end:
                break
            if self.end_behavior == "stop" and leg:
                break
            for distance in self.distance:
                knot = base + (self.distance[-1]-distance if leg % 2 else distance)/self.speed
                if start < knot < end:
                    knots.add(float(knot))
        cuts = sorted(knots)
        for left, right in zip(cuts[:-1], cuts[1:]):
            if right-left <= 1e-12:
                continue
            position, velocity = self.sample(left)
            yield left, right, position, velocity
