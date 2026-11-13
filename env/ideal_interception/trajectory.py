"""Piecewise linear arc-length reference; velocity is right-continuous at knots."""
import math
import hashlib
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
            controls = []
        else:
            span = end - start
            offset = np.array((-span[1], span[0], 0.))
            if np.linalg.norm(offset) < 1e-12:
                offset = np.array((0., 1., 0.))
            offset = .25 * np.linalg.norm(span) * offset / np.linalg.norm(offset)
            a, b = start + span/3 + offset, start + 2*span/3 - offset
            controls = [a.tolist(),b.tolist()]
            t = np.linspace(0., 1., 1025)[:, None]
            points = (1-t)**3*start + 3*(1-t)**2*t*a + 3*(1-t)*t*t*b + t**3*end
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        if not np.isfinite(points).all() or np.any(lengths <= 0):
            raise ValueError("reference path must have finite, nonzero segments")
        self.points, self.lengths = points, lengths
        self.mode, self.controls = mode, controls
        self.distance = np.r_[0., np.cumsum(lengths)]
        self.speed, self.end_behavior = speed, end_behavior
        self.duration = self.distance[-1] / speed

    def sample(self, time):
        if not math.isfinite(time) or time < 0:
            raise ValueError("invalid reference time")
        if self.end_behavior == "stop" and time >= self.duration:
            velocity = (self.points[-1]-self.points[-2])/self.lengths[-1]*self.speed
            return self.points[-1].copy(), velocity if time == self.duration else np.zeros(3)
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

    def specification(self):
        return dict(mode=self.mode,start=self.points[0].tolist(),end=self.points[-1].tolist(),
                    controls=self.controls,speed=self.speed,duration=self.duration,
                    polyline_points=len(self.points),
                    polyline_sha256=hashlib.sha256(self.points.tobytes()).hexdigest())

    def segments(self, start, end):
        """Yield exact linear pieces over [start,end]; no segment crosses a knot."""
        if end < start:
            raise ValueError("reversed reference interval")
        knots = {float(start), float(end)}
        if self.end_behavior == "stop" and start < self.duration < end:
            knots.add(self.duration)
        first_leg = max(0, int(start/self.duration)-1) if self.end_behavior == "ping_pong" else 0
        last_leg = int(end/self.duration)+1 if self.end_behavior == "ping_pong" else 0
        for leg in range(first_leg, last_leg+1):
            base = leg*self.duration
            if base > end:
                break
            local_left = max(0.,start-base)
            local_right = min(self.duration,end-base)
            if local_right < local_left:
                continue
            if leg % 2:
                lower = self.distance[-1]-local_right*self.speed
                upper = self.distance[-1]-local_left*self.speed
            else:
                lower, upper = local_left*self.speed, local_right*self.speed
            begin = np.searchsorted(self.distance,lower,side="left")
            finish = np.searchsorted(self.distance,upper,side="right")
            for distance in self.distance[begin:finish]:
                knot = base + (self.distance[-1]-distance if leg % 2 else distance)/self.speed
                if start < knot < end:
                    knots.add(float(knot))
        cuts = sorted(knots)
        for left, right in zip(cuts[:-1], cuts[1:]):
            if right-left <= 1e-12:
                continue
            position, velocity = self.sample(left)
            yield left, right, position, velocity
