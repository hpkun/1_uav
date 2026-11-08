"""Ground facility and continuous cylindrical protected airspace.

A ground cylinder is a mission region, not a physical building or a weapon range.
All trajectories are continuous; only the intrusion event definition changes.
"""
import numpy as np

from .events import intersection, windows


def navigation_point(config):
    return np.asarray(config.center) + config.navigation_offset


def contains_intruder(position, config):
    position = np.asarray(position, dtype=float)
    center = np.asarray(config.center, dtype=float)
    return bool(np.linalg.norm(position[:2] - center[:2]) <= config.radius and
                center[2] <= position[2] <= center[2] + config.protected_height)


def intrusion_windows(trajectory, acceleration_bound, config):
    """Continuous intervals of an intruder center entering the protected region.

    In the ground-cylinder profile the arena floor is at or above the facility's
    ground datum. Exiting the legal arena is a separate event.
    """
    start, stop = 0., config.dt
    tolerance = config.event_time_tolerance
    center = np.asarray(config.center, dtype=float)
    horizontal = windows(lambda t: trajectory(t)[:2] - center[:2],
                         acceleration_bound, config.radius, start, stop, tolerance)
    below_top = windows(lambda t: np.array([trajectory(t)[2] - center[2]]),
                        acceleration_bound, config.protected_height,
                        start, stop, tolerance)
    return list(intersection(horizontal, below_top))
