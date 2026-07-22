"""Collision interface placeholder."""

from uav_env.core.state import UAVState
from uav_env.core.geometry import euclidean_distance


def has_collision(first: UAVState, second: UAVState, minimum_separation: float) -> bool:
    """Determine whether two UAVs collide under a separation rule."""

    if minimum_separation < 0.0:
        raise ValueError("minimum_separation must be non-negative")
    return minimum_separation > 0.0 and euclidean_distance(first.position_vector(), second.position_vector()) <= minimum_separation
