"""Collision interface placeholder."""

from uav_env.core.state import UAVState


def has_collision(first: UAVState, second: UAVState, minimum_separation: float) -> bool:
    """Determine whether two UAVs collide under a separation rule."""

    raise NotImplementedError("Collision handling is not implemented")
