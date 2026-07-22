"""Reward component interfaces."""

from uav_env.core.state import UAVState


def geometry_reward(ownship: UAVState, opponent: UAVState) -> float:
    """Compute a future geometry-based reward component."""

    raise NotImplementedError("Reward components are not implemented")
