"""Single-agent reward placeholder."""

from uav_env.core.state import UAVState


def single_reward(previous: UAVState, current: UAVState, opponent: UAVState) -> float:
    """Compute a future 1v1 reward."""

    raise NotImplementedError("Single-agent reward is not implemented")
