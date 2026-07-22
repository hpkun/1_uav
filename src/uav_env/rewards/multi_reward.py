"""Multi-agent reward placeholder."""

from collections.abc import Sequence

from uav_env.core.state import UAVState


def multi_reward(team: Sequence[UAVState], opponents: Sequence[UAVState]) -> float:
    """Compute a future team reward."""

    raise NotImplementedError("Multi-agent reward is not implemented")
