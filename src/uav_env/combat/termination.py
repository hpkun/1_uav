"""Episode termination interface placeholder."""

from collections.abc import Sequence

from uav_env.core.state import UAVState


def is_terminal(states: Sequence[UAVState], elapsed_seconds: float, limit_seconds: float) -> bool:
    """Determine future combat-episode termination."""

    raise NotImplementedError("Episode termination rules are not implemented")
