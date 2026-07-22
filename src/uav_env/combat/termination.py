"""Episode termination interface placeholder."""

from collections.abc import Sequence

from uav_env.core.state import UAVState


def is_terminal(states: Sequence[UAVState], elapsed_seconds: float, limit_seconds: float) -> bool:
    """Determine future combat-episode termination."""

    if elapsed_seconds < 0.0 or limit_seconds <= 0.0:
        raise ValueError("Episode times are invalid")
    return any(not state.alive for state in states) or elapsed_seconds >= limit_seconds
