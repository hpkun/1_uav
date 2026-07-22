"""Straight-flight opponent structure."""

from uav_env.actions.discrete_15 import DiscreteAction15
from uav_env.core.state import UAVState
from uav_env.opponents.base import RuleOpponent


class StraightOpponent(RuleOpponent):
    """Future opponent that maintains straight, level flight."""

    def select_action(self, ownship: UAVState, opponent: UAVState) -> DiscreteAction15:
        """Return a straight-flight command once policy details are approved."""

        raise NotImplementedError("Straight opponent policy is not implemented")
