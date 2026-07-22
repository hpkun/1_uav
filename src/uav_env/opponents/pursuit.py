"""Pursuit opponent structure."""

from uav_env.actions.discrete_15 import DiscreteAction15
from uav_env.core.state import UAVState
from uav_env.opponents.base import RuleOpponent


class PursuitOpponent(RuleOpponent):
    """Future line-of-sight pursuit opponent."""

    def select_action(self, ownship: UAVState, opponent: UAVState) -> DiscreteAction15:
        """Select a pursuit action once pursuit rules are specified."""

        raise NotImplementedError("Pursuit opponent policy is not implemented")
