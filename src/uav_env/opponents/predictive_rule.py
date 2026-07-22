"""Predictive rule-opponent structure."""

from uav_env.actions.discrete_15 import DiscreteAction15
from uav_env.core.state import UAVState
from uav_env.opponents.base import RuleOpponent


class PredictiveRuleOpponent(RuleOpponent):
    """Future opponent using predicted target motion."""

    def select_action(self, ownship: UAVState, opponent: UAVState) -> DiscreteAction15:
        """Select a predictive action once the prediction rule is specified."""

        raise NotImplementedError("Predictive opponent policy is not implemented")
