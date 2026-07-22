"""Predictive rule-opponent structure."""

from uav_env.actions.discrete_15 import DiscreteAction15
from uav_env.core.state import UAVState
from uav_env.opponents.base import RuleOpponent
import numpy as np


class PredictiveRuleOpponent(RuleOpponent):
    """Future opponent using predicted target motion."""

    def select_action(
        self,
        ownship: UAVState,
        opponent: UAVState,
        rng: np.random.Generator | None = None,
    ) -> DiscreteAction15:
        """Select a predictive action once the prediction rule is specified."""

        del ownship, opponent, rng
        raise NotImplementedError(
            "The papers do not publish complete definitions for the four threat sub-functions"
        )
