"""Straight-flight opponent structure."""

import numpy as np

from uav_env.actions.discrete_15 import DiscreteAction15
from uav_env.core.state import UAVState
from uav_env.opponents.base import RuleOpponent


class StraightOpponent(RuleOpponent):
    """Transparent baseline that always maintains straight, level flight."""

    def select_action(
        self,
        ownship: UAVState,
        opponent: UAVState,
        rng: np.random.Generator | None = None,
    ) -> DiscreteAction15:
        """Return the level-hold action."""

        del ownship, opponent, rng
        return DiscreteAction15.LEVEL_HOLD
