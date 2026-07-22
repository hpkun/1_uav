"""Uniform random discrete-action opponent."""

from __future__ import annotations

import numpy as np

from uav_env.actions.discrete_15 import DiscreteAction15
from uav_env.core.state import UAVState
from uav_env.opponents.base import RuleOpponent


class RandomOpponent(RuleOpponent):
    """Sample uniformly using only the generator supplied by the environment."""

    def select_action(
        self,
        ownship: UAVState,
        opponent: UAVState,
        rng: np.random.Generator | None = None,
    ) -> DiscreteAction15:
        """Return one reproducible uniform action."""

        del ownship, opponent
        if rng is None:
            raise ValueError("RandomOpponent requires an explicit numpy Generator")
        return DiscreteAction15(int(rng.integers(0, len(DiscreteAction15))))
