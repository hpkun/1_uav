"""Abstract rule-opponent contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

from uav_env.actions.discrete_15 import DiscreteAction15
from uav_env.core.state import UAVState


class RuleOpponent(ABC):
    """Base class for deterministic or stochastic rule opponents."""

    @abstractmethod
    def select_action(self, ownship: UAVState, opponent: UAVState) -> DiscreteAction15:
        """Select one discrete action from current states."""

        raise NotImplementedError
