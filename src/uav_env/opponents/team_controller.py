"""Symmetric fixed-size team rule controller for fair rule evaluation."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from uav_env.actions.discrete_15 import DiscreteAction15
from uav_env.combat.multi_combat import TargetAssignment, assign_nearest_targets_independently, assign_targets
from uav_env.entities.uav import UAV
from uav_env.opponents.base import RuleOpponent


class TeamRuleController:
    """Apply one rule and the same stable target assignment to either team."""

    def __init__(self, name: str, policy: RuleOpponent, seed: int) -> None:
        if name not in {"straight", "random", "pursuit"}:
            raise ValueError(f"Unknown team rule: {name!r}")
        self.name = name
        self.policy = policy
        self.rng = np.random.default_rng(seed)

    def select_actions(self, team: Sequence[UAV], opponents: Sequence[UAV]) -> tuple[list[DiscreteAction15], list[TargetAssignment]]:
        """Return ID-ordered actions and fixed-size nearest assignments."""

        ordered = sorted(team, key=lambda u: u.uav_id)
        assignments = assign_nearest_targets_independently(ordered, opponents) if len(ordered) == 3 else assign_targets(ordered, opponents)
        assignment_map = {item.attacker_id: item.target_id for item in assignments}
        target_map = {u.uav_id: u for u in opponents}
        actions: list[DiscreteAction15] = []
        for aircraft in ordered:
            target_id = assignment_map.get(aircraft.uav_id)
            if not aircraft.is_alive or target_id is None:
                actions.append(DiscreteAction15.LEVEL_HOLD)
            else:
                actions.append(self.policy.select_action(aircraft.state.copy(), target_map[target_id].state.copy(), self.rng))
        return actions, assignments
