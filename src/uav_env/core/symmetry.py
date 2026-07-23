"""Exact reflection mappings used by environment fairness diagnostics."""

from __future__ import annotations

from dataclasses import replace
from math import pi

from uav_env.actions.discrete_15 import DiscreteAction15
from uav_env.core.state import UAVState


_ACTION_MIRROR = {
    DiscreteAction15.LEFT_HOLD: DiscreteAction15.RIGHT_HOLD,
    DiscreteAction15.LEFT_ACCELERATE: DiscreteAction15.RIGHT_ACCELERATE,
    DiscreteAction15.LEFT_DECELERATE: DiscreteAction15.RIGHT_DECELERATE,
    DiscreteAction15.RIGHT_HOLD: DiscreteAction15.LEFT_HOLD,
    DiscreteAction15.RIGHT_ACCELERATE: DiscreteAction15.LEFT_ACCELERATE,
    DiscreteAction15.RIGHT_DECELERATE: DiscreteAction15.LEFT_DECELERATE,
}


def mirror_state_xz(state: UAVState) -> UAVState:
    """Reflect a state across the x-z plane."""

    return replace(state, y=-state.y, heading_angle=(-state.heading_angle) % (2.0 * pi))


def mirror_action_xz(action: int | DiscreteAction15) -> DiscreteAction15:
    """Swap left/right maneuvers while retaining longitudinal behavior."""

    parsed = DiscreteAction15(int(action))
    return _ACTION_MIRROR.get(parsed, parsed)
