"""The fixed 15-action manoeuvre table."""

from __future__ import annotations

from enum import IntEnum
from math import acos, cos, isclose

from uav_env.core.control import ControlInput


class DiscreteAction15(IntEnum):
    """Identifiers for the 15 discrete overload commands."""

    LEVEL_HOLD = 0
    LEVEL_ACCELERATE = 1
    LEVEL_DECELERATE = 2
    CLIMB_HOLD = 3
    CLIMB_ACCELERATE = 4
    CLIMB_DECELERATE = 5
    DIVE_HOLD = 6
    DIVE_ACCELERATE = 7
    DIVE_DECELERATE = 8
    LEFT_HOLD = 9
    LEFT_ACCELERATE = 10
    LEFT_DECELERATE = 11
    RIGHT_HOLD = 12
    RIGHT_ACCELERATE = 13
    RIGHT_DECELERATE = 14


_TURN_BANK = acos(2.0 / 7.0)

ACTION_NAMES: dict[DiscreteAction15, str] = {
    DiscreteAction15.LEVEL_HOLD: "平飞保持",
    DiscreteAction15.LEVEL_ACCELERATE: "平飞加速",
    DiscreteAction15.LEVEL_DECELERATE: "平飞减速",
    DiscreteAction15.CLIMB_HOLD: "爬升保持",
    DiscreteAction15.CLIMB_ACCELERATE: "爬升加速",
    DiscreteAction15.CLIMB_DECELERATE: "爬升减速",
    DiscreteAction15.DIVE_HOLD: "俯冲保持",
    DiscreteAction15.DIVE_ACCELERATE: "俯冲加速",
    DiscreteAction15.DIVE_DECELERATE: "俯冲减速",
    DiscreteAction15.LEFT_HOLD: "左转保持",
    DiscreteAction15.LEFT_ACCELERATE: "左转加速",
    DiscreteAction15.LEFT_DECELERATE: "左转减速",
    DiscreteAction15.RIGHT_HOLD: "右转保持",
    DiscreteAction15.RIGHT_ACCELERATE: "右转加速",
    DiscreteAction15.RIGHT_DECELERATE: "右转减速",
}

ACTION_CONTROLS: dict[DiscreteAction15, ControlInput] = {
    DiscreteAction15.LEVEL_HOLD: ControlInput(0.0, 1.0, 0.0),
    DiscreteAction15.LEVEL_ACCELERATE: ControlInput(2.0, 1.0, 0.0),
    DiscreteAction15.LEVEL_DECELERATE: ControlInput(-1.0, 1.0, 0.0),
    DiscreteAction15.CLIMB_HOLD: ControlInput(0.0, 3.5, 0.0),
    DiscreteAction15.CLIMB_ACCELERATE: ControlInput(2.0, 3.5, 0.0),
    DiscreteAction15.CLIMB_DECELERATE: ControlInput(-1.0, 3.5, 0.0),
    DiscreteAction15.DIVE_HOLD: ControlInput(0.0, -3.5, 0.0),
    DiscreteAction15.DIVE_ACCELERATE: ControlInput(2.0, -3.5, 0.0),
    DiscreteAction15.DIVE_DECELERATE: ControlInput(-1.0, -3.5, 0.0),
    DiscreteAction15.LEFT_HOLD: ControlInput(0.0, 3.5, _TURN_BANK),
    DiscreteAction15.LEFT_ACCELERATE: ControlInput(2.0, 3.5, _TURN_BANK),
    DiscreteAction15.LEFT_DECELERATE: ControlInput(-1.0, 3.5, _TURN_BANK),
    DiscreteAction15.RIGHT_HOLD: ControlInput(0.0, 3.5, -_TURN_BANK),
    DiscreteAction15.RIGHT_ACCELERATE: ControlInput(2.0, 3.5, -_TURN_BANK),
    DiscreteAction15.RIGHT_DECELERATE: ControlInput(-1.0, 3.5, -_TURN_BANK),
}


def _coerce_action(action: int | DiscreteAction15) -> DiscreteAction15:
    try:
        return DiscreteAction15(action)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid discrete action: {action!r}") from exc


def get_control(action: int | DiscreteAction15) -> ControlInput:
    """Return the immutable control command for *action*."""

    return ACTION_CONTROLS[_coerce_action(action)]


def get_action_name(action: int | DiscreteAction15) -> str:
    """Return the Chinese display name for *action*."""

    return ACTION_NAMES[_coerce_action(action)]


def validate_action_table() -> None:
    """Raise if the action table violates its defining invariants."""

    actions = list(DiscreteAction15)
    if len(actions) != 15 or [int(action) for action in actions] != list(range(15)):
        raise ValueError("DiscreteAction15 must contain exactly the IDs 0 through 14")
    if set(ACTION_NAMES) != set(actions) or set(ACTION_CONTROLS) != set(actions):
        raise ValueError("Every action must have exactly one name and control mapping")
    left = get_control(DiscreteAction15.LEFT_HOLD)
    right = get_control(DiscreteAction15.RIGHT_HOLD)
    if not isclose(left.bank_angle, -right.bank_angle):
        raise ValueError("Left and right bank angles must have opposite signs")
    if not isclose(3.5 * cos(left.bank_angle), 1.0):
        raise ValueError("Turn bank angle does not preserve unit vertical overload")
