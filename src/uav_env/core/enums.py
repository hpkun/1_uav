"""Shared enumerations."""

from enum import Enum, IntEnum


class Team(IntEnum):
    """Conventional team identifiers."""

    BLUE = 0
    RED = 1


class CombatEventType(str, Enum):
    """Events emitted by a combat episode."""

    ENTER_ATTACK_AREA = "enter_attack_area"
    ENTER_ADVANTAGE_AREA = "enter_advantage_area"
    ATTACK_TRIGGERED = "attack_triggered"
    HIT = "hit"
    MISS = "miss"
    DESTROYED = "destroyed"
    GROUND_CRASH = "ground_crash"
    CEILING_VIOLATION = "ceiling_violation"
    COLLISION = "collision"
    TIMEOUT = "timeout"
    WIN = "win"
    LOSS = "loss"
    DRAW = "draw"
