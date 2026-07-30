"""Shared enumerations."""

from enum import Enum, IntEnum


class Team(IntEnum):
    """Conventional team identifiers."""

    BLUE = 0
    RED = 1


class CombatRole(str, Enum):
    """Functional role independent of physical flight profile."""

    COMBAT = "combat"
    SUPPORT = "support"


def role_flag(role: str | CombatRole) -> float:
    """Return normalized role flag: support=+1, combat=-1."""

    parsed = CombatRole(role)
    return 1.0 if parsed is CombatRole.SUPPORT else -1.0


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
