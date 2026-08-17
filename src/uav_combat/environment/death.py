"""Auditable one-way aircraft death ledger."""
from __future__ import annotations

from enum import IntEnum
import numpy as np


class DeathCause(IntEnum):
    NONE = 0
    ATTACK = 1
    BOUNDARY = 2


def death_summary(causes: np.ndarray) -> dict[str, int]:
    return {
        "survivors": int(np.count_nonzero(causes == DeathCause.NONE)),
        "attack_deaths": int(np.count_nonzero(causes == DeathCause.ATTACK)),
        "boundary_deaths": int(np.count_nonzero(causes == DeathCause.BOUNDARY)),
    }
