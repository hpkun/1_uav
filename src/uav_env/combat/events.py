"""Structured combat events and episode outcomes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from uav_env.core.enums import CombatEventType


@dataclass(frozen=True)
class CombatEvent:
    """One event emitted at a decision boundary."""

    event_type: CombatEventType
    decision_step: int
    simulation_time: float
    source_id: str | None = None
    target_id: str | None = None
    value: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeOutcome:
    """Current or terminal episode summary from the red-agent viewpoint."""

    winner: str | None
    red_alive: bool
    blue_alive: bool
    termination_reason: str
    decision_steps: int
    simulation_time: float
    red_survivors: int | None = None
    blue_survivors: int | None = None
