"""UAV entity container."""

from __future__ import annotations

from dataclasses import dataclass

from uav_env.core.state import UAVState
from uav_env.entities.type_profiles import UAVTypeProfile


@dataclass
class UAV:
    """Associate mutable UAV state with an immutable type profile."""

    state: UAVState
    profile: UAVTypeProfile

    def __post_init__(self) -> None:
        if self.state.type_id != self.profile.type_id:
            raise ValueError("State and profile type_id values must match")
