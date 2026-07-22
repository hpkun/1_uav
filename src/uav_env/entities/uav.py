"""UAV entity container."""

from __future__ import annotations

from dataclasses import dataclass, replace

from uav_env.actions.discrete_15 import DiscreteAction15, get_control
from uav_env.core.control import ControlInput
from uav_env.core.state import UAVState
from uav_env.dynamics.propagation import ActionHoldResult, propagate_action_hold
from uav_env.entities.type_profiles import UAVTypeProfile


@dataclass
class UAV:
    """Associate mutable UAV state with an immutable type profile."""

    uav_id: str
    team: int
    state: UAVState
    profile: UAVTypeProfile
    initial_state: UAVState | None = None

    def __post_init__(self) -> None:
        if self.state.type_id != self.profile.type_id:
            raise ValueError("State and profile type_id values must match")
        if self.state.team_id != self.team:
            raise ValueError("State and UAV team values must match")
        if self.initial_state is None:
            self.initial_state = self.state.copy()

    @property
    def is_alive(self) -> bool:
        """Return whether the aircraft remains active."""

        return self.state.alive

    def control_for_action(self, action: int | DiscreteAction15) -> ControlInput:
        """Resolve one discrete action to its overload control."""

        return get_control(action)

    def execute_action_hold(
        self,
        action: int | DiscreteAction15,
        physics_dt: float,
        physics_steps: int,
        gravity: float,
        min_altitude: float,
        max_altitude: float,
    ) -> ActionHoldResult:
        """Execute an action hold and store its final state."""

        self.state = replace(self.state, last_action=int(action))
        result = propagate_action_hold(
            self.state,
            self.control_for_action(action),
            self.profile,
            physics_dt,
            physics_steps,
            gravity,
            min_altitude,
            max_altitude,
        )
        self.state = result.final_state
        return result

    def reset(self, state: UAVState | None = None) -> None:
        """Reset to a supplied state or the construction-time state."""

        chosen = state if state is not None else self.initial_state
        if chosen is None:
            raise RuntimeError("No initial state is available")
        self.state = chosen.copy()
