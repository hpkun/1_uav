from __future__ import annotations

from conftest import make_state
from uav_env.actions.discrete_15 import DiscreteAction15, get_control
from uav_env.dynamics.propagation import propagate_action_hold
from uav_env.entities.type_profiles import UAVTypeProfile


def test_action_hold_executes_exactly_five_steps(profile: UAVTypeProfile) -> None:
    result = propagate_action_hold(
        make_state(profile),
        get_control(DiscreteAction15.LEVEL_HOLD),
        profile,
        0.1,
        5,
        9.81,
        0.0,
        5000.0,
    )
    assert len(result.substep_states) == 5
    assert result.final_state.x == 50.0
    assert not result.ground_crash
    assert not result.ceiling_violation


def test_action_hold_stops_immediately_at_ground(profile: UAVTypeProfile) -> None:
    state = make_state(profile, z=0.01)
    state.flight_path_angle = profile.min_flight_path_angle
    result = propagate_action_hold(state, get_control(DiscreteAction15.DIVE_HOLD), profile, 0.1, 5, 9.81, 0.0, 5000.0)
    assert result.ground_crash
    assert len(result.substep_states) == 1
    assert not result.final_state.alive
