"""Constrained state propagation built on the point-mass model."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import pi
from typing import TYPE_CHECKING

import numpy as np

from uav_env.core.control import ControlInput
from uav_env.core.state import UAVState
from uav_env.dynamics.point_mass_3d import point_mass_3d_derivative
from uav_env.dynamics.rk4 import rk4_step
if TYPE_CHECKING:
    from uav_env.entities.type_profiles import UAVTypeProfile


@dataclass(frozen=True)
class ActionHoldResult:
    """Result and diagnostic samples from one held discrete action."""

    final_state: UAVState
    substep_states: list[UAVState]
    ground_crash: bool
    ceiling_violation: bool
    actual_control: ControlInput


def clip_control(control: ControlInput, profile: UAVTypeProfile) -> ControlInput:
    """Clip an overload command to platform performance limits."""

    return ControlInput(
        float(np.clip(control.tangential_overload, profile.min_tangential_overload, profile.max_tangential_overload)),
        float(np.clip(control.normal_overload, profile.min_normal_overload, profile.max_normal_overload)),
        control.bank_angle,
    )


def propagate_state(
    state: UAVState,
    control: ControlInput,
    profile: UAVTypeProfile,
    dt: float,
    gravity: float = 9.81,
) -> UAVState:
    """Advance one physical step and enforce platform kinematic limits."""

    state.validate_finite()
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    actual = clip_control(control, profile)

    def derivative(time: float, vector: np.ndarray, command: ControlInput) -> np.ndarray:
        del time
        return point_mass_3d_derivative(vector, command, gravity=gravity)

    vector = rk4_step(derivative, 0.0, state.to_kinematic_vector(), dt, actual)
    vector[3] = np.clip(vector[3], profile.min_speed, profile.max_speed)
    vector[4] = np.clip(vector[4], profile.min_flight_path_angle, profile.max_flight_path_angle)
    vector[5] = vector[5] % (2.0 * pi)
    result = state.with_kinematic_vector(vector)
    result.validate_finite()
    return result


def propagate_action_hold(
    state: UAVState,
    control: ControlInput,
    profile: UAVTypeProfile,
    physics_dt: float,
    physics_steps: int,
    gravity: float,
    min_altitude: float,
    max_altitude: float,
) -> ActionHoldResult:
    """Hold one command for fixed physical substeps, stopping at boundaries."""

    if physics_steps <= 0:
        raise ValueError("physics_steps must be positive")
    if not min_altitude < max_altitude:
        raise ValueError("Altitude bounds are invalid")
    actual = clip_control(control, profile)
    current = state.copy()
    samples: list[UAVState] = []
    ground_crash = False
    ceiling_violation = False
    for _ in range(physics_steps):
        current = propagate_state(current, actual, profile, physics_dt, gravity)
        ground_crash = current.z <= min_altitude
        ceiling_violation = current.z > max_altitude
        if ground_crash or ceiling_violation:
            current = replace(current, health=0.0, alive=False, crashed=True, damaged=True)
        samples.append(current.copy())
        if ground_crash or ceiling_violation:
            break
    return ActionHoldResult(current, samples, ground_crash, ceiling_violation, actual)
