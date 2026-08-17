"""Equation (23) action mapping and the minimal unpublished low-level bridge."""
from __future__ import annotations

import numpy as np
from .math_utils import angle_difference, safe_clip, wrap_angle
from .models import AircraftSpec, AircraftState, ControlCommand, TargetCommand


class TargetStateController:
    def __init__(
        self,
        delta_yaw_max: float = np.pi,
        delta_pitch_max: float = np.pi / 3,
        delta_speed_max: float = 50.0,
        gravity: float = 9.81,
    ) -> None:
        self.delta_yaw_max = float(delta_yaw_max)
        self.delta_pitch_max = float(delta_pitch_max)
        self.delta_speed_max = float(delta_speed_max)
        self.gravity = float(gravity)

    def action_to_target(self, state: AircraftState, action: np.ndarray, spec: AircraftSpec) -> TargetCommand:
        """Table 2 / Eq.(23): normalized network output to desired state."""
        action = np.asarray(action, dtype=float)
        if action.shape != (3,) or not np.all(np.isfinite(action)):
            raise ValueError("action must be a finite array with shape (3,)")
        action = np.clip(action, -1.0, 1.0)
        return TargetCommand(
            wrap_angle(state.psi + action[0] * self.delta_yaw_max),
            safe_clip(state.theta + action[1] * self.delta_pitch_max, spec.theta_min, spec.theta_max),
            safe_clip(state.v + action[2] * self.delta_speed_max, spec.v_min, spec.v_max),
        )

    def compute_control(self, state: AircraftState, target: TargetCommand, spec: AircraftSpec) -> ControlCommand:
        """Assumed proportional inverse-dynamics bridge to [phi,nz,nx]."""
        psi_dot = safe_clip(spec.k_yaw * angle_difference(target.desired_psi, state.psi), -spec.yaw_rate_max, spec.yaw_rate_max)
        theta_dot = safe_clip(spec.k_pitch * (target.desired_theta - state.theta), -spec.pitch_rate_max, spec.pitch_rate_max)
        v_dot = safe_clip(spec.k_speed * (target.desired_v - state.v), -spec.acceleration_max, spec.acceleration_max)
        nx = safe_clip(v_dot / self.gravity + np.sin(state.theta), spec.nx_min, spec.nx_max)
        vertical = np.cos(state.theta) + state.v * theta_dot / self.gravity
        lateral = state.v * np.cos(state.theta) * psi_dot / self.gravity
        nz = safe_clip(float(np.hypot(vertical, lateral)), spec.nz_min, spec.nz_max)
        phi = safe_clip(float(np.arctan2(lateral, vertical)), spec.phi_min, spec.phi_max)
        return ControlCommand(nx, nz, phi)

    def control_from_action(self, state: AircraftState, action: np.ndarray, spec: AircraftSpec) -> tuple[TargetCommand, ControlCommand]:
        target = self.action_to_target(state, action, spec)
        return target, self.compute_control(state, target, spec)
