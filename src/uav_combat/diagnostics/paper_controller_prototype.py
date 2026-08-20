"""Diagnostic-only controller constrained by Li et al. (2023).

Nothing in this module is imported by the active environment.  The action ranges and
3-DOF model are PAPER; the desired-state interface is also predecessor-supported.  The
P feedback architecture is a RECONSTRUCTION and its algebraic inversion is DERIVED.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models import AircraftState, ControlCommand


PSI_INCREMENT_MAX = np.pi
THETA_INCREMENT_MAX = np.pi / 3.0
SPEED_INCREMENT_MAX = 50.0
THETA_MIN, THETA_MAX = -np.pi / 3.0, np.pi / 3.0
SPEED_MIN, SPEED_MAX = 150.0, 300.0
PHI_MIN, PHI_MAX = -np.pi / 2.0, np.pi / 2.0


def wrap_angle(value: float) -> float:
    return float((value + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass(frozen=True)
class DesiredCommand:
    psi: float
    theta: float
    speed: float
    delta_psi: float
    delta_theta: float
    delta_speed: float


@dataclass(frozen=True)
class PrototypeControl:
    command: ControlCommand
    raw_nx: float
    raw_nz: float
    raw_phi: float
    v_dot_command: float
    theta_dot_command: float
    psi_dot_command: float
    speed_error: float
    theta_error: float
    psi_error: float


def command_from_normalized(state: AircraftState, action: np.ndarray) -> DesiredCommand:
    """Map diagnostic ``[a_psi,a_theta,a_v]`` to the paper high-level command."""
    a_psi, a_theta, a_speed = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
    delta_psi = float(a_psi * PSI_INCREMENT_MAX)
    delta_theta = float(a_theta * THETA_INCREMENT_MAX)
    delta_speed = float(a_speed * SPEED_INCREMENT_MAX)
    return DesiredCommand(
        psi=wrap_angle(state.psi + delta_psi),
        theta=float(np.clip(state.theta + delta_theta, THETA_MIN, THETA_MAX)),
        speed=float(np.clip(state.v + delta_speed, SPEED_MIN, SPEED_MAX)),
        delta_psi=delta_psi,
        delta_theta=delta_theta,
        delta_speed=delta_speed,
    )


class ModelFeedbackPController:
    """P feedback on desired rates followed by exact 3-DOF algebraic inversion."""

    def __init__(
        self, tau_psi: float, tau_theta: float, tau_speed: float, gravity: float = 9.81
    ) -> None:
        if min(tau_psi, tau_theta, tau_speed) <= 0.0:
            raise ValueError("time constants must be positive")
        self.tau_psi = float(tau_psi)
        self.tau_theta = float(tau_theta)
        self.tau_speed = float(tau_speed)
        self.gravity = float(gravity)

    def control(self, state: AircraftState, desired: DesiredCommand) -> PrototypeControl:
        e_speed = float(desired.speed - state.v)
        e_theta = float(desired.theta - state.theta)
        e_psi = wrap_angle(desired.psi - state.psi)
        v_dot = e_speed / self.tau_speed
        theta_dot = e_theta / self.tau_theta
        psi_dot = e_psi / self.tau_psi
        nx = float(np.sin(state.theta) + v_dot / self.gravity)
        a_term = float(np.cos(state.theta) + state.v * theta_dot / self.gravity)
        b_term = float(state.v * np.cos(state.theta) * psi_dot / self.gravity)
        nz = float(np.hypot(a_term, b_term))
        raw_phi = float(np.arctan2(b_term, a_term))
        phi = float(np.clip(raw_phi, PHI_MIN, PHI_MAX))
        return PrototypeControl(
            command=ControlCommand(nx=nx, nz=nz, phi=phi),
            raw_nx=nx, raw_nz=nz, raw_phi=raw_phi,
            v_dot_command=v_dot, theta_dot_command=theta_dot,
            psi_dot_command=psi_dot, speed_error=e_speed,
            theta_error=e_theta, psi_error=e_psi,
        )


class FeasibleProjectedPController(ModelFeedbackPController):
    """Project inversion onto the PAPER roll-feasible half-plane before control.

    With ``nz >= 0`` and ``|phi| <= pi/2``, the realizable component
    ``nz*cos(phi)`` cannot be negative.  When the unconstrained inversion asks for
    ``A < 0``, this candidate uses the closest feasible value ``A=0`` while
    preserving the requested yaw component B.  This is a DERIVED projection, not a
    published controller law.
    """

    def control(self, state: AircraftState, desired: DesiredCommand) -> PrototypeControl:
        e_speed = float(desired.speed - state.v)
        e_theta = float(desired.theta - state.theta)
        e_psi = wrap_angle(desired.psi - state.psi)
        v_dot = e_speed / self.tau_speed
        theta_dot = e_theta / self.tau_theta
        psi_dot = e_psi / self.tau_psi
        nx = float(np.sin(state.theta) + v_dot / self.gravity)
        a_raw = float(np.cos(state.theta) + state.v * theta_dot / self.gravity)
        b_term = float(state.v * np.cos(state.theta) * psi_dot / self.gravity)
        raw_nz = float(np.hypot(a_raw, b_term))
        raw_phi = float(np.arctan2(b_term, a_raw))
        a_feasible = max(a_raw, 0.0)
        nz = float(np.hypot(a_feasible, b_term))
        phi = float(np.clip(np.arctan2(b_term, a_feasible), PHI_MIN, PHI_MAX))
        return PrototypeControl(
            command=ControlCommand(nx=nx, nz=nz, phi=phi),
            raw_nx=nx, raw_nz=raw_nz, raw_phi=raw_phi,
            v_dot_command=v_dot, theta_dot_command=theta_dot,
            psi_dot_command=psi_dot, speed_error=e_speed,
            theta_error=e_theta, psi_error=e_psi,
        )


EVIDENCE = {
    "normalized_action_mapping": "RECONSTRUCTION representation of PAPER ranges",
    "action_ranges": "PAPER Table 2",
    "desired_state_equation": "PAPER Eq. (23)",
    "desired_to_control_interface": "PAPER plus PREDECESSOR support",
    "3dof_dynamics": "PAPER Eq. (2)",
    "feedback_architecture": "PREDECESSOR-supported idea; P-only law is RECONSTRUCTION",
    "inversion": "DERIVED from Eq. (2)",
    "feasible_projection": "DERIVED from nz>=0 and PAPER |phi|<=pi/2; candidate only",
    "phi_limit": "PAPER Table 1",
    "nx_nz_limits": "not applied; paper underspecified",
}


__all__ = [
    "DesiredCommand", "EVIDENCE", "FeasibleProjectedPController",
    "ModelFeedbackPController", "PrototypeControl",
    "command_from_normalized", "wrap_angle",
]
