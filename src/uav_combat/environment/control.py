"""High-level maneuver commands mapped to the 3DOF point-mass controls."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..math_utils import wrap_angle
from ..models import AircraftState, ControlCommand


@dataclass(frozen=True)
class ManeuverTarget:
    heading: float
    pitch: float
    speed: float


def action_to_target(
    state: AircraftState, action: np.ndarray, config: dict
) -> ManeuverTarget:
    """Decode normalized ``[heading, pitch, speed]`` maneuver commands."""
    heading, pitch, speed = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
    speed_min = float(config["speed_command_min"])
    speed_max = float(config["speed_command_max"])
    return ManeuverTarget(
        heading=float(wrap_angle(
            state.psi + float(config["heading_delta_max"]) * heading
        )),
        pitch=float(config["pitch_command_max"] * pitch),
        speed=float(speed_min + 0.5 * (speed + 1.0) * (speed_max - speed_min)),
    )


def target_to_control(
    state: AircraftState,
    target: ManeuverTarget,
    controller: dict,
    gravity: float = 9.81,
) -> ControlCommand:
    """Apply a bounded, memoryless first-order aircraft response."""
    heading_rate = float(np.clip(
        float(controller["heading_gain"]) * wrap_angle(target.heading - state.psi),
        -float(controller["heading_rate_max"]),
        float(controller["heading_rate_max"]),
    ))
    pitch_rate = float(np.clip(
        float(controller["pitch_gain"]) * (target.pitch - state.theta),
        -float(controller["pitch_rate_max"]),
        float(controller["pitch_rate_max"]),
    ))
    acceleration = float(np.clip(
        float(controller["speed_gain"]) * (target.speed - state.v),
        -float(controller["acceleration_max"]),
        float(controller["acceleration_max"]),
    ))

    # Solve the existing theta/psi rate equations for nz and phi.
    vertical = max(
        np.cos(state.theta) + state.v * pitch_rate / gravity,
        float(controller["normal_load_min"]),
    )
    lateral = state.v * np.cos(state.theta) * heading_rate / gravity
    bank_max = float(controller["bank_max"])
    lateral_limit = abs(vertical) * np.tan(bank_max)
    lateral = float(np.clip(lateral, -lateral_limit, lateral_limit))
    phi = float(np.arctan2(lateral, vertical))
    nz = float(np.clip(
        np.hypot(vertical, lateral),
        float(controller["normal_load_min"]),
        float(controller["normal_load_max"]),
    ))
    nx = float(np.clip(
        np.sin(state.theta) + acceleration / gravity,
        float(controller["tangential_load_min"]),
        float(controller["tangential_load_max"]),
    ))
    return ControlCommand(nx=nx, nz=nz, phi=phi)


def action_to_control(
    state: AircraftState, action: np.ndarray, config: dict, gravity: float = 9.81
) -> ControlCommand:
    """Map one normalized maneuver action through the response layer."""
    return target_to_control(
        state,
        action_to_target(state, action, config["command"]),
        config["controller"],
        gravity,
    )


__all__ = [
    "ManeuverTarget", "action_to_control", "action_to_target", "target_to_control"
]
