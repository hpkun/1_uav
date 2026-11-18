"""Causal G0/G1/G2/G3 guidance; every command uses the shared dynamics API."""
from dataclasses import dataclass, field

import numpy as np

from prediction.intercept_prediction import solve_constant_velocity_pip

from .dynamics import Command, DynamicsConfig, PointState, WingState
from .prediction import TargetHistory, feasible_intercept, forecast


@dataclass
class Controller:
    method: str
    dynamics: DynamicsConfig
    history: TargetHistory
    replan_interval: float = .25
    _cached: dict[int, tuple[float, Command]] = field(default_factory=dict)

    def __post_init__(self):
        if self.method not in ("G0", "G1", "G2", "G3", "G4"):
            raise ValueError("unknown guidance method")
        if self.method == "G4":
            from .mpc.controller import MPCController
            self._mpc = MPCController(self.dynamics,self.history)

    def command(self, index: int, state: PointState | WingState,
                target_index: int, target_position, time: float):
        if self.method == "G4":
            return self._mpc.command(index,state,target_index,target_position,time)
        target = np.asarray(target_position, float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("invalid observed target position")
        if self.method in ("G2", "G3") and index in self._cached:
            previous_time, previous_command = self._cached[index]
            if time-previous_time < self.replan_interval-1e-9:
                return previous_command, {"status": "HELD_COMMAND"}
        aim = target
        status = "PURE_PURSUIT"
        if self.method != "G0":
            forecast_method = "CA" if self.method == "G3" else "CV"
            try:
                predicted = forecast(self.history, target_index, forecast_method)
            except ValueError:
                predicted = None
            if predicted is not None:
                if self.method == "G1":
                    pip = solve_constant_velocity_pip(state.position, self.dynamics.v_max,
                                                      predicted.position, predicted.velocity)
                    if pip.time_to_pip_s is not None:
                        aim = predicted.at(pip.time_to_pip_s)
                        status = "THREE_POINT_CV_PIP"
                    else:
                        status = "PIP_NO_POSITIVE_ROOT"
                else:
                    plan = feasible_intercept(state, predicted, self.dynamics)
                    command = plan.command
                    self._cached[index] = (time, command)
                    return command, {"status": plan.status,
                                     "closest_time_s": plan.closest_time_s,
                                     "minimum_distance_m": plan.minimum_distance_m,
                                     "reference_pip_time_s": plan.reference_pip_time_s}
            else:
                status = "INSUFFICIENT_HISTORY"
        delta = aim-state.position
        norm = float(np.linalg.norm(delta))
        if norm <= 1e-12:
            delta = target-state.position
            norm = float(np.linalg.norm(delta))
        if norm <= 1e-12:
            delta = np.array((1., 0., 0.))
            norm = 1.
        command = Command(tuple(delta/norm), self.dynamics.v_max)
        if self.method in ("G2", "G3"):
            self._cached[index] = (time, command)
        return command, {"status": status}
