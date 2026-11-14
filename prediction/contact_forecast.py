"""Counterfactual geometric contact on two explicitly supplied 3D tracks."""
from dataclasses import dataclass

import numpy as np

from env.ideal_interception.geometry import proximity
from .forecast import TargetForecast


@dataclass(frozen=True)
class ContactForecast:
    defender_id: int
    assigned_target_id: int
    prediction_method: str
    observation_time: float | None
    forecast_window: tuple[float, float] | None
    contact_predicted: bool
    predicted_contact_time: float | None
    predicted_contact_position: tuple[float, float, float] | None
    target_forecast_status: str
    defender_trajectory_source: str
    scenario_id: str | None
    status: str


def estimate_contact(defender_id, assigned_target_id, target: TargetForecast,
                     defender_times, defender_positions, contact_distance,
                     defender_trajectory_source="recorded_virtual"):
    """Interpolate supplied knots linearly and find first continuous sphere contact.

    The forecast is conditional on the given defender track; it never changes
    an environment event, action, assignment or member state.
    """
    if (defender_id < 0 or assigned_target_id != target.target_id or
            not np.isfinite(contact_distance) or contact_distance <= 0 or
            defender_trajectory_source not in ("recorded_virtual", "fixed_single_step", "experiment_assumption")):
        raise ValueError("invalid contact forecast identity, geometry or source")
    times = np.asarray(defender_times, dtype=np.float64)
    positions = np.asarray(defender_positions, dtype=np.float64)
    if (times.ndim != 1 or len(times) < 2 or positions.shape != (len(times), 3)
            or not np.isfinite(times).all() or not np.isfinite(positions).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError("invalid defender trajectory")
    ready = target.status == "READY" and len(target.future_times) > 0
    window = (float(target.observation_time), float(target.future_times[-1])) if ready else None
    base = dict(defender_id=defender_id, assigned_target_id=assigned_target_id,
                prediction_method=target.method, observation_time=target.observation_time,
                forecast_window=window, target_forecast_status=target.status,
                defender_trajectory_source=defender_trajectory_source,
                scenario_id=target.scenario_id)
    if not ready:
        return ContactForecast(**base, contact_predicted=False, predicted_contact_time=None,
                               predicted_contact_position=None, status="NOT_EVALUABLE")
    target_times = np.asarray((target.observation_time, *target.future_times), dtype=float)
    target_positions = np.asarray((target.current_position, *target.future_positions), dtype=float)
    if (times[0] > target_times[0]+1e-9 or times[-1] < target_times[-1]-1e-9):
        return ContactForecast(**base, contact_predicted=False, predicted_contact_time=None,
                               predicted_contact_position=None, status="NOT_EVALUABLE")
    knots = sorted(set(target_times.tolist()+[float(t) for t in times if target_times[0] < t < target_times[-1]]))
    def at(clock, track_times, track_positions):
        index = min(max(np.searchsorted(track_times, clock, side="right")-1, 0), len(track_times)-2)
        fraction = (clock-track_times[index])/(track_times[index+1]-track_times[index])
        return track_positions[index]+fraction*(track_positions[index+1]-track_positions[index])
    for left, right in zip(knots[:-1], knots[1:]):
        p = at(left, times, positions)
        q = at(left, target_times, target_positions)
        v = (at(right, times, positions)-p)/(right-left)
        w = (at(right, target_times, target_positions)-q)/(right-left)
        tau = proximity(p, v, q, w, contact_distance, right-left)
        if tau is not None:
            # Match events.py's first recorded position: defender sphere center.
            point = p+v*tau
            return ContactForecast(**base, contact_predicted=True,
                predicted_contact_time=float(left+tau),
                predicted_contact_position=tuple(float(x) for x in point), status="CONTACT_IN_FORECAST_WINDOW")
    return ContactForecast(**base, contact_predicted=False, predicted_contact_time=None,
                           predicted_contact_position=None, status="NO_CONTACT_IN_FORECAST_WINDOW")
