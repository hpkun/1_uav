"""Causal 1v1 target-point estimates; no environment actions are chosen here."""
from dataclasses import dataclass
import math

import numpy as np

from .intercept_time import zuo_lookahead_time
from .three_point import predict_at
from .zuo_linear import predict_zuo_at


@dataclass(frozen=True)
class InterceptPrediction:
    method: str
    status: str
    observation_time_s: float | None
    horizon_s: float | None
    predicted_time_s: float | None
    predicted_target_position_m: tuple[float, float, float] | None
    estimated_target_velocity_mps: tuple[float, float, float] | None
    estimated_target_acceleration_mps2: tuple[float, float, float] | None
    horizon_kind: str
    equation_residual_m: float | None
    history_count: int
    paper_direction_unit: tuple[float, float, float] | None = None
    range_m: float | None = None
    radial_target_speed_mps: float | None = None
    effective_closing_speed_mps: float | None = None
    range_horizon_s: float | None = None
    maneuver_horizon_s: float | None = None
    horizon_limited_by: str | None = None


def _vector(value):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError("defender position must be a finite 3-vector")
    return array


def _tuple(value):
    return tuple(float(x) for x in value)


def _unavailable(method, kind, history, target_index, status):
    times, positions = history.observed(target_index)
    return InterceptPrediction(method, status, float(times[-1]) if len(times) else None,
        None, None, None, None, None, kind, None, len(positions))


def estimate_zuo_prediction_from_history(history, target_index, defender_position,
                                         defender_speed):
    """Zuo equations (30)-(33), with the paper horizon kept separate from contact time."""
    defender = _vector(defender_position)
    status = history.method_status(target_index, "zuo_linear")
    if status != "READY":
        return _unavailable("zuo_tp", "paper_lookahead", history, target_index, status)
    times, positions = history.observed(target_index)
    velocity = (positions[-1]-positions[-2])/(times[-1]-times[-2])
    target_speed = float(np.linalg.norm(velocity))
    if not np.isfinite(velocity).all() or not math.isfinite(target_speed):
        return _unavailable("zuo_tp", "paper_lookahead", history, target_index,
                            "INVALID_PREDICTION")
    horizon = zuo_lookahead_time(defender, positions[-1], defender_speed,
                                 target_speed)
    try:
        point = (predict_zuo_at(positions[-2:], history.dt, [horizon])[0]
                 if horizon > 0 else positions[-1])
    except FloatingPointError:
        return _unavailable("zuo_tp", "paper_lookahead", history, target_index,
                            "INVALID_PREDICTION")
    direction = point-defender
    magnitude = float(np.linalg.norm(direction))
    unit = _tuple(direction/magnitude) if magnitude > 0 and math.isfinite(magnitude) else None
    return InterceptPrediction("zuo_tp", "READY", float(times[-1]), horizon,
        float(times[-1]+horizon), _tuple(point), _tuple(velocity), None,
        "paper_lookahead", None, 2, unit)


def estimate_three_point_adaptive_from_history(
        history, target_index, defender_position, defender_speed, contact_distance,
        *, eta=.1, minimum_closing_speed=1., maximum_horizon_s=1.,
        epsilon_acceleration=1e-6):
    """Project-specific radial/maneuver look-ahead, not an impact-time root."""
    defender = _vector(defender_position)
    parameters = (defender_speed, contact_distance, eta, minimum_closing_speed,
                  maximum_horizon_s, epsilon_acceleration)
    if (not all(math.isfinite(value) for value in parameters) or
            defender_speed <= 0 or contact_distance <= 0 or eta <= 0 or
            minimum_closing_speed <= 0 or maximum_horizon_s < history.dt or
            epsilon_acceleration <= 0):
        raise ValueError("invalid adaptive look-ahead parameters")
    status = history.method_status(target_index, "tan_quadratic")
    if status != "READY":
        return _unavailable("three_point_adaptive", "adaptive_lookahead",
                            history, target_index, status)
    times, positions = history.observed(target_index)
    old, middle, current = positions[-3:]
    dt = history.dt
    velocity = (3*current-4*middle+old)/(2*dt)
    acceleration = (current-2*middle+old)/(dt*dt)
    relative = current-defender
    distance = float(np.linalg.norm(relative))
    v_norm, a_norm = float(np.linalg.norm(velocity)), float(np.linalg.norm(acceleration))
    if not all(math.isfinite(x) for x in (distance,v_norm,a_norm)):
        return _unavailable("three_point_adaptive", "adaptive_lookahead", history,
                            target_index, "INVALID_PREDICTION")
    if distance <= contact_distance:
        horizon, radial, effective, range_time, maneuver_time, limiter = 0.,None,None,None,None,"contact"
        point = current
    else:
        radial = float((relative/distance)@velocity)
        effective = defender_speed-radial
        range_time = (distance-contact_distance)/max(effective,minimum_closing_speed)
        maneuver_time = math.inf if a_norm <= epsilon_acceleration else eta*v_norm/a_norm
        raw = min(range_time,maneuver_time)
        horizon = float(np.clip(raw,dt,maximum_horizon_s))
        limiter = ("minimum" if raw < dt else "maximum" if raw > maximum_horizon_s
                   else "maneuver" if maneuver_time < range_time else "range")
        try:
            point = predict_at(positions[-3:],dt,[horizon])[0]
        except FloatingPointError:
            return _unavailable("three_point_adaptive", "adaptive_lookahead", history,
                                target_index, "INVALID_PREDICTION")
    if not np.isfinite(point).all() or not math.isfinite(horizon):
        return _unavailable("three_point_adaptive", "adaptive_lookahead", history,
                            target_index, "INVALID_PREDICTION")
    return InterceptPrediction("three_point_adaptive","READY",float(times[-1]),
        horizon,float(times[-1]+horizon),_tuple(point),_tuple(velocity),
        _tuple(acceleration),"adaptive_lookahead",None,3,
        range_m=distance,radial_target_speed_mps=radial,
        effective_closing_speed_mps=effective,range_horizon_s=range_time,
        maneuver_horizon_s=maneuver_time,horizon_limited_by=limiter)
