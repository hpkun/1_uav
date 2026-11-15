"""Causal 1v1 target point estimates; these functions never choose UAV actions."""
from dataclasses import dataclass
import math

import numpy as np

from .intercept_time import estimate_constant_velocity_contact, zuo_lookahead_time
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
    """Zuo equations (30)-(32): two-point velocity, paper horizon, linear point."""
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


def estimate_three_point_contact_from_history(history, target_index,
                                              defender_position, defender_speed,
                                              contact_distance, max_horizon_s=None):
    """First 3D sphere contact using the local three-point quadratic target."""
    defender = _vector(defender_position)
    status = history.method_status(target_index, "tan_quadratic")
    if status != "READY":
        return _unavailable("three_point_kinematic", "kinematic_contact",
                            history, target_index, status)
    times, positions = history.observed(target_index)
    if (not math.isfinite(defender_speed) or defender_speed <= 0 or
            not math.isfinite(contact_distance) or contact_distance <= 0 or
            max_horizon_s is not None and
            (not math.isfinite(max_horizon_s) or max_horizon_s <= 0)):
        raise ValueError("invalid contact geometry or horizon")
    dt = history.dt
    old, middle, current = positions[-3:]
    velocity = (3*current-4*middle+old)/(2*dt)
    acceleration = (current-2*middle+old)/(dt*dt)
    if not np.isfinite(velocity).all() or not np.isfinite(acceleration).all():
        return _unavailable("three_point_kinematic", "kinematic_contact", history,
                            target_index, "INVALID_PREDICTION")
    q = acceleration/2
    r = current-defender
    distance = float(np.linalg.norm(r))
    if distance <= contact_distance:
        tau, residual, result_status = 0., 0., "CONTACT_POSSIBLE"
    elif np.linalg.norm(q) <= 64*np.finfo(float).eps*max(
            1., np.linalg.norm(velocity)/dt, defender_speed/dt):
        # Solve unbounded first so no solution differs from a root beyond the horizon.
        constant = estimate_constant_velocity_contact(defender,defender_speed,current,
            velocity,contact_distance,float(times[-1]))
        tau = constant.time_to_contact_s
        result_status = ("CONTACT_POSSIBLE" if tau is not None else
                         "NO_QUADRATIC_CONTACT_SOLUTION")
        if tau is not None and max_horizon_s is not None and tau > max_horizon_s:
            tau, result_status = None, "NO_CONTACT_IN_HORIZON"
        residual = (abs(np.linalg.norm(r+velocity*tau)-defender_speed*tau-contact_distance)
                    if tau is not None else None)
    else:
        coefficients = np.array([q@q, 2*velocity@q,
            velocity@velocity+2*r@q-defender_speed**2,
            2*r@velocity-2*defender_speed*contact_distance,
            r@r-contact_distance**2],dtype=np.float64)
        if not np.isfinite(coefficients).all():
            result_status, tau, residual = "INVALID_PREDICTION", None, None
        else:
            scale_time = max(dt, distance/(defender_speed+np.linalg.norm(velocity)+1e-12))
            powers = scale_time**np.arange(4,-1,-1)
            scaled = coefficients*powers
            size = float(np.max(np.abs(scaled)))
            if not np.isfinite(scaled).all() or size == 0:
                result_status, tau, residual = "INVALID_PREDICTION", None, None
            else:
                scaled /= size
                while len(scaled)>2 and abs(scaled[0]) <= 32*np.finfo(float).eps:
                    scaled = scaled[1:]
                try:
                    roots = np.roots(scaled)
                except np.linalg.LinAlgError:
                    return _unavailable("three_point_kinematic", "kinematic_contact",
                                        history, target_index, "INVALID_PREDICTION")
                valid = []
                for root in roots:
                    if (not np.isfinite(root) or abs(root.imag) >
                            1e-8*max(1.,abs(root.real))):
                        continue
                    candidate = float(root.real*scale_time)
                    if candidate < 0 or not math.isfinite(candidate):
                        continue
                    deviation = abs(float(np.linalg.norm(r+velocity*candidate+q*candidate**2)
                                          -(defender_speed*candidate+contact_distance)))
                    tolerance = 1e-9*max(1.,distance,defender_speed*candidate+contact_distance)
                    if math.isfinite(deviation) and deviation <= tolerance:
                        valid.append((candidate,deviation))
                eligible = [(t,e) for t,e in valid if max_horizon_s is None or
                            t <= max_horizon_s+1e-10]
                if eligible:
                    tau,residual = min(eligible)
                    result_status = "CONTACT_POSSIBLE"
                else:
                    tau,residual = None,None
                    result_status = ("NO_CONTACT_IN_HORIZON" if valid and max_horizon_s is not None
                                     else "NO_QUADRATIC_CONTACT_SOLUTION")
    try:
        point = None if tau is None else (predict_at(positions[-3:],dt,[tau])[0]
                                           if tau > 0 else current)
    except FloatingPointError:
        return _unavailable("three_point_kinematic", "kinematic_contact", history,
                            target_index, "INVALID_PREDICTION")
    return InterceptPrediction("three_point_kinematic",result_status,float(times[-1]),
        tau,float(times[-1]+tau) if tau is not None else None,
        _tuple(point) if point is not None else None,_tuple(velocity),_tuple(acceleration),
        "kinematic_contact",float(residual) if residual is not None else None,3)
