"""Causal 1v1 target-point estimates; no environment actions are chosen here."""
from dataclasses import dataclass
import math

import numpy as np



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
    pip_discriminant: float | None = None
    pip_equation_degree: int | None = None


@dataclass(frozen=True)
class PIPSolution:
    status: str
    time_to_pip_s: float | None
    predicted_interception_position_m: tuple[float, float, float] | None
    discriminant: float | None
    equation_degree: int | None


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


def solve_constant_velocity_pip(defender_position, defender_speed, target_position,
                                target_velocity):
    """Earliest positive center-coincidence time for a fixed-speed defender."""
    defender = _vector(defender_position)
    target = _vector(target_position)
    velocity = _vector(target_velocity)
    if not math.isfinite(defender_speed) or defender_speed <= 0:
        raise ValueError("defender speed must be finite and positive")
    relative = target-defender
    a = float(velocity@velocity-defender_speed**2)
    b = float(2*relative@velocity)
    c = float(relative@relative)
    if not all(map(math.isfinite,(a,b,c))):
        return PIPSolution("INVALID_PREDICTION",None,None,None,None)
    scale = max(float(velocity@velocity),defender_speed**2,1.)
    if abs(a) <= 1e-12*scale:
        degree, discriminant = 1, None
        roots = [-c/b] if abs(b)>1e-12*max(math.sqrt(c)*math.sqrt(scale),1.) else []
    else:
        degree = 2
        discriminant = b*b-4*a*c
        if not math.isfinite(discriminant):
            return PIPSolution("INVALID_PREDICTION",None,None,None,degree)
        tolerance=1e-12*max(b*b,abs(4*a*c),1.)
        if -tolerance<=discriminant<0:
            discriminant=0.
        if discriminant < 0:
            roots = []
        else:
            square = math.sqrt(discriminant)
            q = -.5*(b+math.copysign(square,b))
            roots = [q/a,c/q] if q else [-b/(2*a)]
    positive = [root for root in roots if math.isfinite(root) and root>0]
    if not positive:
        return PIPSolution("NO_POSITIVE_PIP_SOLUTION",None,None,discriminant,degree)
    horizon = min(positive)
    point = target+velocity*horizon
    if not np.isfinite(point).all():
        return PIPSolution("INVALID_PREDICTION",None,None,discriminant,degree)
    return PIPSolution("READY",horizon,_tuple(point),discriminant,degree)


def estimate_two_point_pip_from_history(history, target_index, defender_position,
                                       defender_speed):
    """Two-point finite-difference velocity with analytic constant-velocity PIP."""
    defender=_vector(defender_position)
    status=history.method_status(target_index,"two_point_pip")
    if status!="READY":
        return _unavailable("two_point_pip","constant_velocity_pip",
                            history,target_index,status)
    times,positions=history.observed(target_index)
    velocity=(positions[-1]-positions[-2])/(times[-1]-times[-2])
    if not np.isfinite(velocity).all():
        return _unavailable("two_point_pip","constant_velocity_pip",
                            history,target_index,"INVALID_PREDICTION")
    solution=solve_constant_velocity_pip(defender,defender_speed,positions[-1],velocity)
    horizon=solution.time_to_pip_s
    point=None
    if solution.status=="READY":
        try:
            point=(positions[-1][None,:]+np.asarray([horizon])[:,None]*
                   ((positions[-1]-positions[-2])/history.dt)[None,:])[0]
            if not np.isfinite(point).all():
                raise FloatingPointError("nonfinite linear prediction")
        except FloatingPointError:
            return _unavailable("two_point_pip","constant_velocity_pip",
                                history,target_index,"INVALID_PREDICTION")
        if (not np.isfinite(point).all() or np.linalg.norm(
                point-np.asarray(solution.predicted_interception_position_m))>
                1e-8*max(1.,np.linalg.norm(point))):
            return _unavailable("two_point_pip","constant_velocity_pip",
                                history,target_index,"INVALID_PREDICTION")
    return InterceptPrediction("two_point_pip",solution.status,float(times[-1]),
        horizon,float(times[-1]+horizon) if horizon is not None else None,
        _tuple(point) if point is not None else None,_tuple(velocity),None,
        "constant_velocity_pip",None,2,
        pip_discriminant=solution.discriminant,
        pip_equation_degree=solution.equation_degree)


def estimate_three_point_pip_from_history(history, target_index, defender_position,
                                          defender_speed):
    """Three observations estimate current velocity; only that velocity enters PIP."""
    defender = _vector(defender_position)
    if not math.isfinite(defender_speed) or defender_speed <= 0:
        raise ValueError("defender speed must be finite and positive")
    status = history.method_status(target_index, "three_point_pip")
    if status != "READY":
        return _unavailable("three_point_pip", "constant_velocity_pip",
                            history, target_index, status)
    times, positions = history.observed(target_index)
    old, middle, current = positions[-3:]
    dt = history.dt
    velocity = (3*current-4*middle+old)/(2*dt)
    acceleration = (current-2*middle+old)/(dt*dt)
    if not np.isfinite(velocity).all() or not np.isfinite(acceleration).all():
        return _unavailable("three_point_pip", "constant_velocity_pip", history,
                            target_index, "INVALID_PREDICTION")
    solution = solve_constant_velocity_pip(defender,defender_speed,current,velocity)
    horizon = solution.time_to_pip_s
    return InterceptPrediction("three_point_pip",solution.status,float(times[-1]),
        horizon,float(times[-1]+horizon) if horizon is not None else None,
        solution.predicted_interception_position_m,_tuple(velocity),
        _tuple(acceleration),"constant_velocity_pip",None,3,
        pip_discriminant=solution.discriminant,
        pip_equation_degree=solution.equation_degree)
