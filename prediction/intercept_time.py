"""1v1 ideal-model encounter times; no actions or environment state changes."""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class InterceptTimeEstimate:
    status: str
    time_to_contact_s: float | None
    predicted_contact_time_s: float | None
    predicted_target_position_m: tuple[float, float, float] | None
    model: str


def zuo_lookahead_time(defender_position, target_position, defender_speed,
                       target_speed, *, relative_speed_floor=15.0):
    """Paper's distance-based prediction horizon, not a physical contact time."""
    defender = _vector(defender_position)
    target = _vector(target_position)
    if (not all(math.isfinite(x) for x in (defender_speed,target_speed,relative_speed_floor))
            or defender_speed <= 0 or target_speed < 0 or relative_speed_floor <= 0):
        raise ValueError("invalid speed or lookahead floor")
    return float(np.linalg.norm(target-defender)/max(defender_speed-.2*target_speed,
                                                     relative_speed_floor))


def estimate_constant_velocity_contact(defender_position, defender_speed,
                                       target_position, target_velocity,
                                       contact_distance, observation_time=0.,
                                       max_horizon_s=None):
    """Earliest geometric contact under a constant-speed reachable ray.

    The target keeps its observed velocity. The defender may choose one
    direction instantly, then travels straight at its fixed speed. The
    estimate is recomputed from current observations; it is not an observed
    event or a promise that a later controller will follow that ray.
    """
    defender = _vector(defender_position)
    target = _vector(target_position)
    velocity = _vector(target_velocity)
    values = (defender_speed, contact_distance, observation_time)
    if (not all(math.isfinite(x) for x in values) or defender_speed <= 0
            or contact_distance <= 0 or observation_time < 0
            or max_horizon_s is not None and
            (not math.isfinite(max_horizon_s) or max_horizon_s <= 0)):
        raise ValueError("invalid speed, contact distance or time")
    relative = target-defender
    distance = float(np.linalg.norm(relative))
    if distance <= contact_distance:
        tau = 0.
    else:
        # ||r + v*t|| = speed*t + contact_distance.
        a = float(velocity@velocity-defender_speed**2)
        b = float(2*(relative@velocity-defender_speed*contact_distance))
        c = float(relative@relative-contact_distance**2)
        if abs(a) <= 1e-12*max(1.,defender_speed**2,float(velocity@velocity)):
            roots = [-c/b] if b < 0 else []
        else:
            discriminant = b*b-4*a*c
            if discriminant < 0:
                roots = []
            else:
                root = math.sqrt(max(0.,discriminant))
                roots = [(-b-root)/(2*a),(-b+root)/(2*a)]
        candidates = [t for t in roots if math.isfinite(t) and t >= 0 and
                      (max_horizon_s is None or t <= max_horizon_s+1e-10)]
        tau = min(candidates) if candidates else None
    if tau is None or max_horizon_s is not None and tau > max_horizon_s+1e-10:
        return InterceptTimeEstimate("NO_CONTACT_IN_HORIZON" if max_horizon_s is not None
            else "NO_CONSTANT_VELOCITY_SOLUTION",None,None,None,"constant_velocity_1v1")
    position = target+velocity*tau
    return InterceptTimeEstimate("CONTACT_POSSIBLE",float(tau),
        float(observation_time+tau),tuple(float(x) for x in position),
        "constant_velocity_1v1")


def estimate_from_history(history, target_index, defender_position,
                          defender_speed, contact_distance, max_horizon_s=None):
    """Causal adapter using the same last-two-position estimate as Zuo."""
    status = history.method_status(target_index, "zuo_linear")
    if status != "READY":
        return InterceptTimeEstimate(status,None,None,None,"constant_velocity_1v1")
    times, positions = history.observed(target_index)
    velocity = (positions[-1]-positions[-2])/(times[-1]-times[-2])
    return estimate_constant_velocity_contact(defender_position,defender_speed,
        positions[-1],velocity,contact_distance,float(times[-1]),max_horizon_s)


def _vector(value):
    result = np.asarray(value,dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("position or velocity must be a finite three-vector")
    return result
