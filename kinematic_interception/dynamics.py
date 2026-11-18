"""Shared execution and prediction stepping for two simplified 3D UAV models."""
from dataclasses import dataclass
import math

import numpy as np


def _vector(value):
    result = np.asarray(value, float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("finite 3D vector required")
    return result


def _unit(value):
    vector = _vector(value)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("nonzero desired direction required")
    return vector/norm


def wrap(angle):
    return (angle+math.pi)%(2*math.pi)-math.pi


@dataclass(frozen=True)
class DynamicsConfig:
    dt: float = .05
    v_max: float = 20.
    a_max: float = 10.
    v_min: float = 12.
    yaw_rate_max: float = .5
    flightpath_rate_max: float = .3
    flightpath_max: float = math.radians(20.)
    lateral_acceleration_max: float = 10.
    acceleration_max: float = 3.
    braking_max: float = 4.

    def __post_init__(self):
        values = tuple(vars(self).values())
        if (not all(math.isfinite(x) and x > 0 for x in values) or
                self.v_min >= self.v_max or self.flightpath_max >= math.pi/2):
            raise ValueError("invalid kinematic constraints")


@dataclass(frozen=True)
class Command:
    direction: tuple[float,float,float]
    speed: float

    def validated(self, maximum):
        direction = _unit(self.direction)
        if not math.isfinite(self.speed) or not 0 <= self.speed <= maximum:
            raise ValueError("invalid desired speed")
        return direction


@dataclass
class PointState:
    position: np.ndarray
    velocity: np.ndarray

    def copy(self):
        return PointState(self.position.copy(),self.velocity.copy())


@dataclass
class WingState:
    position: np.ndarray
    speed: float
    yaw: float
    flightpath: float

    def copy(self):
        return WingState(self.position.copy(),self.speed,self.yaw,self.flightpath)


def wing_velocity(state):
    return state.speed*np.array((math.cos(state.flightpath)*math.cos(state.yaw),
                                 math.cos(state.flightpath)*math.sin(state.yaw),
                                 math.sin(state.flightpath)))


def velocity(state):
    return state.velocity.copy() if isinstance(state,PointState) else wing_velocity(state)


def step(state, command, config, dt=None):
    """Return next state, actual mean acceleration, and applied-control diagnostics."""
    h = config.dt if dt is None else dt
    if not math.isfinite(h) or h <= 0:
        raise ValueError("positive integration interval required")
    direction = command.validated(config.v_max)
    if isinstance(state,PointState):
        position, old = _vector(state.position), _vector(state.velocity)
        if np.linalg.norm(old) > config.v_max+1e-9:
            raise ValueError("initial speed exceeds limit")
        delta = command.speed*direction-old
        magnitude = float(np.linalg.norm(delta))
        executed = delta*min(1.,config.a_max*h/magnitude) if magnitude else np.zeros(3)
        acceleration = executed/h
        result = PointState(position+old*h+.5*acceleration*h*h,old+executed)
        return result, acceleration, dict(model="A",acceleration_norm=float(np.linalg.norm(acceleration)),
                                           speed=float(np.linalg.norm(result.velocity)))
    if not isinstance(state,WingState):
        raise TypeError("unknown kinematic state")
    position = _vector(state.position)
    if (not all(math.isfinite(x) for x in (state.speed,state.yaw,state.flightpath)) or
            not config.v_min <= state.speed <= config.v_max or
            abs(state.flightpath)>config.flightpath_max+1e-10):
        raise ValueError("invalid wing state")
    horizontal = math.hypot(direction[0],direction[1])
    desired_yaw = state.yaw if horizontal <= 1e-12 else math.atan2(direction[1],direction[0])
    desired_gamma = math.atan2(direction[2],horizontal)
    desired_gamma = float(np.clip(desired_gamma,-config.flightpath_max,config.flightpath_max))
    yaw_rate = float(np.clip(wrap(desired_yaw-state.yaw)/h,
                             -config.yaw_rate_max,config.yaw_rate_max))
    gamma_rate = float(np.clip((desired_gamma-state.flightpath)/h,
                               -config.flightpath_rate_max,config.flightpath_rate_max))
    # Bound the combined speed-direction acceleration at the fastest speed in the interval.
    maximum_speed = max(state.speed,min(config.v_max,command.speed))
    next_gamma = state.flightpath+gamma_rate*h
    cosine_bound = (1. if min(state.flightpath,next_gamma) <= 0 <= max(state.flightpath,next_gamma)
                    else max(math.cos(state.flightpath),math.cos(next_gamma)))
    lateral = maximum_speed*math.hypot(cosine_bound*yaw_rate,gamma_rate)
    if lateral > config.lateral_acceleration_max:
        scale = config.lateral_acceleration_max/lateral
        yaw_rate *= scale
        gamma_rate *= scale
    requested_acceleration = (command.speed-state.speed)/h
    tangent = float(np.clip(requested_acceleration,-config.braking_max,
                            config.acceleration_max))
    speed = float(np.clip(state.speed+tangent*h,config.v_min,config.v_max))
    yaw = state.yaw+yaw_rate*h
    gamma = float(np.clip(state.flightpath+gamma_rate*h,
                          -config.flightpath_max,config.flightpath_max))
    mid = WingState(position,(state.speed+speed)/2,
                    state.yaw+yaw_rate*h/2,(state.flightpath+gamma)/2)
    result = WingState(position+wing_velocity(mid)*h,speed,yaw,gamma)
    actual = (wing_velocity(result)-wing_velocity(state))/h
    return result,actual,dict(model="B",speed=speed,yaw_rate=yaw_rate,
                              flightpath_rate=gamma_rate,tangential_acceleration=tangent,
                              lateral_acceleration=maximum_speed*math.hypot(
                                  cosine_bound*yaw_rate,gamma_rate))


def propagate(state, command, config, duration, interval=.01):
    """Execute a held high-level command through identical substeps for truth and prediction."""
    if not math.isfinite(duration) or duration < 0 or interval <= 0:
        raise ValueError("invalid propagation duration")
    current = state.copy()
    rows = [(0.,current.copy())]
    elapsed = 0.
    while elapsed < duration-1e-12:
        h = min(interval,duration-elapsed)
        current,_,_ = step(current,command,config,h)
        elapsed += h
        rows.append((elapsed,current.copy()))
    return rows
