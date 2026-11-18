"""Direct controls used identically by MPC prediction and environment execution."""
from dataclasses import dataclass
import math

import numpy as np

from ..dynamics import DynamicsConfig, PointState, WingState, wing_velocity, wrap


@dataclass(frozen=True)
class AControl:
    acceleration: tuple[float, float, float]


@dataclass(frozen=True)
class BControl:
    tangential_acceleration: float
    yaw_rate: float
    flightpath_rate: float


def direct_step(state, control, config: DynamicsConfig, dt: float):
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("positive finite integration duration required")
    if isinstance(state, PointState) and isinstance(control, AControl):
        a = np.asarray(control.acceleration,float)
        if a.shape != (3,) or not np.isfinite(a).all():
            raise ValueError("invalid A acceleration")
        norm = float(np.linalg.norm(a))
        if norm > config.a_max+1e-6:
            raise ValueError("A acceleration limit exceeded")
        if norm > config.a_max:
            a = a*config.a_max/norm
        new_v = state.velocity+a*dt
        if np.linalg.norm(new_v) > config.v_max+1e-6:
            raise ValueError("A speed limit exceeded")
        return PointState(state.position+state.velocity*dt+.5*a*dt*dt,new_v)
    if isinstance(state, WingState) and isinstance(control, BControl):
        a,yaw_rate,gamma_rate = (control.tangential_acceleration,
                                 control.yaw_rate,control.flightpath_rate)
        if not all(math.isfinite(v) for v in (a,yaw_rate,gamma_rate)):
            raise ValueError("invalid B control")
        if (not -config.braking_max-1e-6 <= a <= config.acceleration_max+1e-6
                or abs(yaw_rate)>config.yaw_rate_max+1e-6
                or abs(gamma_rate)>config.flightpath_rate_max+1e-6):
            raise ValueError("B input limit exceeded")
        a = float(np.clip(a,-config.braking_max,config.acceleration_max))
        yaw_rate = float(np.clip(yaw_rate,-config.yaw_rate_max,config.yaw_rate_max))
        gamma_rate = float(np.clip(gamma_rate,-config.flightpath_rate_max,
                                   config.flightpath_rate_max))
        next_speed = float(np.clip(state.speed+a*dt,config.v_min,config.v_max))
        next_gamma = float(np.clip(state.flightpath+gamma_rate*dt,
                                   -config.flightpath_max,config.flightpath_max))
        actual_gamma_rate = (next_gamma-state.flightpath)/dt
        cosine = (1. if min(state.flightpath,next_gamma)<=0<=max(state.flightpath,next_gamma)
                  else max(math.cos(state.flightpath),math.cos(next_gamma)))
        lateral = max(state.speed,next_speed)*math.hypot(cosine*yaw_rate,actual_gamma_rate)
        if lateral>config.lateral_acceleration_max+1e-6:
            raise ValueError("B lateral acceleration limit exceeded")
        if lateral>config.lateral_acceleration_max:
            factor=config.lateral_acceleration_max/lateral
            yaw_rate*=factor
            actual_gamma_rate*=factor
            next_gamma=state.flightpath+actual_gamma_rate*dt
        next_yaw=wrap(state.yaw+yaw_rate*dt)
        mid=WingState(state.position,(state.speed+next_speed)/2,
                      state.yaw+yaw_rate*dt/2,(state.flightpath+next_gamma)/2)
        return WingState(state.position+wing_velocity(mid)*dt,
                         next_speed,next_yaw,next_gamma)
    raise TypeError("direct control does not match A/B state")
