"""Seven-sample causal target history, CV/CA forecasts, and finite-candidate K-FIP."""
from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from prediction.intercept_prediction import solve_constant_velocity_pip
from .dynamics import Command, DynamicsConfig, PointState, WingState, propagate


class TargetHistory:
    def __init__(self, slots=1, capacity=7):
        if type(slots) is not int or slots < 1 or capacity != 7:
            raise ValueError("seven-sample history and positive slots required")
        self.slots = slots
        self.rows = [deque(maxlen=capacity) for _ in range(slots)]
        self.retired = np.zeros(slots,bool)

    def observe(self, time, positions, mask):
        values = np.asarray(positions,float)
        active = np.asarray(mask,bool)
        if (not math.isfinite(time) or time < 0 or values.shape != (self.slots,3)
                or active.shape != (self.slots,) or not np.isfinite(values).all()):
            raise ValueError("invalid target observation")
        for j in range(self.slots):
            if not active[j]:
                if self.rows[j]:
                    self.retired[j] = True
                self.rows[j].clear()
                continue
            if self.retired[j]:
                raise ValueError("target reactivated within a scenario")
            if self.rows[j] and time <= self.rows[j][-1][0]:
                raise ValueError("target timestamps must increase")
            self.rows[j].append((float(time),values[j].copy()))


@dataclass(frozen=True)
class Forecast:
    method: str
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    horizon_limit: float
    speed_limit: float
    count: int

    def at(self, horizon):
        if not math.isfinite(horizon) or horizon < 0:
            raise ValueError("nonnegative finite forecast horizon required")
        t = min(horizon,self.horizon_limit)
        point = self.position+self.velocity*t+.5*self.acceleration*t*t
        end_velocity = self.velocity+self.acceleration*t
        return point+end_velocity*(horizon-t)


def forecast(history, index, method="CV", speed_limit=20., ca_horizon=2.):
    rows = list(history.rows[index])
    required = 3 if method == "CV" else 7 if method == "CA" else None
    if required is None or len(rows) < required:
        raise ValueError("insufficient history or unknown forecast method")
    if not math.isfinite(speed_limit) or speed_limit <= 0 or ca_horizon <= 0:
        raise ValueError("invalid forecast limits")
    time = np.array([r[0] for r in rows],float)
    positions = np.asarray([r[1] for r in rows],float)
    if method == "CV":
        if not np.allclose(np.diff(time[-3:]),np.diff(time[-3:])[0],rtol=1e-8,atol=1e-9):
            raise ValueError("CV requires evenly spaced three-point history")
        h = time[-1]-time[-2]
        velocity = (3*positions[-1]-4*positions[-2]+positions[-3])/(2*h)
        acceleration = np.zeros(3)
        limit = 0.
    else:
        relative = time-time[-1]
        design = np.column_stack((np.ones(len(rows)),relative,.5*relative**2))
        coefficients = np.linalg.lstsq(design,positions,rcond=None)[0]
        velocity = coefficients[1]
        acceleration = coefficients[2]
        limit = ca_horizon
    norm = np.linalg.norm(velocity)
    if norm > speed_limit:
        velocity = velocity*speed_limit/norm
    if method == "CA":
        end = velocity+acceleration*ca_horizon
        norm = np.linalg.norm(end)
        if norm > speed_limit:
            end *= speed_limit/norm
        acceleration = (end-velocity)/ca_horizon
    return Forecast(method,positions[-1].copy(),velocity,acceleration,limit,
                    speed_limit,len(rows))


@dataclass(frozen=True)
class FeasiblePlan:
    status: str
    reference_pip_time_s: float | None
    candidate_time_s: float
    minimum_distance_m: float
    closest_time_s: float
    predicted_target_position_m: tuple[float,float,float]
    command: Command
    candidates_evaluated: int


def feasible_intercept(state, target, dynamics, *, horizon=15., interval=.1,
                       contact_distance=.1):
    """Finite held-command search, not a proof of dynamical reachability."""
    if not isinstance(state,(PointState,WingState)) or horizon <= 0 or interval <= 0:
        raise ValueError("invalid K-FIP inputs")
    start = np.asarray(state.position,float)
    solution = solve_constant_velocity_pip(start,dynamics.v_max,target.position,target.velocity)
    t0 = solution.time_to_pip_s
    if t0 is None or t0 > horizon:
        candidates = [horizon]
        far = True
    else:
        candidates = sorted({float(np.clip(t0+offset,interval,horizon))
                             for offset in (-2.,-1.,0.,1.,2.)})
        far = False
    scored = []
    for end in candidates:
        aim = target.at(end)
        delta = aim-start
        norm = float(np.linalg.norm(delta))
        if norm <= 1e-12:
            continue
        command = Command(tuple(delta/norm),dynamics.v_max)
        # The executor resolves the curved path at 0.01 s. Use the same
        # substep size here so reaching a speed/turn limit has the same timing.
        rows = propagate(state,command,dynamics,end,min(interval,.01))
        distances = [(float(np.linalg.norm(position.position-target.at(t))),t)
                     for t,position in rows]
        distance,closest_time = min(distances)
        scored.append((distance,closest_time,end,aim,command))
    if not scored:
        raise FloatingPointError("no finite feasible-intercept candidate")
    distance,closest_time,end,aim,command = min(scored,key=lambda x:(x[0],x[1],x[2]))
    status = ("FAR_TRACK" if far else
              "CANDIDATE_CONTACT" if distance <= contact_distance else
              "NO_CONTACT_PREDICTED")
    return FeasiblePlan(status,t0,end,distance,closest_time,tuple(float(x) for x in aim),
                        command,len(scored))
