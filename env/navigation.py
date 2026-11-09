"""Independent single-leg static waypoint task using the v4 analytic motion."""
from dataclasses import asdict, dataclass
from copy import deepcopy
from types import SimpleNamespace
import math

import numpy as np

from .heading_potential import HeadingConfig, heading_transition
from .interception.config import MAX_EVADERS, MAX_PURSUERS
from .interception.events import boundary_contact, windows
from .interception.motion import advance, command, execute_velocity
from .interception.observation import observe


VERSION = "uav-static-navigation-v1.0.0"


@dataclass(frozen=True)
class NavigationConfig(HeadingConfig):
    members: int = 1
    waypoint: tuple = (9., 0., 9.)
    start: tuple = (0., 0., 9.)
    start_jitter: tuple = (.5, 2., 1.)
    initial_velocity: tuple = (.2, .8, 0.)
    lower: tuple = (-20., -20., 2.)
    upper: tuple = (28., 20., 20.)
    radius: float = 1.5
    dt: float = .1
    max_steps: int = 180
    vmax: float = 4.
    tau: float = 2.5
    climb_vmax: float = 3.
    descent_vmax: float = 2.
    horizontal_acceleration_max: float = 4.
    position_scale: float = 20.
    velocity_scale: float = 4.
    progress_scale: float = 15.
    arrival_reward: float = 2.
    safety_cost: float = 2.
    step_cost: float = .002

    def __post_init__(self):
        super().__post_init__()
        for name in ("waypoint", "start", "start_jitter", "initial_velocity", "lower", "upper"):
            value = tuple(getattr(self, name))
            if len(value) != 3 or not all(map(math.isfinite, value)):
                raise ValueError(f"invalid {name}")
            object.__setattr__(self, name, value)
        if type(self.members) is not int or not 1 <= self.members <= MAX_PURSUERS:
            raise ValueError("invalid members")
        if type(self.max_steps) is not int or self.max_steps < 1:
            raise ValueError("invalid max_steps")
        for name in ("radius", "dt", "vmax", "tau", "climb_vmax", "descent_vmax",
                     "horizontal_acceleration_max", "position_scale", "velocity_scale", "progress_scale",
                     "arrival_reward", "safety_cost", "step_cost"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0 or (value == 0 and name not in
                    ("arrival_reward", "safety_cost", "step_cost")):
                raise ValueError(f"invalid {name}")
        if min(self.start_jitter) < 0 or any(not (lo < x-j <= x+j < hi and lo+r < w < hi-r)
                for lo,hi,x,j,w,r in zip(self.lower,self.upper,self.start,self.start_jitter,self.waypoint,
                                       (self.radius,)*3)):
            raise ValueError("invalid navigation bounds")
        nearest = np.maximum(0.,np.abs(np.subtract(self.waypoint,self.start))-self.start_jitter)
        if np.linalg.norm(nearest) <= self.radius or np.linalg.norm(self.initial_velocity)>self.vmax or not (
                -self.descent_vmax <= self.initial_velocity[2] <= self.climb_vmax):
            raise ValueError("invalid navigation start")

    @property
    def center(self):
        return self.waypoint

    @property
    def num_uavs(self):
        return MAX_PURSUERS


class NavigationEnv:
    def __init__(self, config):
        self.config = config
        self.state = None

    def reset(self, seed=None):
        c = self.config
        mask = np.arange(MAX_PURSUERS)<c.members
        states = np.zeros((MAX_PURSUERS,6))
        states[mask,:3] = np.asarray(c.start)+np.random.default_rng(seed).uniform(
            -np.asarray(c.start_jitter),c.start_jitter,(c.members,3))
        states[mask,3:] = c.initial_velocity
        self.state = SimpleNamespace(pursuers=states,pursuer_mask=mask,evaders=np.zeros((MAX_EVADERS,6)),
            evader_mask=np.zeros(MAX_EVADERS,dtype=bool),time=0.,step=0,outcome="ongoing", arrivals=0)
        self.origins = states[:,:3].copy()
        return self._output()

    def _output(self):
        s = self.state
        obs, central = observe(s,self.config)
        return obs, dict(active_mask=s.pursuer_mask.copy(), centralized_state=central,
                         states=s.pursuers.copy(), origins=self.origins.copy(), time=s.time, step=s.step,
                         outcome=s.outcome, arrivals=s.arrivals)

    def step(self, actions):
        s, c = self.state, self.config
        if s is None or s.outcome != "ongoing":
            raise RuntimeError("reset required")
        actions = np.asarray(actions,dtype=float)
        mask = s.pursuer_mask
        if actions.shape != (MAX_PURSUERS,3) or not np.isfinite(actions[mask]).all() or (
                (actions[mask]<0)|(actions[mask]>1)).any():
            raise ValueError("invalid active navigation action")
        before = s.pursuers
        indices = np.flatnonzero(mask)
        u = execute_velocity(command(actions[mask],c.vmax),before[mask,3:],c.vmax,c.tau,c)
        trajectories = {i:(lambda t, i=i, control=control: advance(before[i],control,c.tau,t))
                        for i,control in zip(indices,u)}
        candidates = []
        for i,control in zip(indices,u):
            at = trajectories[i]
            bound = np.linalg.norm(control-before[i,3:])/c.tau
            arrival = windows(lambda t: at(t)[:3]-c.waypoint,bound,c.radius,0.,c.dt,1e-7)
            boundary = boundary_contact(before[i],control,c.tau,at,c.lower,c.upper,c.dt,1e-7)
            if boundary is not None:
                candidates.append((boundary,0,i))
            if arrival:
                candidates.append((arrival[0][0],1,i))
        next_mask = mask.copy()
        times = np.full(MAX_PURSUERS,c.dt)
        elapsed, outcome, count = c.dt,"ongoing",0
        for t,kind,i in sorted(candidates):
            if not next_mask[i]:
                continue
            if kind == 0:
                elapsed, outcome = t,"boundary"
                break
            times[i], next_mask[i] = t,False
            count += 1
            if not next_mask.any():
                elapsed, outcome = t,"success"
                break
        times = np.minimum(times,elapsed)
        after = before.copy()
        for i in indices:
            after[i] = trajectories[i](times[i])
        terminated = outcome != "ongoing"
        truncated = not terminated and s.step+1 >= c.max_steps
        if truncated:
            outcome = "timeout"
        parts = dict(progress=float(np.sum(np.linalg.norm(before[mask,:3]-c.waypoint,axis=1)-
                        np.linalg.norm(after[mask,:3]-c.waypoint,axis=1))/(c.members*c.progress_scale)),
                     arrival=c.arrival_reward*count, safety=-c.safety_cost if outcome=="boundary" else 0.,
                     time=-c.step_cost*elapsed/c.dt)
        heading = heading_transition(before,after,c.waypoint,mask,next_mask,c.members,terminated,c)
        if c.heading_enabled:
            parts["heading"] = heading["reward"]
        reward = np.where(mask,sum(parts.values()),0.).astype(np.float32)
        s.pursuers, s.pursuer_mask, s.time, s.step, s.outcome = after,next_mask,s.time+elapsed,s.step+1,outcome
        s.arrivals += count
        obs, info = self._output()
        info.update(reward_parts=parts,team_reward=float(reward[indices[0]]),heading=heading,
                    durations=np.where(mask,times,0.),heading_discount=c.heading_discount,
                    waypoint=c.waypoint,members=c.members)
        return obs,reward,terminated,truncated,info

    def snapshot(self):
        return deepcopy(dict(version=VERSION,config=asdict(self.config),state=self.state,origins=self.origins))

    def restore(self, snapshot):
        if snapshot["version"] != VERSION or snapshot["config"] != asdict(self.config):
            raise ValueError("navigation snapshot configuration mismatch")
        self.state,self.origins = deepcopy((snapshot["state"],snapshot["origins"]))
