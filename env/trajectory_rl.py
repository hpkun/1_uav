"""Non-contact multi-UAV tracking of immutable preset references."""
from copy import deepcopy
from dataclasses import asdict, dataclass
import math

import numpy as np

from .interception.config import MAX_PURSUERS
from .interception.events import boundary_contact
from .interception.motion import advance, command, execute_velocity, project
from .preset_trajectory import MODES, TrajectoryConfig, initial_states, make_paths


VERSION = "uav-preset-tracking-mappo-v1.0.0"
SLOT = 18
OBS_DIM = STATE_DIM = MAX_PURSUERS*SLOT+1


class TrackingNumericalFailure(FloatingPointError):
    pass


@dataclass(frozen=True)
class TrackingConfig(TrajectoryConfig):
    starts: tuple = ((-22.,-12.,15.),(-22.,0.,24.))
    endpoints: tuple = ((22.,12.,23.),(22.,24.,32.))
    initial_velocities: tuple = ((.4,.2,0.),(.4,.2,0.))
    trajectory_mode: str = "straight"
    observation_position_scale: float = 50.
    position_error_scale: float = 10.
    position_weight: float = 1.
    velocity_weight: float = .25
    completion_weight: float = 5.
    missed_weight: float = 2.
    boundary_weight: float = 10.
    time_weight: float = .002

    def __post_init__(self):
        super().__post_init__()
        if self.trajectory_mode not in MODES or len(self.starts)>MAX_PURSUERS:
            raise ValueError("invalid trajectory mode or slot count")
        for key in ("observation_position_scale","position_error_scale","position_weight","velocity_weight",
                    "completion_weight","missed_weight","boundary_weight","time_weight"):
            v = getattr(self,key)
            if not math.isfinite(v) or v < 0 or (key.endswith("_scale") and v==0):
                raise ValueError(f"invalid {key}")

    @property
    def num_uavs(self):
        return MAX_PURSUERS


class TrackingEnv:
    def __init__(self, config):
        self.config = config
        self.outcome = "uninitialized"

    def reset(self, seed=None):
        c = self.config
        sequence = seed if isinstance(seed,np.random.SeedSequence) else np.random.SeedSequence(seed)
        self.seed = int.from_bytes(sequence.generate_state(4,dtype="<u4").tobytes(),"little")
        initial = initial_states(c,self.seed)
        self.paths = make_paths(c.trajectory_mode,initial,c,self.seed)
        self.deadlines = np.asarray([p.duration+c.hold_seconds for p in self.paths])
        self.actual = np.zeros((MAX_PURSUERS,6))
        self.actual[:len(initial)] = initial
        self.active = np.arange(MAX_PURSUERS)<len(initial)
        self.status = ["active"]*len(initial)
        self.time, self.steps, self.outcome = 0.,0,"ongoing"
        return self.output()

    def output(self):
        c = self.config
        slots = np.zeros((MAX_PURSUERS,SLOT),dtype=np.float32)
        reference = np.zeros((MAX_PURSUERS,9))
        task_duration = c.max_steps*c.dt
        for i,path in enumerate(self.paths):
            p,v,a,_ = path.sample(self.time)
            reference[i] = np.r_[p,v,a]
            if self.active[i]:
                actual = self.actual[i]
                slots[i] = np.r_[actual[:3]/c.observation_position_scale,actual[3:]/c.vmax,
                    (p-actual[:3])/c.observation_position_scale,v/c.vmax,a/c.horizontal_acceleration_max,
                    max(0.,path.duration-self.time)/task_duration,
                    max(0.,self.deadlines[i]-self.time)/task_duration,1.]
        remaining = np.float32(max(0.,1-self.time/task_duration))
        obs = np.zeros((MAX_PURSUERS,OBS_DIM),dtype=np.float32)
        for i in np.flatnonzero(self.active):
            obs[i] = np.r_[slots[i],slots[np.arange(MAX_PURSUERS)!=i].ravel(),remaining]
        central = np.r_[slots.ravel(),remaining]
        if not np.isfinite(obs).all() or not np.isfinite(central).all():
            raise TrackingNumericalFailure("nonfinite tracking observations")
        return obs, dict(active_mask=self.active.copy(),centralized_state=central,
            actual_states=self.actual.copy(),reference=reference,time=self.time,step=self.steps,
            outcome=self.outcome,status=self.status.copy(),seed=self.seed,
            reference_paths=[asdict(path) for path in self.paths] if self.steps==0 else None)

    def step(self, actions):
        if self.outcome != "ongoing":
            raise RuntimeError("reset required")
        actions = np.asarray(actions,dtype=float)
        if actions.shape!=(MAX_PURSUERS,3):
            raise ValueError("invalid tracking action dimensions")
        valid = actions[self.active]
        if not np.isfinite(valid).all():
            raise TrackingNumericalFailure("nonfinite tracking action")
        if ((valid<0)|(valid>1)).any():
            raise ValueError("active actions must be within [0,1]")
        try:
            with np.errstate(over="raise",invalid="raise",divide="raise"):
                return self._step(valid)
        except FloatingPointError as exc:
            raise TrackingNumericalFailure(f"numerical_failure at step {self.steps}: {exc}") from exc

    def _step(self, valid):
        c = self.config
        start = self.time
        before = self.actual
        ids = np.flatnonzero(self.active)
        cap = c.max_steps*c.dt
        end = min((self.steps+1)*c.dt,float(self.deadlines[ids].max()))
        request = command(valid,c.vmax)
        executed = execute_velocity(request,before[ids,3:],c.vmax,c.tau,c)
        vertical = (request[:,2]>c.climb_vmax)|(request[:,2]<-c.descent_vmax)
        horizontal = np.linalg.norm(request[:,:2]-before[ids,3:5],axis=1)>c.tau*c.horizontal_acceleration_max+1e-12
        # The existing action mapping projects the Beta cube into the velocity ball.
        speed = np.linalg.norm(2*valid-1,axis=1)>1+1e-12
        trajectories = {i:(lambda t,i=i,u=u: advance(before[i],u,c.tau,t)) for i,u in zip(ids,executed)}
        low,high = c.flight_bounds
        boundary = None
        for i,u in zip(ids,executed):
            stop = min(end,self.deadlines[i])
            hit = boundary_contact(before[i],u,c.tau,trajectories[i],low,high,stop-start,c.event_time_tolerance)
            if hit is not None and (boundary is None or start+hit < end-c.event_time_tolerance):
                end,boundary = start+hit,int(i)
        successor = before.copy()
        status = self.status.copy()
        next_mask = self.active.copy()
        rows = []
        parts = dict(position=0.,velocity=0.,completion=0.,missed=0.,boundary=0.,
                     time=-c.time_weight*(end-start)/c.dt)
        for k,i in enumerate(ids):
            path = self.paths[i]
            stop = min(end,self.deadlines[i])
            dt = stop-start
            state = successor[i] = trajectories[i](dt)
            p,v,_,_ = path.sample(stop)
            error = float(np.linalg.norm(p-state[:3]))
            ve = float(np.linalg.norm(v-state[3:]))
            weight = dt/(c.dt*len(self.paths))
            parts["position"] -= c.position_weight*error/c.position_error_scale*weight
            parts["velocity"] -= c.velocity_weight*ve/c.vmax*weight
            at_reference_end = (float(np.linalg.norm(trajectories[i](path.duration-start)[:3]-path.points[-1]))
                                if start<path.duration<=stop else None)
            if stop >= self.deadlines[i] and (boundary is None or stop < end):
                success = error<=c.position_tolerance and ve<=c.speed_tolerance
                status[i] = "completed" if success else "missed"
                next_mask[i] = False
                if success:
                    parts["completion"] += c.completion_weight/len(self.paths)
                else:
                    parts["missed"] -= c.missed_weight/len(self.paths)
            rows.append(dict(id=int(i),duration=float(dt),phase_duration=float(max(0.,min(stop,path.duration)-start)),
                position_error=error,velocity_error=ve,reference_end_error=at_reference_end,
                endpoint_error=float(np.linalg.norm(state[:3]-path.points[-1])),
                distance=float(np.linalg.norm(state[:3]-before[i,:3])),completion_time=float(stop) if status[i]=="completed" else None,
                speed_projection=int(speed[k]),vertical_limit=int(vertical[k]),horizontal_limit=int(horizontal[k])))
        if not np.isfinite(successor).all() or not np.isfinite(list(parts.values())).all():
            raise TrackingNumericalFailure("nonfinite state or tracking reward")
        outcome = "ongoing"
        if boundary is not None:
            outcome = "boundary"
            parts["boundary"] = -c.boundary_weight
            status = ["boundary" if i==boundary else ("aborted" if value=="active" else value)
                      for i,value in enumerate(status)]
            next_mask[:] = False
        elif not next_mask.any():
            outcome = "success" if all(x=="completed" for x in status) else "precision_failure"
        elif end >= cap:
            outcome = "time_limit"
        terminated = outcome in ("success","precision_failure","boundary")
        truncated = outcome == "time_limit"
        reward = np.where(self.active,sum(parts.values()),0.).astype(np.float32)
        if not np.isfinite(reward).all():
            raise TrackingNumericalFailure("nonfinite tracking reward")
        self.actual,self.active,self.status,self.time,self.outcome = successor,next_mask,status,end,outcome
        self.steps += 1
        obs,info = self.output()
        info.update(samples=rows,reward_parts=parts,team_reward=float(reward[ids[0]]),boundary_id=boundary)
        return obs,reward,terminated,truncated,info

    def snapshot(self):
        return deepcopy(dict(version=VERSION,config=asdict(self.config),
                             state={k:v for k,v in vars(self).items() if k!="config"}))

    def restore(self, snapshot):
        if snapshot["version"]!=VERSION or snapshot["config"]!=asdict(self.config):
            raise ValueError("tracking snapshot configuration mismatch")
        self.__dict__.update(deepcopy(snapshot["state"]))
