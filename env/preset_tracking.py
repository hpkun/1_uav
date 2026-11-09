"""Read-only reference following using the existing analytic flight model."""
from dataclasses import asdict

import numpy as np

from .interception.events import boundary_contact
from .interception.motion import advance, execute_velocity, project
from .preset_trajectory import VERSION, initial_states, make_paths


def _samples(paths,time):
    positions, velocities, accelerations, modes = zip(*(path.sample(time) for path in paths))
    return np.asarray(positions), np.asarray(velocities), np.asarray(accelerations), modes


def _frame(paths,state,time):
    p,v,a,modes = _samples(paths,time)
    speed = np.linalg.norm(v,axis=1)
    direction = np.divide(v,speed[:,None],out=np.zeros_like(v),where=speed[:,None]>0)
    return dict(time=time,reference_positions=p,reference_velocities=v,reference_accelerations=a,
                reference_directions=direction,reference_modes=modes,actual_states=state,
                position_errors=np.linalg.norm(p-state[:,:3],axis=1),
                velocity_errors=np.linalg.norm(v-state[:,3:],axis=1))


def tracking_request(paths, state, time, dt, config, positions, velocities):
    _, v, a, _ = _samples(paths, time+dt/2)
    return (v+config.tau*a+config.position_gain*(positions-state[:, :3])+
            config.velocity_gain*(velocities-state[:, 3:]))


def _commands(paths,frame,dt,config):
    c = config
    state = frame["actual_states"]
    request = tracking_request(paths, state, frame["time"], dt, c,
                               frame["reference_positions"], frame["reference_velocities"])
    actual = execute_velocity(request,state[:,3:],c.vmax,c.tau,c)
    ball = project(request/c.vmax,c.vmax)
    vertical = (ball[:,2]>c.climb_vmax+1e-12)|(ball[:,2]<-c.descent_vmax-1e-12)
    ball[:,2] = np.clip(ball[:,2],-c.descent_vmax,c.climb_vmax)
    flags = dict(speed=np.linalg.norm(request,axis=1)>c.vmax+1e-12,vertical=vertical,
                 horizontal=np.linalg.norm(ball[:,:2]-state[:,3:5],axis=1)>c.tau*c.horizontal_acceleration_max+1e-12)
    return request,actual,(actual-state[:,3:])/c.tau,flags


def run_tracking(kind,seed,config,*,trace=True):
    state = initial_states(config,seed)
    initial = state.copy()
    paths = make_paths(kind,state,config,seed)
    c = config
    requested_horizon = max(path.duration for path in paths)+c.hold_seconds
    horizon = min(requested_horizon,c.max_steps*c.dt)
    low,high = c.flight_bounds
    outcome, failure, boundary_id = "completed", None, None
    rows = [_frame(paths,state,0.)]
    try:
        with np.errstate(over="raise",invalid="raise",divide="raise"):
            for step in range(c.max_steps):
                frame = rows[-1]
                end = min(horizon,(step+1)*c.dt)
                dt = end-frame["time"]
                request,actual,acceleration,flags = _commands(paths,frame,dt,c)
                for i,(start,control) in enumerate(zip(state,actual)):
                    at = lambda t: advance(start,control,c.tau,t)
                    hit = boundary_contact(start,control,c.tau,at,low,high,dt,c.event_time_tolerance)
                    if hit is not None:
                        dt,boundary_id = hit,i
                state = advance(state,actual,c.tau,dt)
                if not np.isfinite(state).all():
                    raise FloatingPointError("nonfinite physical state")
                frame.update(requested_velocities=request,executed_velocities=actual,
                             actual_accelerations=acceleration,constraints=flags,command_duration=dt)
                if boundary_id is not None:
                    rows.append(_frame(paths,state,frame["time"]+dt))
                    outcome = "boundary"
                    break
                rows.append(_frame(paths,state,end))
                if end == horizon:
                    if horizon < requested_horizon:
                        outcome = "time_limit"
                    break
    except FloatingPointError as exc:
        outcome, failure = "numerical_failure", str(exc)

    positions = np.array([row["actual_states"][:,:3] for row in rows])
    speeds = np.linalg.norm([row["actual_states"][:,3:] for row in rows],axis=2)
    errors = np.array([row["position_errors"] for row in rows])
    velocity_errors = np.array([row["velocity_errors"] for row in rows])
    applied = [row for row in rows if "actual_accelerations" in row]
    horizontal_acceleration = (np.linalg.norm([r["actual_accelerations"][:,:2] for r in applied],axis=2)
                               if applied else np.zeros((1,len(paths))))
    duration = rows[-1]["time"]
    members = []
    for i,path in enumerate(paths):
        end_error = float(np.linalg.norm(positions[-1,i]-path.points[-1]))
        members.append(dict(
            id=i, reference=asdict(path), reference_seed=[seed,1,i] if kind=="bezier" else None,
            actual_length=float(np.linalg.norm(np.diff(positions[:,i],axis=0),axis=1).sum()),
            mean_error=float(errors[:,i].mean()),max_error=float(errors[:,i].max()),
            error_percentiles=dict(zip(("p50","p90","p99"),np.percentile(errors[:,i],[50,90,99]).tolist())),
            mean_velocity_error=float(velocity_errors[:,i].mean()),max_speed=float(speeds[:,i].max()),
            max_horizontal_acceleration=float(horizontal_acceleration[:,i].max()),
            min_vertical_speed=float(min(r["actual_states"][i,5] for r in rows)),
            max_vertical_speed=float(max(r["actual_states"][i,5] for r in rows)),
            constraint_steps={k:sum(bool(r["constraints"][k][i]) for r in applied)
                              for k in ("speed","vertical","horizontal")},
            reference_completed=duration>=path.duration,endpoint_error=end_error,
            settled=duration>=path.duration and end_error<=c.position_tolerance and speeds[-1,i]<=c.speed_tolerance))
    result = dict(version=VERSION,kind=kind,seed=seed,initial_seed=[seed,0],initial_states=initial,
                  active_mask=[True]*len(paths),duration=duration,outcome=outcome,boundary_id=boundary_id,
                  numerical_failure=failure,members=members,config=asdict(c))
    if trace:
        result["frames"] = rows
    return result
