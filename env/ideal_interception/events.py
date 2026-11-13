"""Continuous candidates and deterministic bilateral event state machine."""
import numpy as np

from .geometry import boundary, closest, direction, inside_intervals, proximity, region_entry

PRIORITY = {"region_breach":0, "friendly_collision":1, "interception_contact":2,
            "intruder_collision":3, "boundary_exit":4, "path_complete":5}


def _path_pieces(path, start, end):
    return list(path.segments(start,end))


def _moving_contact(path_a, path_b, start, end, radius):
    cuts = {start,end}
    for path in (path_a,path_b):
        for left,right,_,_ in path.segments(start,end):
            cuts.update((left,right))
    ordered = sorted(cuts)
    for left,right in zip(ordered[:-1],ordered[1:]):
        a,va = path_a.sample(left)
        b,vb = path_b.sample(left)
        tau = proximity(a,va,b,vb,radius,right-left)
        if tau is not None:
            return left+tau
    return None


def candidates(env, velocities, duration):
    c = env.config
    start = env.time
    result = []
    def add(kind, tau, members):
        result.append((float(tau),PRIORITY[kind],tuple(members),kind))
    defenders = np.flatnonzero(env.defender_mask)
    intruders = np.flatnonzero(env.intruder_mask)
    for i in defenders:
        tau = boundary(env.defenders[i,:3],velocities[i],np.asarray(c.lower),
                       np.asarray(c.upper),duration)
        if tau is not None:
            add("boundary_exit",tau,(int(i),))
    for a,i in enumerate(defenders):
        for j in defenders[a+1:]:
            p,q = env.defenders[i,:3],env.defenders[j,:3]
            v,w = velocities[i],velocities[j]
            intervals = inside_intervals(p,v,q,w,c.contact_distance,duration)
            if intervals:
                add("friendly_collision",intervals[0][0],(int(i),int(j)))
    for j in intruders:
        path = env.paths[j]
        if start <= path.duration <= start+duration:
            add("path_complete",path.duration-start,(5+int(j),))
        for left,right,p,v in _path_pieces(path,start,start+duration):
            tau = region_entry(p,v,np.asarray(c.protected_center),
                               c.protected_radius,c.protected_height,right-left)
            if tau is not None:
                add("region_breach",left-start+tau,(5+int(j),))
                break
        for left,right,p,v in _path_pieces(path,start,start+duration):
            tau = boundary(p,v,np.asarray(c.lower),np.asarray(c.upper),right-left)
            if tau is not None:
                add("boundary_exit",left-start+tau,(5+int(j),))
                break
        for i in defenders:
            for left,right,p,v in _path_pieces(path,start,start+duration):
                offset = left-start
                own = env.defenders[i,:3]+offset*velocities[i]
                tau = proximity(own,velocities[i],p,v,c.contact_distance,right-left)
                if tau is not None:
                    add("interception_contact",offset+tau,(int(i),5+int(j)))
                    break
    for a,j in enumerate(intruders):
        for k in intruders[a+1:]:
            hit = _moving_contact(env.paths[j],env.paths[k],start,start+duration,
                                  c.contact_distance)
            if hit is not None:
                add("intruder_collision",hit-start,(5+int(j),5+int(k)))
    return sorted(result)


def advance(env, actions):
    c = env.config
    actions = np.asarray(actions,float)
    if env.outcome != "ongoing":
        raise RuntimeError("reset required")
    if actions.shape != (5,3):
        raise ValueError("invalid defender action shape")
    active = np.flatnonzero(env.defender_mask)
    if not np.isfinite(actions[active]).all() or np.any((actions[active]<0)|(actions[active]>1)):
        raise ValueError("active actions must be finite and within [0,1]")
    velocities = np.zeros((5,3))
    for i in active:
        velocities[i] = direction(actions[i],env.defenders[i,3:]/c.defender_speeds[i])*c.defender_speeds[i]
        env.defenders[i,3:] = velocities[i]
    origin, cursor = env.time, 0.
    start_mask = env.defender_mask.copy()
    events = []
    def emit(kind, members=()):
        positions = []
        states = []
        for member in members:
            value = env.defenders[member] if member < 5 else env.intruders[member-5]
            positions.append(value[:3].tolist())
            states.append(value.tolist())
        events.append(dict(kind=kind,time=env.time,members=list(members),
                           positions=positions,states=states))
    def move_to(tau):
        nonlocal cursor
        elapsed = tau-cursor
        if elapsed < -1e-10:
            raise FloatingPointError("event times reversed")
        if elapsed <= 0:
            return
        ids = np.flatnonzero(env.defender_mask)
        for a,i in enumerate(ids):
            for j in ids[a+1:]:
                value = closest(env.defenders[i,:3],velocities[i],
                                env.defenders[j,:3],velocities[j],elapsed)
                env.minimum_separation = value if env.minimum_separation is None else min(env.minimum_separation,value)
        env.defenders[ids,:3] += velocities[ids]*elapsed
        for j in np.flatnonzero(env.intruder_mask):
            env.intruders[j,:3],env.intruders[j,3:] = env.paths[j].sample(origin+tau)
        env.time = origin+tau
        cursor = tau
    def inactive(member,status):
        if member < 5:
            env.defender_mask[member] = False
            env.defender_status[member] = status
            env.defenders[member,3:] = 0.
        else:
            j = member-5
            env.intruder_mask[j] = False
            env.intruder_status[j] = status
            env.intruders[j,3:] = 0.
        emit("member_inactive",(member,))
    for tau,_,members,kind in candidates(env,velocities,c.dt):
        if any(not (env.defender_mask[m] if m < 5 else env.intruder_mask[m-5])
               for m in members):
            continue
        move_to(tau)
        emit(kind,members)
        if kind == "interception_contact":
            env.interception_count += 1
            inactive(members[0],"contact_exit")
            inactive(members[1],"intercepted")
            if not env.intruder_mask.any():
                env.outcome = "all_intercepted"
            elif not env.defender_mask.any():
                env.outcome = "defenders_exhausted"
        elif kind == "region_breach":
            env.breach_count += 1
            inactive(members[0],"breached")
            env.outcome = "breach"
        elif kind == "friendly_collision":
            env.friendly_collision_count += 1
            for member in members:
                inactive(member,"friendly_contact_exit")
            env.outcome = "friendly_collision"
        elif kind == "intruder_collision":
            env.intruder_collision_count += 1
            for member in members:
                inactive(member,"intruder_contact_exit")
            env.outcome = "intruder_collision"
        elif kind == "boundary_exit":
            inactive(members[0],"boundary_exit")
            env.outcome = "defender_boundary" if members[0] < 5 else "intruder_escape"
        elif kind == "path_complete":
            inactive(members[0],"path_complete")
            env.outcome = "intruder_path_complete"
        if env.outcome != "ongoing":
            emit("episode_end")
            break
    if env.outcome == "ongoing":
        move_to(c.dt)
        if env.step_count+1 >= c.max_steps:
            env.outcome = "deadline_failure"
            emit("episode_end")
    env.step_count += 1
    for j in np.flatnonzero(env.intruder_mask):
        if not np.isclose(np.linalg.norm(env.intruders[j,3:]),c.intruder_speeds[j],rtol=1e-9,atol=1e-9):
            raise FloatingPointError("active intruder speed differs from configured speed")
    for i in np.flatnonzero(env.defender_mask):
        if not np.isclose(np.linalg.norm(env.defenders[i,3:]),c.defender_speeds[i],rtol=1e-9,atol=1e-9):
            raise FloatingPointError("active defender speed differs from configured speed")
    duration = env.time-origin
    parts = dict(interception=c.interception_reward*sum(e["kind"]=="interception_contact" for e in events),
                 protection=c.protection_reward if env.outcome=="all_intercepted" else 0.,
                 breach=-c.failure_penalty if env.outcome=="breach" else 0.,
                 same_side_contact=-c.failure_penalty if env.outcome in ("friendly_collision","intruder_collision") else 0.,
                 boundary=-c.failure_penalty if env.outcome in ("defender_boundary","intruder_escape") else 0.,
                 exhausted=-c.failure_penalty if env.outcome=="defenders_exhausted" else 0.,
                 path_complete=-c.failure_penalty if env.outcome=="intruder_path_complete" else 0.,
                 deadline=-c.failure_penalty if env.outcome=="deadline_failure" else 0.,
                 time=-c.time_penalty*duration)
    reward = float(sum(parts.values()))
    env.events.extend(events)
    obs,info = env.output(events,parts,reward)
    rewards = np.zeros(5,np.float32)
    rewards[start_mask] = reward/max(int(start_mask.sum()),1)
    return obs,rewards,env.outcome!="ongoing",False,info
