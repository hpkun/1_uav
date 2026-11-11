"""Continuous-time event resolution for ideal point-mass episodes."""
import numpy as np

from .config import SLOTS
from .geometry import boundary, closest, direction, inside_intervals, merge_intervals, proximity, subtract_intervals

_NODES16, _WEIGHTS16 = np.polynomial.legendre.leggauss(16)
_NODES32, _WEIGHTS32 = np.polynomial.legendre.leggauss(32)
PRIORITY = {"boundary": 0, "contact": 1, "target_contact": 2,
            "success": 3, "deadline": 4, "warning": 5,
            "safety_violation": 6, "arrival": 7}


def _error_integral(relative, velocity, duration):
    def integrate(nodes, weights, length, origin):
        times = origin + (nodes+1)*length/2
        distances = np.linalg.norm(relative[None, :] + times[:, None]*velocity[None, :], axis=1)
        return length/2 * float(weights@distances)
    def adaptive(origin, length, depth):
        a = integrate(_NODES16, _WEIGHTS16, length, origin)
        b = integrate(_NODES32, _WEIGHTS32, length, origin)
        if abs(a-b) <= 1e-9*max(1., b):
            return b
        if depth >= 10:
            raise FloatingPointError("tracking quadrature did not converge")
        return adaptive(origin, length/2, depth+1)+adaptive(origin+length/2, length/2, depth+1)
    return adaptive(0., duration, 0)


def step(env, actions):
    if env.outcome != "ongoing":
        raise RuntimeError("reset required")
    c = env.config
    actions = np.asarray(actions, float)
    if actions.shape != (SLOTS, 3):
        raise ValueError("invalid action shape")
    active = np.flatnonzero(env.active)
    if not np.isfinite(actions[active]).all() or np.any((actions[active] < 0) | (actions[active] > 1)):
        raise ValueError("active Beta actions must be finite and within [0,1]")
    start = env.states.copy()
    velocities = np.zeros((SLOTS, 3))
    for i in active:
        velocities[i] = direction(actions[i], start[i, 3:]/c.speeds[i])*c.speeds[i]

    departures = {}
    terminals = []
    for i in active:
        edge = boundary(start[i, :3], velocities[i], np.asarray(c.lower), np.asarray(c.upper), c.dt)
        if edge is not None:
            departures[int(i)] = (edge, "boundary")
        if not c.target_count:
            target, _ = env.target(int(i), env.time)
            arrival = proximity(start[i, :3], velocities[i], target, np.zeros(3), c.arrival_distance, c.dt)
            if arrival is not None and (i not in departures or arrival < departures[i][0]):
                departures[int(i)] = (arrival, "arrival")
        if i in departures and departures[i][1] == "boundary":
            terminals.append((departures[i][0], "boundary", (int(i),)))
    for index, i in enumerate(active):
        for j in active[index+1:]:
            duration = min(c.dt, departures.get(i, (c.dt,))[0], departures.get(j, (c.dt,))[0])
            hit = proximity(start[i, :3], velocities[i], start[j, :3], velocities[j],
                            c.contact_distance, duration)
            if hit is not None:
                terminals.append((hit, "contact", (int(i), int(j))))
    for i in active:
        duration = min(c.dt, departures.get(i, (c.dt,))[0])
        for j, path in enumerate(env.references):
            for left, right, target, target_velocity in path.segments(env.time, env.time+duration):
                offset = left-env.time
                hit = proximity(start[i, :3]+offset*velocities[i], velocities[i],
                                target, target_velocity, c.contact_distance, right-left)
                if hit is not None:
                    terminals.append((offset+hit, "target_contact", (int(i), SLOTS+j)))
                    break
    if not c.target_count and all(
            env.status[i] == "completed" or
            (i in departures and departures[i][1] == "arrival")
            for i in range(c.active_count)):
        newest = [departures[i][0] for i in range(c.active_count)
                  if env.status[i] != "completed"]
        terminals.append((max(newest, default=0.), "success", ()))
    if env.steps+1 >= c.max_steps:
        terminals.append((c.dt, "deadline", ()))
    terminal = min(terminals, key=lambda row: (row[0], PRIORITY[row[1]], row[2])) if terminals else None
    duration = terminal[0] if terminal else c.dt
    absolute_stop = env.time+duration
    events = []
    def event(kind, tau, members):
        positions = []
        for member in members:
            if member < SLOTS:
                positions.append((start[member, :3]+velocities[member]*tau).tolist())
            else:
                positions.append(env.references[member-SLOTS].sample(env.time+tau)[0].tolist())
        events.append(dict(kind=kind, time=env.time+tau, members=list(members), positions=positions))

    for i, (tau, kind) in departures.items():
        if tau <= duration+1e-12:
            event(kind, tau, (i,))
    if terminal and terminal[1] in ("contact", "target_contact"):
        event(terminal[1], duration, terminal[2])
    # Each pair has an entry/exit state; intervals also catch a full crossing within one step.
    for index, i in enumerate(active):
        for j in active[index+1:]:
            pair = (int(i), int(j))
            limit = min(duration, departures.get(i, (duration,))[0], departures.get(j, (duration,))[0])
            closest_distance = closest(start[i, :3], velocities[i], start[j, :3], velocities[j], limit)
            env.minimum_separation = (closest_distance if env.minimum_separation is None
                                      else min(env.minimum_separation, closest_distance))
            for kind, radius, state in (("warning", c.warning_distance, env.warning_pairs),
                                        ("safety_violation", c.safety_distance, env.safety_pairs)):
                intervals = inside_intervals(start[i, :3], velocities[i], start[j, :3],
                                             velocities[j], radius, limit)
                inside = pair in state
                for left, right in intervals:
                    if not inside:
                        event(kind, left, pair)
                        if kind == "warning":
                            env.warning_count += 1
                        else:
                            env.safety_count += 1
                    inside = right >= limit-1e-12 and all(
                        departures.get(member, (float("inf"),))[0] > limit+1e-12 for member in pair)
                    if kind == "safety_violation":
                        env.safety_duration += right-left
                if inside:
                    state.add(pair)
                else:
                    state.discard(pair)
    before = {int(i): np.linalg.norm(env.target(int(i), env.time)[0]-start[i, :3]) for i in active}
    for i in active:
        travel = min(duration, departures.get(i, (duration,))[0])
        env.states[i, :3] = start[i, :3]+travel*velocities[i]
        env.distance_traveled[i] += c.speeds[i]*travel
        if i in departures and departures[i][0] <= duration+1e-12:
            env.active[i] = False
            env.status[i] = "completed" if departures[i][1] == "arrival" else "boundary"
            env.states[i, 3:] = 0.
        else:
            env.states[i, 3:] = velocities[i]
    tracking_integral = 0.
    if c.target_count:
        for i in active:
            path = env.references[env.assignment[int(i)]]
            for left, right, target, target_velocity in path.segments(env.time, absolute_stop):
                offset, span = left-env.time, right-left
                own = start[i, :3]+offset*velocities[i]
                relative, relative_velocity = own-target, velocities[i]-target_velocity
                inside = inside_intervals(own, velocities[i], target, target_velocity,
                                          c.tracking_distance, span)
                contact = inside_intervals(own, velocities[i], target, target_velocity,
                                           c.contact_distance, span)
                env.tracking_intervals[i].extend((left+a, left+b) for a,b in subtract_intervals(inside, contact))
                integral = _error_integral(relative, relative_velocity, span)
                env.tracking_error_integral[i] += integral
                tracking_integral += integral
                env.tracking_samples[i].append((
                    float(np.linalg.norm(relative+relative_velocity*span/2)), span))
        for j, path in enumerate(env.references):
            env.reference_motion_time[j] += sum(right-left for left,right,_,velocity in
                path.segments(env.time, absolute_stop) if np.linalg.norm(velocity) > 0)
    env.time = absolute_stop
    env.steps += 1
    if terminal:
        if terminal[1] == "deadline":
            if c.target_count:
                fractions = env.tracking_metrics()
                env.outcome = ("success" if all(row["valid_fraction"] >= c.tracking_success_fraction
                                                for row in fractions) else "tracking_failure")
            else:
                env.outcome = "deadline_failure"
        else:
            env.outcome = terminal[1]
    else:
        env.outcome = "ongoing"
    events = [item for item in events if item["time"] <= env.time+1e-12]
    events.sort(key=lambda item: (item["time"], PRIORITY[item["kind"]], item["members"]))
    env.contact_count += sum(item["kind"] in ("contact", "target_contact") for item in events)
    parts = dict(progress=c.progress_weight*sum(
        before[i]-np.linalg.norm(env.target(i, env.time)[0]-env.states[i, :3]) for i in active)
        if not c.target_count else 0.,
        tracking=-c.tracking_weight*tracking_integral/max(len(active),1) if c.target_count else 0.,
        completion=c.completion_weight if env.outcome == "success" else 0.,
        danger=-c.danger_weight*sum(item["kind"] == "warning" for item in events),
        boundary=-c.boundary_weight if env.outcome == "boundary" else 0.,
        contact_failure=-c.boundary_weight if env.outcome in ("contact", "target_contact") else 0.,
        time=-c.time_weight*duration)
    reward = float(sum(parts.values()))
    obs, info = env.output(events, parts, reward)
    rewards = np.zeros(SLOTS, np.float32)
    rewards[active] = reward/max(len(active), 1)
    return obs, rewards, env.outcome != "ongoing", False, info
