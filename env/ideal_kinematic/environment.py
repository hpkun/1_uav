"""Independent ideal three-dimensional point-mass multi-agent environment."""
from copy import deepcopy
import numpy as np

from .config import IdealConfig, OBS_DIM, STATE_DIM, SLOTS, TARGET_SLOTS
from .geometry import boundary, closest, direction, proximity
from .reference import Reference


def fixed_assignment(states, available_tasks, task_status):
    """Assign active UAVs without controlling their actions."""
    count = len(available_tasks)
    return {i: i % count for i in range(len(states)) if task_status[i] == "active"}


class IdealEnv:
    def __init__(self, config: IdealConfig):
        self.config = config
        self.outcome = "uninitialized"

    def reset(self, seed=None):
        c = self.config
        sequence = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(c.seed if seed is None else seed)
        self.seed = int.from_bytes(sequence.generate_state(4, dtype="<u4").tobytes(), "little")
        rng = np.random.default_rng(sequence)
        self.states = np.zeros((SLOTS, 6), dtype=float)
        self.active = np.arange(SLOTS) < c.active_count
        self.status = ["active" if flag else "unused" for flag in self.active]
        self.states[:c.active_count, :3] = np.asarray(c.starts[:c.active_count])
        if c.initialization == "random":
            self.states[:c.active_count, :3] += rng.uniform(-c.random_range, c.random_range, (c.active_count, 3))
        directions = np.asarray(c.directions[:c.active_count], float)
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        self.states[:c.active_count, 3:] = directions * np.asarray(c.speeds[:c.active_count])[:, None]
        self.references = [Reference(c.target_starts[i], c.target_ends[i], c.target_speed, c.target_mode)
                           for i in range(c.target_count)]
        self.assignment = fixed_assignment(self.states[:c.active_count],
                            list(range(c.target_count if c.target_count else c.active_count)), self.status[:c.active_count])
        self.time, self.steps, self.outcome = 0., 0, "ongoing"
        self.minimum_separation = None
        self.warning_pairs = set()
        return self.output()

    def target(self, i, time):
        if self.config.target_count:
            return self.references[self.assignment[i]].sample(time)
        return np.asarray(self.config.waypoints[i], float), np.zeros(3)

    def output(self, events=(), parts=None, team_reward=0.):
        c = self.config
        task = np.eye(4, dtype=float)[int(c.task[-1])]
        remaining = max(0., 1. - self.time / (c.dt*c.max_steps))
        reference = np.zeros((TARGET_SLOTS, 6))
        for j, path in enumerate(self.references):
            reference[j, :3], reference[j, 3:] = path.sample(self.time)
        obs = np.zeros((SLOTS, OBS_DIM), dtype=np.float32)
        for i in np.flatnonzero(self.active):
            own = self.states[i]
            target_pos, target_vel = self.target(i, self.time)
            peers = [np.r_[(self.states[j, :3]-own[:3])/c.position_scale,
                            (self.states[j, 3:]-own[3:])/c.speed_scale, float(self.active[j])]
                     for j in range(SLOTS) if j != i]
            targets = [np.r_[(reference[j, :3]-own[:3])/c.position_scale,
                              (reference[j, 3:]-own[3:])/c.speed_scale, 1.]
                       if j < c.target_count else np.zeros(7) for j in range(TARGET_SLOTS)]
            obs[i] = np.concatenate((own[:3]/c.position_scale, own[3:]/c.speed_scale,
                          (target_pos-own[:3])/c.position_scale, (target_vel-own[3:])/c.speed_scale,
                          *peers, *targets, task, [remaining], self.active.astype(float)))
        central = np.concatenate((
            *[np.r_[self.states[i, :3]/c.position_scale, self.states[i, 3:]/c.speed_scale,
                    float(self.active[i]), self.assignment.get(i, -1)/max(1, c.target_count)]
              for i in range(SLOTS)],
            *[np.r_[reference[j, :3]/c.position_scale, reference[j, 3:]/c.speed_scale,
                    float(j < c.target_count)] for j in range(TARGET_SLOTS)],
            task, [remaining])).astype(np.float32)
        if obs.shape != (SLOTS, OBS_DIM) or central.shape != (STATE_DIM,) or not np.isfinite(obs).all() or not np.isfinite(central).all():
            raise FloatingPointError("invalid ideal observations")
        info = dict(active_mask=self.active.copy(), centralized_state=central,
                    actual_states=self.states.copy(), reference=reference.copy(),
                    assignment=self.assignment.copy(), status=self.status.copy(),
                    time=self.time, step=self.steps, outcome=self.outcome,
                    events=list(events), reward_parts=dict(parts or {}),
                    team_reward=team_reward, minimum_separation=self.minimum_separation,
                    seed=self.seed, task=c.task, action_semantics="desired_motion_direction")
        return obs, info

    def step(self, actions):
        if self.outcome != "ongoing":
            raise RuntimeError("reset required")
        c = self.config
        actions = np.asarray(actions, float)
        if actions.shape != (SLOTS, 3):
            raise ValueError("invalid action shape")
        if not np.isfinite(actions[self.active]).all() or np.any((actions[self.active] < 0) | (actions[self.active] > 1)):
            raise ValueError("active Beta actions must be finite and within [0,1]")
        start = self.states.copy()
        velocities = np.zeros((SLOTS, 3), float)
        for i in np.flatnonzero(self.active):
            previous = start[i, 3:] / c.speeds[i]
            velocities[i] = direction(actions[i], previous) * c.speeds[i]
        departures = {}
        events = []
        priority = {"boundary": 0, "contact": 1, "warning": 2, "arrival": 3}
        def event(kind, tau, members, positions):
            events.append(dict(kind=kind, time=self.time + tau, members=list(members),
                               positions=[np.asarray(p).tolist() for p in positions]))
        for i in np.flatnonzero(self.active):
            tau = boundary(start[i, :3], velocities[i], np.asarray(c.lower), np.asarray(c.upper), c.dt)
            if tau is not None:
                departures[i] = (tau, "boundary")
            if not c.target_count:
                target, _ = self.target(i, self.time)
                arrival = proximity(start[i, :3], velocities[i], target, np.zeros(3),
                                    c.arrival_distance, c.dt)
                if arrival is not None and (i not in departures or arrival < departures[i][0]):
                    departures[i] = (arrival, "arrival")
        for i, (tau, kind) in departures.items():
            event(kind, tau, [i], [start[i, :3] + tau*velocities[i]])
        active_ids = np.flatnonzero(self.active)
        minimum = self.minimum_separation
        for k, i in enumerate(active_ids):
            for j in active_ids[k+1:]:
                duration = min(c.dt, departures.get(i, (c.dt,))[0], departures.get(j, (c.dt,))[0])
                distance = closest(start[i, :3], velocities[i], start[j, :3], velocities[j], duration)
                minimum = distance if minimum is None else min(minimum, distance)
                for kind, radius in (("contact", c.contact_distance), ("warning", c.warning_distance)):
                    tau = proximity(start[i, :3], velocities[i], start[j, :3], velocities[j], radius, duration)
                    if tau is not None and (kind != "warning" or (int(i), int(j)) not in self.warning_pairs):
                        event(kind, tau, [int(i), int(j)],
                              [start[i, :3]+tau*velocities[i], start[j, :3]+tau*velocities[j]])
                        if kind == "warning":
                            self.warning_pairs.add((int(i), int(j)))
        for i in active_ids:
            if not c.target_count:
                continue
            j = self.assignment[int(i)]
            path = self.references[j]
            duration = min(c.dt, departures.get(i, (c.dt,))[0])
            begin, end = self.time, self.time + duration
            knots = path.distance / path.speed
            cuts = [begin] + [float(t) for t in knots if begin < t < end] + [end]
            for left, right in zip(cuts[:-1], cuts[1:]):
                reference_position, reference_velocity = path.sample(left)
                tau = proximity(start[i, :3] + (left-begin)*velocities[i], velocities[i],
                                reference_position, reference_velocity, c.contact_distance,
                                right-left)
                if tau is not None:
                    absolute = left-begin+tau
                    event("target_contact", absolute, [int(i), SLOTS+j],
                          [start[i, :3]+absolute*velocities[i],
                           reference_position+tau*reference_velocity])
                    break
        self.minimum_separation = minimum
        priority["target_contact"] = 1
        events.sort(key=lambda item: (item["time"], priority[item["kind"]], item["members"]))
        before = {i: np.linalg.norm(self.target(i, self.time)[0]-start[i, :3]) for i in active_ids}
        for i in active_ids:
            tau = departures.get(i, (c.dt,))[0]
            self.states[i, :3] = start[i, :3] + tau*velocities[i]
            if i in departures:
                self.active[i] = False
                self.status[i] = "completed" if departures[i][1] == "arrival" else "boundary"
                self.states[i, 3:] = 0.
            else:
                self.states[i, 3:] = velocities[i]
        self.time += c.dt
        self.steps += 1
        # Moving-reference tasks require all members to track inside a non-contact band at the horizon.
        tracking_errors = [np.linalg.norm(self.target(i, self.time)[0]-self.states[i, :3])
                           for i in range(c.active_count)] if c.target_count else []
        if any(s == "boundary" for s in self.status):
            self.outcome = "boundary"
        elif any(event["kind"] == "target_contact" for event in events):
            self.outcome = "contact"
        elif not c.target_count and all(s == "completed" for s in self.status[:c.active_count]):
            self.outcome = "success"
        elif self.steps >= c.max_steps:
            self.outcome = ("success" if c.target_count and all(
                c.contact_distance < error <= c.tracking_distance for error in tracking_errors)
                else "time_limit")
        parts = dict(progress=c.progress_weight * sum(
            before[i] - np.linalg.norm(self.target(i, self.time)[0]-self.states[i, :3]) for i in active_ids)
            if not c.target_count else 0.,
            tracking=-c.tracking_weight * sum(tracking_errors)/len(tracking_errors) * c.dt
            if c.target_count else 0.,
            completion=c.completion_weight if self.outcome == "success" else 0.,
            danger=-c.danger_weight * sum(e["kind"] == "warning" for e in events),
            boundary=-c.boundary_weight if self.outcome == "boundary" else 0.,
            time=-c.time_weight*c.dt)
        reward = float(sum(parts.values()))
        obs, info = self.output(events, parts, reward)
        return obs, np.where(np.asarray([i in active_ids for i in range(SLOTS)]),
                             reward/max(len(active_ids), 1), 0.).astype(np.float32), (
                             self.outcome not in ("ongoing", "time_limit")), self.outcome == "time_limit", info

    def snapshot(self):
        return deepcopy(self.__dict__)

    def restore(self, state):
        self.__dict__.update(deepcopy(state))
