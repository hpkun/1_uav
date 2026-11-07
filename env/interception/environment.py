from copy import deepcopy
from dataclasses import asdict, dataclass
from itertools import combinations
from functools import lru_cache
import numpy as np

from .config import InterceptionConfig, VERSION, MAX_PURSUERS, MAX_EVADERS, OBS_DIM, STATE_DIM
from .events import windows, intersection, minimum_distance, boundary_contact
from .motion import advance, command
from .observation import observe
from .reward import reward_parts
from .scenario import generate, evader_commands


class SimulationFailure(FloatingPointError):
    pass


@dataclass
class InterceptionState:
    pursuers: np.ndarray
    evaders: np.ndarray
    pursuer_mask: np.ndarray
    evader_mask: np.ndarray
    dwell: np.ndarray
    phases: np.ndarray
    step: int
    time: float
    events: list
    outcome: str
    spawn_rejections: dict
    minimum_separation: float
    minimum_surface_clearance: float
    near_pending: dict
    participation: np.ndarray
    initialization: str


class InterceptionEnv:
    version = VERSION
    observation_dim = OBS_DIM
    state_dim = STATE_DIM

    def __init__(self, config=None):
        self.config = config or InterceptionConfig()
        self.rng = np.random.default_rng()
        self.state = None

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}
        p, e, failures = generate(self.config, self.rng, options)
        pursuers = np.zeros((MAX_PURSUERS, 6))
        evaders = np.zeros((MAX_EVADERS, 6))
        pursuers[:len(p)], evaders[:len(e)] = p, e
        points = np.vstack((p, e))[:, :3]
        i, j = np.triu_indices(len(points), 1)
        minimum = float(np.linalg.norm(points[i] - points[j], axis=1).min())
        sizes = np.repeat([self.config.pursuer_collision_radius, self.config.evader_collision_radius],
                          [len(p), len(e)])
        clearance = float((np.linalg.norm(points[i]-points[j], axis=1)-sizes[i]-sizes[j]).min())
        phases = (np.zeros(MAX_EVADERS) if options.get("scenario") == "fixed"
                  else self.rng.uniform(0., 2 * np.pi, MAX_EVADERS))
        self.state = InterceptionState(pursuers, evaders, np.arange(MAX_PURSUERS) < len(p),
                                       np.arange(MAX_EVADERS) < len(e),
                                       np.zeros((MAX_PURSUERS, MAX_EVADERS)),
                                       phases,
                                       0, 0., [], "ongoing", failures, minimum, clearance, {},
                                       np.zeros(MAX_PURSUERS), options.get("scenario", "random"))
        return self._output([], {}, 0.)

    def _output(self, events, parts, effort):
        s = self.state
        obs, central = observe(s, self.config)
        return obs, dict(centralized_state=central, active_mask=s.pursuer_mask.copy(),
                         evader_mask=s.evader_mask.copy(), pursuers=s.pursuers.copy(),
                         evaders=s.evaders.copy(), dwell_seconds=s.dwell.copy(),
                         dwell_steps=s.dwell.copy() / self.config.dt, events=deepcopy(events),
                         outcome=s.outcome, step=s.step, time=s.time, reward_parts=parts,
                         control_effort=effort, minimum_separation=s.minimum_separation,
                         minimum_center_distance=s.minimum_separation,
                         minimum_surface_clearance=s.minimum_surface_clearance,
                         participation_seconds=s.participation.copy(), task=self.config.task,
                         initialization=s.initialization,
                         spawn_rejections=s.spawn_rejections.copy())

    def step(self, actions):
        s, c = self.state, self.config
        if s is None or s.outcome != "ongoing":
            raise RuntimeError("reset required")
        actions = np.asarray(actions, dtype=float)
        if actions.shape != (MAX_PURSUERS, 3):
            raise ValueError("expected actions with shape [5,3]")
        active = actions[s.pursuer_mask]
        if not np.isfinite(active).all():
            raise SimulationFailure(f"simulation_failure at step {s.step}: nonfinite command")
        if np.any((active < 0) | (active > 1)):
            raise ValueError("active actions must be in [0,1]")
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                return self._step(s, c, active)
        except (FloatingPointError, OverflowError) as exc:
            raise SimulationFailure(f"simulation_failure at step {s.step}: {exc}") from exc

    def _step(self, s, c, active):
        states = np.vstack((s.pursuers, s.evaders))
        valid = np.flatnonzero(np.r_[s.pursuer_mask, s.evader_mask])
        targets = np.zeros((len(states), 3))
        targets[np.flatnonzero(s.pursuer_mask)] = command(active, c.pursuer_vmax)
        targets[MAX_PURSUERS:] = evader_commands(s.evaders, s.evader_mask,
                                                s.pursuers[s.pursuer_mask], c, s.phases, s.time)
        taus = np.r_[np.full(MAX_PURSUERS, c.pursuer_tau), np.full(MAX_EVADERS, c.evader_tau)]
        acceleration = (targets - states[:, 3:]) / taus[:, None]
        acc = np.linalg.norm(acceleration, axis=1)
        radii = np.r_[np.full(MAX_PURSUERS, c.pursuer_collision_radius),
                      np.full(MAX_EVADERS, c.evader_collision_radius)]

        @lru_cache(maxsize=None)
        def at(i, time):
            return advance(states[i], targets[i], taus[i], time)

        def threshold(curve, bound, radius, **kwargs):
            return windows(curve, bound, radius, 0., c.dt, c.event_time_tolerance, **kwargs)

        def relative(i, j, segment):
            return lambda t: at(i, t)[segment] - at(j, t)[segment]

        position, velocity = slice(0, 3), slice(3, 6)
        candidates, pairs = [], []
        qualified = {}
        dwell = np.zeros_like(s.dwell)
        for i, j in combinations(valid.tolist(), 2):
            curve = relative(i, j, position)
            shared_tau = taus[i] == taus[j]
            bound = (np.linalg.norm(acceleration[i] - acceleration[j]) if shared_tau
                     else acc[i] + acc[j])
            radius = radii[i]+radii[j]
            pairs.append((i, j, curve, bound, radius))
            collision = threshold(curve, bound, radius)
            if collision:
                candidates.append((collision[0][0], 0, i, j, "collision"))
            if i < MAX_PURSUERS <= j:
                target = j - MAX_PURSUERS
                near = threshold(curve, bound, c.capture_outer_radius)
                if near:
                    near = list(intersection(near, threshold(curve, bound, c.capture_inner_radius,
                                                             outside=True)))
                jerk = bound / taus[i] if shared_tau else acc[i] / taus[i] + acc[j] / taus[j]
                slow = threshold(relative(i, j, velocity), jerk,
                                 c.capture_relative_speed_max) if near else []
                qualified[i, j] = list(intersection(near, slow))
                for start, end in qualified[i, j]:
                    previous = s.dwell[i, target] if start <= 1e-13 else 0.
                    reached = start + max(0., c.capture_dwell_seconds - previous)
                    if reached <= end + 1e-13:
                        candidates.append((min(reached, end), 3, i, j, "capture"))
                    if end >= c.dt - 1e-13:
                        dwell[i, target] = previous + end - start
        center = np.array(c.center)
        for j in valid[valid >= MAX_PURSUERS]:
            breach = threshold(lambda t: at(j, t)[:3] - center, acc[j], c.radius)
            if breach:
                candidates.append((breach[0][0], 1, int(j), int(j), "breach"))
        for i in valid:
            contact = boundary_contact(states[i], targets[i], taus[i], lambda t: at(i, t),
                                       np.asarray(c.lower)+radii[i], np.asarray(c.upper)-radii[i],
                                       c.dt, c.event_time_tolerance)
            if contact is not None:
                kind = "pursuer_boundary_violation" if i < MAX_PURSUERS else "evader_escape"
                candidates.append((contact, 2, int(i), int(i), kind))

        def event_row(kind, i, j, time):
            first, second = at(i, time), at(j, time)
            ids = [int(i), int(j)]
            types = ["pursuer" if k < MAX_PURSUERS else "evader" for k in ids]
            return dict(kind=kind, time=s.time+time, step=s.step+1, entities=ids,
                        entity_ids=ids, entity_types=types, positions=[first[:3].tolist(), second[:3].tolist()],
                        velocities=[first[3:].tolist(), second[3:].tolist()],
                        center_distance=float(np.linalg.norm(first[:3]-second[:3])) if i != j else None,
                        collision_threshold=float(radii[i]+radii[j]) if i != j else None)

        alive = s.evader_mask.copy()
        stop_times = np.full(len(states), c.dt)
        elapsed, outcome, events = c.dt, "ongoing", []
        candidates.sort()
        while candidates:
            candidates = [event for event in candidates if all(
                alive[k - MAX_PURSUERS] for k in event[2:4] if k >= MAX_PURSUERS
            )]
            if not candidates:
                break
            earliest = candidates[0][0]
            simultaneous = [event for event in candidates if event[0] <= earliest + c.event_time_tolerance]
            event = min(simultaneous, key=lambda row: (row[1], row[0], row[2], row[3]))
            candidates.remove(event)
            time, _, i, j, kind = event
            events.append(event_row(kind, i, j, time))
            if kind == "capture":
                alive[j - MAX_PURSUERS] = False
                stop_times[j] = time
                dwell[:, j - MAX_PURSUERS] = 0.
                if alive.any():
                    continue
                outcome = "success"
            else:
                outcome = kind
            elapsed = time
            break
        stop_times = np.minimum(stop_times, elapsed)
        if outcome != "ongoing":
            dwell.fill(0.)
            for (i, j), intervals in qualified.items():
                if alive[j-MAX_PURSUERS]:
                    for start, stop in intervals:
                        if start <= elapsed <= stop+1e-13:
                            previous = s.dwell[i, j-MAX_PURSUERS] if start <= 1e-13 else 0.
                            dwell[i, j-MAX_PURSUERS] = previous + max(0., elapsed-start)
        next_states = states.copy()
        for i in valid:
            next_states[i] = at(i, stop_times[i])
        if not np.isfinite(next_states).all():
            raise FloatingPointError("nonfinite motion")
        minimum, clearance = s.minimum_separation, s.minimum_surface_clearance
        pending = {}
        collision_pair = next((tuple(event["entities"]) for event in events
                               if event["kind"] == "collision"), None)
        closing_episode = outcome != "ongoing" or s.step+1 >= c.max_steps
        for i, j, curve, bound, radius in pairs:
            end = min(stop_times[i], stop_times[j])
            closest = minimum_distance(curve, bound, end)
            minimum, clearance = min(minimum, closest), min(clearance, closest-radius)
            if c.near_miss_clearance == 0 or closest > radius+c.near_miss_clearance+.001:
                continue
            key = (i, j)
            intervals = windows(curve, bound, radius+c.near_miss_clearance, 0., end,
                                c.event_time_tolerance)
            for start, stop in intervals:
                entry = s.near_pending.get(key) if start <= 1e-13 else None
                entry = entry or event_row("near_miss", i, j, start)
                if key == collision_pair and stop >= end-1e-13:
                    continue
                if stop < end-1e-13 or closing_episode or not all(
                    alive[k-MAX_PURSUERS] for k in key if k >= MAX_PURSUERS
                ):
                    events.append(entry)
                else:
                    pending[key] = entry
        participation = s.participation.copy()
        for (i, j), intervals in qualified.items():
            end = min(stop_times[i], stop_times[j])
            participation[i] += sum(max(0., min(stop, end)-start) for start, stop in intervals)
        events.sort(key=lambda event: event["time"])
        # Progress uses the same set of targets at both ends. A captured
        # target is compared at its capture endpoint with pursuers at that time.
        initial_distances, final_distances = [], []
        pursuers = np.flatnonzero(s.pursuer_mask)
        for j in valid[valid >= MAX_PURSUERS]:
            initial_distances.append(np.linalg.norm(states[pursuers, :3] - states[j, :3], axis=1).min())
            end = stop_times[j]
            final_distances.append(min(np.linalg.norm(at(i, end)[:3] - next_states[j, :3])
                                       for i in pursuers))
        parts = reward_parts(initial_distances, final_distances, int((s.evader_mask & ~alive).sum()),
                             outcome, elapsed, c)
        terminated = outcome != "ongoing"
        truncated = not terminated and s.step + 1 >= c.max_steps
        if truncated:
            outcome = "timeout"
        effort = float(elapsed * np.square(targets[:MAX_PURSUERS]).sum() / s.pursuer_mask.sum())
        rewards = np.where(s.pursuer_mask, sum(parts.values()), 0.).astype(np.float32)
        if not np.isfinite(rewards).all():
            raise FloatingPointError("nonfinite reward")
        successor = InterceptionState(next_states[:MAX_PURSUERS], next_states[MAX_PURSUERS:],
                                      s.pursuer_mask.copy(), alive, dwell, s.phases.copy(),
                                      s.step + 1, s.time + elapsed, s.events + events, outcome,
                                      s.spawn_rejections.copy(), minimum, clearance, pending, participation,
                                      s.initialization)
        self.state = successor
        try:
            obs, info = self._output(events, parts, effort)
        except FloatingPointError:
            self.state = s
            raise
        info["commands"] = targets
        return obs, rewards, terminated, truncated, info

    def snapshot(self):
        return deepcopy(dict(version=VERSION, config=asdict(self.config),
                             state=self.state, rng=self.rng.bit_generator.state))

    def restore(self, snapshot):
        if snapshot.get("version") != VERSION or snapshot.get("config") != asdict(self.config):
            raise ValueError("incompatible interception snapshot")
        snapshot = deepcopy(snapshot)
        self.rng.bit_generator.state = snapshot["rng"]
        self.state = snapshot["state"]
