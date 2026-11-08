from copy import deepcopy
from dataclasses import asdict, dataclass
from itertools import combinations
from functools import lru_cache
import numpy as np

from .config import InterceptionConfig, VERSION, MAX_PURSUERS, MAX_EVADERS, OBS_DIM, STATE_DIM, SLOT, matches_saved_config
from .events import windows, minimum_distance, boundary_contact
from .motion import advance, command, execute_velocity
from .observation import observe
from .reward import reward_parts
from .scenario import generate, evader_commands
from .protected_area import intrusion_windows, navigation_point


class SimulationFailure(FloatingPointError):
    pass


@dataclass
class InterceptionState:
    pursuers: np.ndarray
    evaders: np.ndarray
    pursuer_mask: np.ndarray
    evader_mask: np.ndarray
    evader_origins: np.ndarray
    phases: np.ndarray
    step: int
    time: float
    events: list
    outcome: str
    spawn_rejections: dict
    minimum_separation: float
    minimum_surface_clearance: float
    warning_pairs: set
    reset_seed: object
    initialization: str


class InterceptionEnv:
    version = VERSION
    observation_dim = OBS_DIM
    state_dim = STATE_DIM

    def __init__(self, config=None, *, diagnostics=False):
        self.config = config or InterceptionConfig()
        self.diagnostics = diagnostics
        extra = (self.config.evader_slots-MAX_EVADERS)*SLOT
        self.observation_dim = OBS_DIM+extra
        self.state_dim = STATE_DIM+extra
        self.rng = np.random.default_rng()
        self.state = None

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}
        p, e, failures = generate(self.config, self.rng, options)
        pursuers = np.zeros((MAX_PURSUERS, 6))
        slots = self.config.evader_slots
        evaders = np.zeros((slots, 6))
        pursuers[:len(p)], evaders[:len(e)] = p, e
        points = np.vstack((p, e))[:, :3]
        i, j = np.triu_indices(len(points), 1)
        minimum = float(np.linalg.norm(points[i] - points[j], axis=1).min())
        sizes = np.repeat([self.config.pursuer_collision_radius, self.config.evader_collision_radius],
                          [len(p), len(e)])
        clearance = float((np.linalg.norm(points[i]-points[j], axis=1)-sizes[i]-sizes[j]).min())
        phases = (np.zeros(slots) if options.get("scenario") == "fixed"
                  else self.rng.uniform(0., 2 * np.pi, slots))
        self.state = InterceptionState(pursuers, evaders, np.arange(MAX_PURSUERS) < len(p),
                                       np.arange(slots) < len(e),
                                       evaders[:, :3].copy(),
                                       phases,
                                       0, 0., [], "ongoing", failures, minimum, clearance, set(),
                                       (seed.state if isinstance(seed, np.random.SeedSequence) else seed),
                                       options.get("scenario", "random"))
        obs, info = self._output([], {}, 0.)
        def geometry(states, origin):
            offset = states[:, :3]-origin
            return dict(distance=np.linalg.norm(offset, axis=1).tolist(),
                        horizontal_distance=np.linalg.norm(offset[:, :2], axis=1).tolist(),
                        azimuth=np.arctan2(offset[:, 1], offset[:, 0]).tolist(),
                        altitude_above_facility=(states[:, 2]-self.config.center[2]).tolist())
        info["initial_geometry"] = dict(pursuers=geometry(p, self.config.center),
                                        evaders=geometry(e, navigation_point(self.config)))
        return obs, info

    def _output(self, events, parts, effort):
        s = self.state
        obs, central = observe(s, self.config)
        return obs, dict(centralized_state=central, active_mask=s.pursuer_mask.copy(),
                         evader_mask=s.evader_mask.copy(), pursuers=s.pursuers.copy(),
                         evaders=s.evaders.copy(), events=deepcopy(events),
                         outcome=s.outcome, step=s.step, time=s.time, reward_parts=parts,
                         control_effort=effort, minimum_separation=s.minimum_separation,
                         minimum_center_distance=s.minimum_separation,
                         minimum_surface_clearance=s.minimum_surface_clearance,
                         task=self.config.task, reset_seed=deepcopy(s.reset_seed),
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
        requests = np.zeros((len(states), 3))
        requests[s.pursuer_mask.nonzero()[0]] = command(active, c.pursuer_vmax)
        requests[MAX_PURSUERS:] = evader_commands(s.evaders, s.evader_mask,
                                                  s.pursuers[s.pursuer_mask], c, s.phases, s.time,
                                                  s.evader_origins)
        taus = np.r_[np.full(MAX_PURSUERS, c.pursuer_tau), np.full(len(s.evaders), c.evader_tau)]
        targets = np.zeros_like(requests)
        acceleration = np.zeros_like(requests)
        for ids, vmax in ((valid[valid < MAX_PURSUERS], c.pursuer_vmax),
                          (valid[valid >= MAX_PURSUERS], c.evader_vmax)):
            targets[ids] = execute_velocity(requests[ids], states[ids, 3:], vmax, taus[ids], c)
            acceleration[ids] = (targets[ids]-states[ids, 3:])/taus[ids, None]
        acc = np.linalg.norm(acceleration, axis=1)
        radii = np.r_[np.full(MAX_PURSUERS, c.pursuer_collision_radius),
                      np.full(len(s.evaders), c.evader_collision_radius)]

        @lru_cache(maxsize=None)
        def at(i, time):
            return advance(states[i], targets[i], taus[i], time)

        candidates, pairs = [], []
        for i, j in combinations(valid.tolist(), 2):
            def relative_position(t, i=i, j=j):
                return at(i, t)[:3]-at(j, t)[:3]

            bound = (np.linalg.norm(acceleration[i]-acceleration[j]) if taus[i] == taus[j]
                     else acc[i]+acc[j])
            radius = radii[i]+radii[j]
            pairs.append((i, j, relative_position, bound, radius))
            contact = windows(relative_position, bound, radius, 0., c.dt, c.event_time_tolerance)
            if contact:
                if j < MAX_PURSUERS:
                    priority, kind = 0, "friendly_collision"
                elif i >= MAX_PURSUERS:
                    priority, kind = 0, "evader_collision"
                else:
                    priority, kind = 3, "interception_contact"
                candidates.append((contact[0][0], priority, i, j, kind))
        for j in valid[valid >= MAX_PURSUERS]:
            breach = intrusion_windows(lambda t: at(j, t)[:3], acc[j], c)
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
                        collision_threshold=float(radii[i]+radii[j]) if i != j else None,
                        relative_speed=float(np.linalg.norm(first[3:]-second[3:])) if i != j else None)

        alive = np.r_[s.pursuer_mask, s.evader_mask]
        stop_times = np.full(len(states), c.dt)
        elapsed, outcome, events = c.dt, "ongoing", []
        candidates.sort()
        while candidates:
            candidates = [event for event in candidates if all(
                alive[k] for k in event[2:4]
            )]
            if not candidates:
                break
            earliest = candidates[0][0]
            simultaneous = [event for event in candidates if event[0] <= earliest + c.event_time_tolerance]
            event = min(simultaneous, key=lambda row: (row[1], row[0], row[2], row[3]))
            candidates.remove(event)
            time, _, i, j, kind = event
            events.append(event_row(kind, i, j, time))
            if kind == "interception_contact":
                alive[[i, j]] = False
                stop_times[[i, j]] = time
                if alive[MAX_PURSUERS:].any():
                    continue
                outcome = "success"
            else:
                outcome = kind
            elapsed = time
            break
        stop_times = np.minimum(stop_times, elapsed)
        next_states = states.copy()
        for i in valid:
            next_states[i] = at(i, stop_times[i])
        if not np.isfinite(next_states).all():
            raise FloatingPointError("nonfinite motion")
        minimum, clearance = s.minimum_separation, s.minimum_surface_clearance
        warning_pairs = set()
        closest_targets = {}
        closest_friendly = None
        for i, j, curve, bound, radius in pairs:
            end = min(stop_times[i], stop_times[j])
            closest, nearest_time = minimum_distance(curve, bound, end, with_time=True)
            minimum, clearance = min(minimum, closest), min(clearance, closest-radius)
            if self.diagnostics:
                if i < MAX_PURSUERS <= j and (j not in closest_targets or
                        closest < closest_targets[j]["center_distance"]):
                    row = event_row("closest_approach", i, j, nearest_time)
                    row["pursuer_horizontal_acceleration"] = float(
                        np.linalg.norm(acceleration[i, :2])*np.exp(-nearest_time/taus[i]))
                    closest_targets[j] = row
                if j < MAX_PURSUERS:
                    closest_friendly = closest if closest_friendly is None else min(closest_friendly, closest)
            if j >= MAX_PURSUERS or closest > c.friendly_warning_distance+.001:
                continue
            pair = (i, j)
            for start, stop in windows(curve, bound, c.friendly_warning_distance, 0., end,
                                       c.event_time_tolerance):
                if start > 1e-13 or pair not in s.warning_pairs:
                    events.append(event_row("friendly_warning", i, j, start))
                if stop >= elapsed-1e-13 and alive[i] and alive[j]:
                    warning_pairs.add(pair)
        events.sort(key=lambda event: event["time"])
        # Progress uses the same set of targets at both ends. A captured
        # target is compared at its capture endpoint with pursuers at that time.
        initial_distances, final_distances = [], []
        pursuers = np.flatnonzero(s.pursuer_mask)
        for j in valid[valid >= MAX_PURSUERS]:
            initial_distances.append(np.linalg.norm(states[pursuers, :3] - states[j, :3], axis=1).min())
            end = stop_times[j]
            final_distances.append(min(np.linalg.norm(at(i, min(end, stop_times[i]))[:3] - next_states[j, :3])
                                       for i in pursuers))
        parts = reward_parts(initial_distances, final_distances, int((s.evader_mask & ~alive[MAX_PURSUERS:]).sum()),
                             outcome, elapsed, c, episode_time=s.time+elapsed)
        terminated = outcome != "ongoing"
        truncated = not terminated and s.step + 1 >= c.max_steps
        if truncated:
            outcome = "timeout"
        effort = float(np.sum(stop_times[:MAX_PURSUERS, None]*np.square(targets[:MAX_PURSUERS]))
                       / s.pursuer_mask.sum())
        rewards = np.where(s.pursuer_mask, sum(parts.values()), 0.).astype(np.float32)
        if not np.isfinite(rewards).all():
            raise FloatingPointError("nonfinite reward")
        successor = InterceptionState(next_states[:MAX_PURSUERS], next_states[MAX_PURSUERS:],
                                      alive[:MAX_PURSUERS], alive[MAX_PURSUERS:], s.evader_origins, s.phases,
                                      s.step + 1, s.time + elapsed, s.events + events, outcome,
                                      s.spawn_rejections, minimum, clearance, warning_pairs, s.reset_seed,
                                      s.initialization)
        self.state = successor
        try:
            obs, info = self._output(events, parts, effort)
        except FloatingPointError:
            self.state = s
            raise
        info.update(commands=targets, requested_velocities=requests, initial_accelerations=acceleration,
                    command_duration=np.where(np.r_[s.pursuer_mask, s.evader_mask], stop_times, 0.),
                    team_reward=float(np.float32(sum(parts.values()))))
        if self.diagnostics:
            info.update(closest_targets=closest_targets, closest_friendly=closest_friendly)
        return obs, rewards, terminated, truncated, info

    def snapshot(self):
        return deepcopy(dict(version=VERSION, config=asdict(self.config),
                             state=self.state, rng=self.rng.bit_generator.state))

    def restore(self, snapshot):
        if snapshot.get("version") != VERSION or not matches_saved_config(snapshot.get("config"), self.config):
            raise ValueError("incompatible interception snapshot")
        snapshot = deepcopy(snapshot)
        self.rng.bit_generator.state = snapshot["rng"]
        self.state = snapshot["state"]
