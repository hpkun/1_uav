from copy import deepcopy
from dataclasses import asdict, dataclass
from itertools import combinations
import numpy as np

from .config import InterceptionConfig, VERSION, MAX_PURSUERS, MAX_EVADERS, OBS_DIM, STATE_DIM
from .events import windows, intersection, minimum_distance
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
        phases = (np.zeros(MAX_EVADERS) if options.get("scenario") == "fixed"
                  else self.rng.uniform(0., 2 * np.pi, MAX_EVADERS))
        self.state = InterceptionState(pursuers, evaders, np.arange(MAX_PURSUERS) < len(p),
                                       np.arange(MAX_EVADERS) < len(e),
                                       np.zeros((MAX_PURSUERS, MAX_EVADERS)),
                                       phases,
                                       0, 0., [], "ongoing", failures, minimum)
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

        def at(i, time):
            return advance(states[i], targets[i], taus[i], time)

        def threshold(curve, bound, radius):
            return windows(curve, bound, radius, 0., c.dt, c.event_time_tolerance)

        def relative(i, j, segment):
            return lambda t: at(i, t)[segment] - at(j, t)[segment]

        position, velocity = slice(0, 3), slice(3, 6)
        candidates, pairs = [], []
        dwell = np.zeros_like(s.dwell)
        for i, j in combinations(valid.tolist(), 2):
            curve = relative(i, j, position)
            shared_tau = taus[i] == taus[j]
            bound = (np.linalg.norm(acceleration[i] - acceleration[j]) if shared_tau
                     else acc[i] + acc[j])
            pairs.append((i, j, curve, bound))
            collision = threshold(curve, bound, c.collision_distance)
            if collision:
                candidates.append((collision[0][0], 0, i, j, "collision"))
            if i < MAX_PURSUERS <= j:
                target = j - MAX_PURSUERS
                near = threshold(curve, bound, c.capture_radius)
                jerk = bound / taus[i] if shared_tau else acc[i] / taus[i] + acc[j] / taus[j]
                slow = threshold(relative(i, j, velocity), jerk,
                                 c.capture_relative_speed_max) if near else []
                for start, end in intersection(near, slow):
                    previous = s.dwell[i, target] if start <= 1e-13 else 0.
                    reached = start + max(0., c.dwell_seconds - previous)
                    if reached <= end + 1e-13:
                        candidates.append((min(reached, end), 2, i, j, "capture"))
                    if end >= c.dt - 1e-13:
                        dwell[i, target] = previous + end - start
        center = np.array(c.center)
        for j in valid[valid >= MAX_PURSUERS]:
            breach = threshold(lambda t: at(j, t)[:3] - center, acc[j], c.radius)
            if breach:
                candidates.append((breach[0][0], 1, int(j), int(j), "breach"))

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
            events.append(dict(kind=kind, time=s.time + time, step=s.step + 1, entities=[i, j],
                               positions=[at(i, time)[:3].tolist(), at(j, time)[:3].tolist()]))
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
        next_states = states.copy()
        for i in valid:
            next_states[i] = at(i, stop_times[i])
        if not np.isfinite(next_states).all():
            raise FloatingPointError("nonfinite motion")
        minimum = min([s.minimum_separation] + [
            minimum_distance(curve, bound, min(stop_times[i], stop_times[j]))
            for i, j, curve, bound in pairs
        ])
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
                                      s.spawn_rejections.copy(), minimum)
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
