"""Independent ideal three-dimensional point-mass multi-agent environment."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import numpy as np

from .config import IdealConfig, PROTOCOL, OBS_DIM, STATE_DIM, SLOTS, TARGET_SLOTS
from .geometry import boundary, closest, direction, proximity, inside_intervals, merge_intervals, subtract_intervals
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
        directions = np.asarray(c.directions[:c.active_count], float)
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        self.states[:c.active_count, 3:] = directions * np.asarray(c.speeds[:c.active_count])[:, None]
        self.references = [Reference(c.target_starts[i], c.target_ends[i], c.target_speed,
                                     c.target_mode, c.reference_end)
                           for i in range(c.target_count)]
        failure = None
        for attempt in range(c.init_max_attempts):
            positions = np.asarray(c.starts[:c.active_count], float).copy()
            if c.initialization == "random":
                positions += rng.uniform(-c.random_range, c.random_range, positions.shape)
            failure = self._initial_failure(positions)
            if failure is None:
                self.states[:c.active_count, :3] = positions
                break
        else:
            raise ValueError(f"initialization failed after {c.init_max_attempts} attempts: {failure}")
        self.assignment = fixed_assignment(self.states[:c.active_count],
                            list(range(c.target_count if c.target_count else c.active_count)), self.status[:c.active_count])
        identity_config = asdict(c)
        for key in ("seed", "initialization", "random_range", "init_max_attempts"):
            identity_config.pop(key)
        identity = dict(configuration=identity_config, initial_states=self.states.tolist(),
                        assignment=self.assignment)
        self.scenario_id = hashlib.sha256(json.dumps(identity, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        self.time, self.steps, self.outcome = 0., 0, "ongoing"
        self.minimum_separation = None
        self.warning_pairs = set()
        self.safety_pairs = set()
        self.warning_count = self.safety_count = self.contact_count = 0
        self.safety_duration = 0.
        self.distance_traveled = np.zeros(SLOTS)
        self.tracking_intervals = [[] for _ in range(SLOTS)]
        self.tracking_error_integral = np.zeros(SLOTS)
        self.tracking_samples = [[] for _ in range(SLOTS)]
        self.reference_motion_time = np.zeros(TARGET_SLOTS)
        return self.output()

    def _initial_failure(self, positions):
        c = self.config
        if not np.isfinite(positions).all() or not np.isfinite(self.states).all():
            return "nonfinite state"
        if np.any(positions <= np.asarray(c.lower)) or np.any(positions >= np.asarray(c.upper)):
            return "UAV outside region"
        if not np.allclose(np.linalg.norm(self.states[:c.active_count, 3:], axis=1),
                           c.speeds[:c.active_count], rtol=0, atol=1e-10):
            return "initial speed mismatch"
        if any(np.linalg.norm(positions[i]-positions[j]) <= c.safety_distance
               for i in range(c.active_count) for j in range(i+1, c.active_count)):
            return "UAV safety overlap"
        targets = np.asarray(c.target_starts[:c.target_count])
        if any(np.linalg.norm(p-q) <= c.contact_distance for p in positions for q in targets):
            return "UAV reference overlap"
        if any(np.linalg.norm(targets[i]-targets[j]) <= c.contact_distance
               for i in range(c.target_count) for j in range(i+1, c.target_count)):
            return "reference overlap"
        return None

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
                            (self.states[j, 3:]-own[3:])/c.speed_scale, 1.]
                     if self.active[j] else np.zeros(7)
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
                    warning_count=self.warning_count, safety_count=self.safety_count,
                    safety_duration=self.safety_duration, contact_count=self.contact_count,
                    distance_traveled=self.distance_traveled.copy(),
                    tracking=self.tracking_metrics(),
                    reference_motion_time=self.reference_motion_time,
                    seed=self.seed, scenario_id=self.scenario_id, task=c.task,
                    action_semantics="desired_motion_direction")
        return obs, info

    def tracking_metrics(self):
        if not self.config.target_count:
            return []
        elapsed = self.time
        result = []
        for i in range(self.config.active_count):
            intervals = merge_intervals(self.tracking_intervals[i])
            valid = sum(b-a for a,b in intervals)
            longest = max((b-a for a,b in intervals), default=0.)
            result.append(dict(instantaneous_error=float(np.linalg.norm(
                self.target(i, self.time)[0]-self.states[i, :3])),
                mean_error=float(self.tracking_error_integral[i]/elapsed) if elapsed else None,
                valid_fraction=float(valid/elapsed) if elapsed else None,
                longest_valid_time=longest,
                departure_count=sum(right < elapsed-1e-10 for _, right in intervals),
                outside_time=max(0., elapsed-valid),
                error_samples=self.tracking_samples[i].copy()))
        return result

    from .step import step

    def snapshot(self):
        state = {key: deepcopy(value) for key, value in self.__dict__.items() if key != "config"}
        return dict(protocol=PROTOCOL, configuration=asdict(self.config), state=state)

    def restore(self, state):
        if state.get("protocol") != PROTOCOL or state.get("configuration") != asdict(self.config):
            raise ValueError("incompatible ideal environment snapshot")
        self.__dict__.update(deepcopy(state["state"]))
