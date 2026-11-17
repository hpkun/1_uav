"""Independent 20v20 deterministic runtime; V3.5's 15-slot environment is untouched."""
from collections import Counter
from dataclasses import asdict, dataclass, fields
import hashlib
import json

import numpy as np

from .assignment import nearest_distance_assignment
from .intruder_policy import sample_level2_bezier_plan
from .large_scale_20v20 import (TWENTY_COUNT, initial_positions, PROTECTED_CENTER,
    PROTECTED_RADIUS_M, ARENA_LOWER, ARENA_UPPER)
from .large_scale_20v20_events import advance
from .scenario import diagnostics, inside_region


_DEFENDER_ANCHORS, _INTRUDER_ANCHORS = initial_positions()
PROTOCOL_20V20 = "uav-ideal-interception-20v20-nearest-v1"


@dataclass(frozen=True)
class TwentyConfig:
    """Frozen formal comparison profile; only the external episode seed varies."""
    scale: str = "20v20"
    assignment_policy: str = "nearest_dynamic"
    intruder_policy: str = "zuo_bezier"
    intruder_motion_level: str = "level2_maneuvering"
    dt: float = .05
    max_steps: int = 600
    lower: tuple = ARENA_LOWER
    upper: tuple = ARENA_UPPER
    defender_speeds: tuple = (20.,)*TWENTY_COUNT
    intruder_speeds: tuple = (20.,)*TWENTY_COUNT
    defender_starts: tuple = tuple(map(tuple, _DEFENDER_ANCHORS))
    intruder_starts: tuple = tuple(map(tuple, _INTRUDER_ANCHORS))
    defender_directions: tuple = ((1., 0., 0.),)*TWENTY_COUNT
    intruder_start_jitter_m: float = 5.
    intruder_goal_altitude_mode: str = "preserve_start_altitude"
    protected_center: tuple = PROTECTED_CENTER
    protected_radius: float = PROTECTED_RADIUS_M
    protected_height: float = 100.
    contact_distance: float = .1
    init_max_attempts: int = 128
    interception_reward: float = 5.
    protection_reward: float = 5.
    failure_penalty: float = 5.
    time_penalty: float = .01
    seed: int = 17

    def __post_init__(self):
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if any(getattr(self, field.name) != field.default
               for field in fields(self) if field.name != "seed"):
            raise ValueError("20v20 formal profile is fixed")

    @property
    def defender_count(self):
        return TWENTY_COUNT

    @property
    def intruder_count(self):
        return TWENTY_COUNT


class TwentyVTwentyEnv:
    def __init__(self, config=None):
        self.config = TwentyConfig() if config is None else config
        if not isinstance(self.config, TwentyConfig):
            raise TypeError("20v20 requires TwentyConfig")
        self.outcome = "uninitialized"

    def record_assignment(self):
        counts = Counter(self.assignment.values())
        self.assignment_history.append(dict(time=float(self.time),
            assignment={int(i): int(j) for i, j in sorted(self.assignment.items())},
            unique_target_count=len(counts),
            max_defenders_per_target=max(counts.values(), default=0)))

    def reset(self, seed=None):
        c = self.config
        sequence = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(
            c.seed if seed is None else seed)
        seed_id = int.from_bytes(sequence.generate_state(4, dtype="<u4").tobytes(), "little")
        rng = np.random.default_rng(sequence)
        defender_positions = np.asarray(c.defender_starts, dtype=float)
        for _ in range(c.init_max_attempts):
            try:
                plans = [sample_level2_bezier_plan(c, j, rng, seed_id)
                         for j in range(TWENTY_COUNT)]
            except ValueError:
                continue
            intruder_positions = np.asarray([plan.start for plan in plans], dtype=float)
            positions = np.concatenate((defender_positions, intruder_positions))
            paths = [plan.reference for plan in plans]
            if (np.any(positions <= c.lower) or np.any(positions >= c.upper)
                    or any(np.linalg.norm(positions[a]-positions[b]) <= c.contact_distance+1e-9
                           for a in range(len(positions)) for b in range(a+1, len(positions)))
                    or any(inside_region(point, c) for point in intruder_positions)
                    or not diagnostics(c, paths)["scenario_validity"]["region_crossing"]):
                continue
            break
        else:
            raise ValueError("20v20 Level-2 scene generation exhausted attempts")
        for plan in plans:
            plan.reference.plan_metadata = plan.metadata()
        self.paths = paths
        self.scene_diagnostics = diagnostics(c, paths)
        self.defenders = np.zeros((TWENTY_COUNT, 6), dtype=float)
        self.intruders = np.zeros((TWENTY_COUNT, 6), dtype=float)
        self.defenders[:, :3] = defender_positions
        self.defenders[:, 3:] = np.asarray(c.defender_directions)*np.asarray(c.defender_speeds)[:, None]
        self.intruders[:, :3] = intruder_positions
        for j, path in enumerate(paths):
            self.intruders[j, 3:] = path.sample(0.)[1]
        self.defender_mask = np.ones(TWENTY_COUNT, dtype=bool)
        self.intruder_mask = np.ones(TWENTY_COUNT, dtype=bool)
        self.assignment = nearest_distance_assignment(self.defenders[:, :3],
            self.intruders[:, :3], self.defender_mask, self.intruder_mask)
        self.original_assignment = self.assignment.copy()
        self.per_pair_initial_distance_m = {i: float(np.linalg.norm(
            self.defenders[i, :3]-self.intruders[j, :3])) for i, j in self.assignment.items()}
        self.assignment_total_cost_m = sum(self.per_pair_initial_distance_m.values())
        self.defender_status = ["active"]*TWENTY_COUNT
        self.intruder_status = ["active"]*TWENTY_COUNT
        self.time, self.step_count, self.outcome = 0., 0, "ongoing"
        self.events = []
        self.minimum_separation = None
        self.interception_count = self.breach_count = 0
        self.friendly_collision_count = self.intruder_collision_count = 0
        self.defender_boundary_count = 0
        self.seed = seed_id
        identity = dict(config=asdict(c), seed_id=seed_id,
            defender_starts=defender_positions.tolist(), intruder_starts=intruder_positions.tolist(),
            paths=[path.specification() for path in paths])
        self.scenario_id = hashlib.sha256(json.dumps(identity, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        self.assignment_history = []
        self.record_assignment()
        observation, info = self.output(include_paths=True)
        info["assignment_updates"] = self.assignment_history.copy()
        return observation, info

    def output(self, events=(), parts=None, team_reward=0., include_paths=False):
        c = self.config
        history = self.assignment_history
        info = dict(protocol=PROTOCOL_20V20, defenders=self.defenders.copy(),
            intruders=self.intruders.copy(), defender_mask=self.defender_mask.copy(),
            intruder_mask=self.intruder_mask.copy(), defender_status=self.defender_status.copy(),
            intruder_status=self.intruder_status.copy(), assignment=self.assignment.copy(),
            original_assignment=self.original_assignment.copy(),
            assignment_total_cost_m=self.assignment_total_cost_m,
            per_pair_initial_distance_m=self.per_pair_initial_distance_m.copy(),
            events=list(events), reward_parts=dict(parts or {}), team_reward=team_reward,
            time=self.time, step=self.step_count, outcome=self.outcome,
            seed=self.seed, scenario_id=self.scenario_id, scale=c.scale,
            interception_count=self.interception_count, breached_count=self.breach_count,
            intercepted_count=self.interception_count, breach_count=self.breach_count,
            friendly_collision_count=self.friendly_collision_count,
            intruder_collision_count=self.intruder_collision_count,
            defender_boundary_count=self.defender_boundary_count,
            remaining_intruder_count=int(self.intruder_mask.sum()),
            other_resolved_count=sum(status not in ("active", "intercepted", "breached")
                                     for status in self.intruder_status),
            active_defenders_remaining=int(self.defender_mask.sum()),
            initial_intruder_count=TWENTY_COUNT,
            mission_success=self.outcome == "all_intercepted",
            interception_ratio=self.interception_count/TWENTY_COUNT,
            breach_ratio=self.breach_count/TWENTY_COUNT,
            minimum_separation=self.minimum_separation,
            initial_unique_target_count=history[0]["unique_target_count"],
            initial_max_defenders_per_target=history[0]["max_defenders_per_target"],
            assignment_update_count=len(history)-1,
            mean_unique_targets_assigned=float(np.mean([row["unique_target_count"] for row in history])),
            minimum_unique_targets_assigned=min(row["unique_target_count"] for row in history),
            maximum_defenders_per_single_target=max(row["max_defenders_per_target"] for row in history))
        if include_paths:
            info["intruder_paths"] = [path.specification() for path in self.paths]
            info["intruder_plans"] = [path.plan_metadata for path in self.paths]
            info["scene_diagnostics"] = self.scene_diagnostics
        return self.defenders.copy(), info

    step = advance
