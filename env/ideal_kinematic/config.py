"""Independent SI configuration for ideal point-mass experiments."""
from dataclasses import dataclass
import math
import numpy as np

PROTOCOL = "uav-ideal-kinematic-v1.0.0"
TASK_COUNTS = {"IK0": (1, 0), "IK1": (3, 0), "IK2": (3, 1), "IK3": (5, 2)}
SLOTS = 5
TARGET_SLOTS = 2
OBS_DIM = 64
STATE_DIM = 5 * 8 + 2 * 7 + 4 + 1


@dataclass(frozen=True)
class IdealConfig:
    task: str = "IK0"
    dt: float = 0.5
    max_steps: int = 80
    lower: tuple = (-30., -30., 0.)
    upper: tuple = (30., 30., 40.)
    speeds: tuple = (2., 2., 2., 2., 2.)
    starts: tuple = ((-15., -12., 10.), (-15., 0., 15.), (-15., 12., 20.),
                     (-20., -6., 25.), (-20., 6., 30.))
    directions: tuple = ((1., 0., 0.),) * 5
    waypoints: tuple = ((15., -12., 10.), (15., 0., 15.), (15., 12., 20.),
                        (15., -6., 25.), (15., 6., 30.))
    target_starts: tuple = ((0., -5., 15.), (0., 7., 25.))
    target_ends: tuple = ((12., -5., 15.), (12., 7., 25.))
    target_speed: float = 1.
    target_mode: str = "straight"
    initialization: str = "fixed"
    random_range: float = 1.
    safety_distance: float = 2.
    warning_distance: float = 4.
    contact_distance: float = 0.5
    arrival_distance: float = 1.5
    tracking_distance: float = 3.
    position_scale: float = 30.
    speed_scale: float = 3.
    progress_weight: float = 1.
    tracking_weight: float = 0.2
    completion_weight: float = 5.
    danger_weight: float = 1.
    boundary_weight: float = 5.
    time_weight: float = 0.01
    seed: int = 17

    def __post_init__(self):
        if self.task not in TASK_COUNTS or self.target_mode not in ("straight", "bezier") or self.initialization not in ("fixed", "random"):
            raise ValueError("invalid task, target mode or initialization")
        if type(self.max_steps) is not int or self.max_steps < 1 or type(self.seed) is not int or self.seed < 0:
            raise ValueError("invalid step budget or seed")
        positive = ("dt", "target_speed", "safety_distance", "warning_distance",
                    "contact_distance", "arrival_distance", "tracking_distance",
                    "position_scale", "speed_scale")
        nonnegative = ("random_range", "progress_weight", "tracking_weight",
                       "completion_weight", "danger_weight", "boundary_weight", "time_weight")
        for key in positive + nonnegative:
            value = getattr(self, key)
            if not math.isfinite(value) or (value <= 0 if key in positive else value < 0):
                raise ValueError(f"invalid {key}")
        if not self.contact_distance < self.safety_distance < self.warning_distance:
            raise ValueError("contact, safety and warning distances must increase")
        for key, count in (("lower", 3), ("upper", 3), ("speeds", SLOTS),
                           ("starts", SLOTS), ("directions", SLOTS), ("waypoints", SLOTS),
                           ("target_starts", TARGET_SLOTS), ("target_ends", TARGET_SLOTS)):
            array = np.asarray(getattr(self, key), dtype=float)
            shape = (count,) if key == "speeds" else (3,) if key in ("lower", "upper") else (count, 3)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"invalid {key}")
        if np.any(np.asarray(self.upper) <= np.asarray(self.lower)) or np.any(np.asarray(self.speeds) <= 0):
            raise ValueError("invalid region or fixed speeds")
        if np.any(np.linalg.norm(np.asarray(self.directions), axis=1) <= 0):
            raise ValueError("initial directions must be nonzero")
        n, targets = TASK_COUNTS[self.task]
        points = np.asarray(self.starts[:n])
        if np.any(points - self.random_range <= np.asarray(self.lower)) or np.any(points + self.random_range >= np.asarray(self.upper)):
            raise ValueError("initial UAV positions must stay inside the region")
        if any(np.linalg.norm(points[i] - points[j]) - 2 * self.random_range <= self.safety_distance
               for i in range(n) for j in range(i + 1, n)):
            raise ValueError("initial UAV safety envelopes overlap")
        for key in ("waypoints", "target_starts", "target_ends"):
            values = np.asarray(getattr(self, key)[:n if key == "waypoints" else targets])
            if len(values) and (np.any(values <= np.asarray(self.lower)) or np.any(values >= np.asarray(self.upper))):
                raise ValueError(f"{key} outside region")
        references = np.asarray(self.target_starts[:targets])
        if any(np.linalg.norm(point - target) - self.random_range <= self.contact_distance
               for point in points for target in references):
            raise ValueError("initial UAV and reference target overlap")
        if any(np.linalg.norm(references[i] - references[j]) <= self.contact_distance
               for i in range(targets) for j in range(i + 1, targets)):
            raise ValueError("initial reference targets overlap")

    @property
    def num_uavs(self):
        return SLOTS

    @property
    def active_count(self):
        return TASK_COUNTS[self.task][0]

    @property
    def target_count(self):
        return TASK_COUNTS[self.task][1]
