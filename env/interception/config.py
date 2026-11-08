from dataclasses import dataclass, fields
import math

VERSION = "uav-interception-v2.0.0"
SCALES = {"1v1": (1, 1), "3v1": (3, 1), "5v2": (5, 2)}
SCENARIO_SCALES = {**SCALES, "3v3": (3, 3)}
TASKS = {"S0": ("1v1", "goal_seeking"), "S1": ("3v1", "goal_seeking"),
         "S2": ("3v1", "reactive_evasion"), "S3": ("5v2", "reactive_evasion"),
         "reference_3v3": ("3v3", "reactive_evasion")}
MAX_PURSUERS, MAX_EVADERS, AXES = 5, 2, 3
ENTITY = 2 * AXES
SLOT = ENTITY + 1
OBS_DIM = 3 * AXES + 1 + (MAX_PURSUERS - 1 + MAX_EVADERS) * SLOT
STATE_DIM = (MAX_PURSUERS + MAX_EVADERS) * SLOT + AXES + 2
MODES = ("constant_velocity", "goal_seeking", "bounded_maneuver", "reactive_evasion")


@dataclass(frozen=True)
class InterceptionConfig:
    scale: str = "3v1"
    task: str | None = None
    dt: float = 0.1
    max_steps: int = 600
    pursuer_vmax: float = 8.0
    evader_vmax: float = 6.0
    pursuer_tau: float = 1.5
    evader_tau: float = 1.5
    lower: tuple = (-80., -80., 3.)
    upper: tuple = (80., 80., 40.)
    center: tuple = (0., 0., 15.)
    radius: float = 5.
    spawn_distance: tuple = (35., 55.)
    defense_distance: tuple = (12., 20.)
    evader_azimuth: tuple = (-math.pi, math.pi)
    pursuer_azimuth: tuple = (-math.pi, math.pi)
    evader_height: tuple = (9., 25.)
    pursuer_height: tuple = (8., 24.)
    pursuer_initial_speed: tuple = (0., 1.5)
    evader_initial_speed: tuple = (0., 2.)
    minimum_spawn_clearance: float = 1.
    spawn_attempts: int = 128
    evader_mode: str | None = None
    evader_speed_fraction: float = 1.
    constant_direction: tuple = (-1., 0., 0.)
    maneuver_amplitude: float = .45
    maneuver_period: float = 8.
    reactive_range: float = 15.
    reactive_horizon: float = 2.
    reactive_danger_distance: float = 5.
    reactive_risk_weight: float = 2.
    reactive_turn_weight: float = .15
    reactive_max_angle_deg: float = 55.
    pursuer_collision_radius: float = .3
    evader_collision_radius: float = .3
    capture_inner_radius: float = 1.1
    capture_outer_radius: float = 3.
    capture_relative_speed_max: float = 1.5
    capture_dwell_seconds: float = .5
    near_miss_clearance: float = .3
    event_time_tolerance: float = 1e-7
    position_scale: float = 80.
    velocity_scale: float = 8.
    capture_reward: float = 5.
    success_reward: float = 10.
    breach_cost: float = 10.
    collision_cost: float = 10.
    boundary_cost: float = 10.
    step_cost: float = .01
    progress_weight: float = 1.
    progress_reference: float = 40.

    def __post_init__(self):
        task = ({"1v1": "S0", "3v1": "S2", "5v2": "S3", "3v3": "reference_3v3"}.get(self.scale)
                if self.task is None else self.task)
        if task not in TASKS or TASKS[task][0] != self.scale:
            raise ValueError("task and scale disagree")
        object.__setattr__(self, "task", task)
        if self.evader_mode is None:
            object.__setattr__(self, "evader_mode", TASKS[task][1])
        ranges = ("spawn_distance", "defense_distance", "evader_azimuth", "pursuer_azimuth",
                  "evader_height", "pursuer_height", "pursuer_initial_speed", "evader_initial_speed")
        for name, size in (("lower", 3), ("upper", 3), ("center", 3),
                           ("constant_direction", 3), *((name, 2) for name in ranges)):
            value = tuple(getattr(self, name))
            if len(value) != size or not all(map(math.isfinite, value)):
                raise ValueError(f"invalid {name}")
            object.__setattr__(self, name, value)
            if name in ranges and value[0] > value[1]:
                raise ValueError(f"reversed range {name}")
        for field in (field for field in fields(self) if field.type in (int, float)):
            value = getattr(self, field.name)
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"invalid {field.name}")
        for name in ("max_steps", "spawn_attempts"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"invalid {name}")
        for name in ("dt", "pursuer_vmax", "evader_vmax", "pursuer_tau", "evader_tau",
                     "radius", "capture_relative_speed_max", "capture_dwell_seconds",
                     "pursuer_collision_radius", "evader_collision_radius",
                     "reactive_horizon", "reactive_danger_distance", "event_time_tolerance", "position_scale",
                     "velocity_scale", "progress_reference", "maneuver_period", "reactive_range"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.scale not in SCENARIO_SCALES or self.evader_mode not in MODES:
            raise ValueError("unknown scale or evader mode")
        if not 0 < self.evader_speed_fraction <= 1 or not any(self.constant_direction):
            raise ValueError("invalid evader speed or constant direction")
        if not self.radius < self.spawn_distance[0] <= self.spawn_distance[1]:
            raise ValueError("invalid spawn distances")
        if not self.pursuer_collision_radius + self.evader_collision_radius < self.capture_inner_radius < self.capture_outer_radius:
            raise ValueError("capture band must be outside the collision envelope")
        if not 0 < self.reactive_max_angle_deg < 90 or self.defense_distance[0] <= 0:
            raise ValueError("invalid deployment or navigation cone")
        if any(lo < 0 or hi > vmax for (lo, hi), vmax in
               ((self.pursuer_initial_speed, self.pursuer_vmax),
                (self.evader_initial_speed, self.evader_vmax))):
            raise ValueError("initial speed ranges exceed limits")
        if not 1e-10 <= self.event_time_tolerance < self.dt / 10:
            raise ValueError("invalid event resolution")
        if not all(lo + self.radius <= center <= hi - self.radius and lo < hi
                   for lo, center, hi in zip(self.lower, self.center, self.upper)):
            raise ValueError("protected sphere must fit inside initialization bounds")

    @property
    def num_uavs(self):
        return MAX_PURSUERS

    @property
    def evader_slots(self):
        return max(MAX_EVADERS, SCENARIO_SCALES[self.scale][1])
