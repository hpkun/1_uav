from dataclasses import dataclass, fields
import math

VERSION = "uav-interception-v1.0.0"
SCALES = {"1v1": (1, 1), "3v1": (3, 1), "5v2": (5, 2)}
MAX_PURSUERS, MAX_EVADERS, AXES = 5, 2, 3
ENTITY = 2 * AXES
SLOT = ENTITY + 1
OBS_DIM = 3 * AXES + 1 + (MAX_PURSUERS - 1 + MAX_EVADERS) * SLOT
STATE_DIM = (MAX_PURSUERS + MAX_EVADERS) * SLOT + AXES + 2
MODES = ("constant_velocity", "goal_seeking", "bounded_maneuver", "reactive_evasion")


@dataclass(frozen=True)
class InterceptionConfig:
    scale: str = "3v1"
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
    defense_distance: float = 14.
    spawn_attempts: int = 128
    evader_mode: str = "goal_seeking"
    evader_speed_fraction: float = 1.
    constant_direction: tuple = (-1., 0., 0.)
    maneuver_amplitude: float = .45
    maneuver_period: float = 8.
    reactive_range: float = 15.
    reactive_gain: float = 1.2
    capture_radius: float = 2.
    capture_relative_speed_max: float = 1.5
    capture_dwell_steps: int = 5
    collision_distance: float = .6
    event_time_tolerance: float = 1e-7
    position_scale: float = 80.
    velocity_scale: float = 8.
    capture_reward: float = 5.
    success_reward: float = 10.
    breach_cost: float = 10.
    collision_cost: float = 10.
    step_cost: float = .01
    progress_weight: float = 1.
    progress_reference: float = 40.

    def __post_init__(self):
        for name, size in (("lower", 3), ("upper", 3), ("center", 3),
                           ("spawn_distance", 2), ("constant_direction", 3)):
            value = tuple(getattr(self, name))
            if len(value) != size or not all(map(math.isfinite, value)):
                raise ValueError(f"invalid {name}")
            object.__setattr__(self, name, value)
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, (int, float)) and (
                isinstance(value, bool) or not math.isfinite(value) or value < 0
            ):
                raise ValueError(f"invalid {field.name}")
        for name in ("max_steps", "spawn_attempts", "capture_dwell_steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"invalid {name}")
        for name in ("dt", "pursuer_vmax", "evader_vmax", "pursuer_tau", "evader_tau",
                     "radius", "defense_distance", "capture_relative_speed_max",
                     "collision_distance", "event_time_tolerance", "position_scale",
                     "velocity_scale", "progress_reference", "maneuver_period", "reactive_range"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.scale not in SCALES or self.evader_mode not in MODES:
            raise ValueError("unknown scale or evader mode")
        if not 0 < self.evader_speed_fraction <= 1 or not any(self.constant_direction):
            raise ValueError("invalid evader speed or constant direction")
        if not self.radius < self.spawn_distance[0] <= self.spawn_distance[1]:
            raise ValueError("invalid spawn distances")
        if not self.collision_distance < self.capture_radius:
            raise ValueError("capture radius must exceed collision distance")
        if not 1e-10 <= self.event_time_tolerance < self.dt / 10:
            raise ValueError("invalid event resolution")
        if not all(lo + self.radius <= center <= hi - self.radius and lo < hi
                   for lo, center, hi in zip(self.lower, self.center, self.upper)):
            raise ValueError("protected sphere must fit inside initialization bounds")

    @property
    def num_uavs(self):
        return MAX_PURSUERS

    @property
    def dwell_seconds(self):
        return self.capture_dwell_steps * self.dt
