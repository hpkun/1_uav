"""Independent SI configuration for abstract geometric interception."""
from dataclasses import dataclass
import math
import numpy as np

PROTOCOL = "uav-ideal-interception-v3.1.0"
SCALES = {"1v1": (1, 1), "3v1": (3, 1), "5v2": (5, 2)}
DEFENDER_SLOTS = 5
INTRUDER_SLOTS = 2
OBS_DIM = 68
STATE_DIM = 66


@dataclass(frozen=True)
class InterceptionConfig:
    scale: str = "1v1"
    dt: float = .05
    max_steps: int = 600
    lower: tuple = (-300., -300., 0.)
    upper: tuple = (300., 300., 120.)
    defender_speeds: tuple = (20., 20., 20., 20., 20.)
    intruder_speeds: tuple = (20., 20.)
    defender_starts: tuple = ((-100., -50., 60.), (-100., 0., 60.), (-100., 50., 60.),
                              (-120., -80., 80.), (-120., 80., 80.))
    defender_directions: tuple = ((1., 0., 0.),)*5
    intruder_starts: tuple = ((200., -10., 60.), (200., 10., 80.))
    intruder_ends: tuple = ((-260., -10., 60.), (-260., 10., 80.))
    intruder_policy: str = "zuo_bezier"
    intruder_mode: str | None = None  # Explicit legacy/test reference fixture.
    intruder_start_jitter_m: float = 5.
    intruder_bezier_jitter_ratio: float = .04
    initialization: str = "fixed"
    random_range: float = 5.
    init_max_attempts: int = 128
    protected_center: tuple = (-220., 0., 0.)
    protected_radius: float = 20.
    protected_height: float = 100.
    contact_distance: float = .1
    position_scale: float = 300.
    speed_scale: float = 20.
    interception_reward: float = 5.
    protection_reward: float = 5.
    failure_penalty: float = 5.
    time_penalty: float = .01
    seed: int = 17

    def __post_init__(self):
        if self.scale not in SCALES or self.initialization not in ("fixed", "random"):
            raise ValueError("invalid scale or initialization")
        if self.intruder_policy not in ("zuo_bezier", "legacy_reference") or self.intruder_mode not in (None,"straight","bezier"):
            raise ValueError("invalid intruder policy or fixture mode")
        if self.intruder_mode is not None and self.intruder_policy=="zuo_bezier":
            # Old explicit intruder_mode configurations remain geometric fixtures.
            object.__setattr__(self,"intruder_policy","legacy_reference")
        if self.intruder_policy=="legacy_reference" and self.intruder_mode is None:
            raise ValueError("legacy reference fixture requires intruder_mode")
        if (type(self.max_steps) is not int or self.max_steps < 1
                or type(self.init_max_attempts) is not int or self.init_max_attempts < 1
                or type(self.seed) is not int or self.seed < 0):
            raise ValueError("invalid step, retry or seed")
        for name in ("dt", "protected_radius", "protected_height", "contact_distance",
                     "position_scale", "speed_scale"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"invalid {name}")
        for name in ("random_range", "intruder_start_jitter_m", "intruder_bezier_jitter_ratio",
                     "interception_reward", "protection_reward",
                     "failure_penalty", "time_penalty"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {name}")
        shapes = dict(lower=(3,), upper=(3,), defender_speeds=(5,), intruder_speeds=(2,),
                      defender_starts=(5,3), defender_directions=(5,3),
                      intruder_starts=(2,3), intruder_ends=(2,3), protected_center=(3,))
        for name, shape in shapes.items():
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"invalid {name}")
            object.__setattr__(self, name, tuple(tuple(row) if isinstance(row,list) else row
                                                  for row in value.tolist()))
        if np.any(np.asarray(self.upper) <= self.lower):
            raise ValueError("invalid region")
        if np.any(np.asarray(self.defender_speeds) <= 0) or np.any(np.asarray(self.intruder_speeds) <= 0):
            raise ValueError("fixed speeds must be positive")
        if np.any(np.linalg.norm(np.asarray(self.defender_directions), axis=1) <= 0):
            raise ValueError("defender directions must be nonzero")
        d, i = SCALES[self.scale]
        starts = np.r_[np.asarray(self.defender_starts[:d]), np.asarray(self.intruder_starts[:i])]
        margin=max(self.random_range,self.intruder_start_jitter_m if self.intruder_policy=="zuo_bezier" else 0.)
        if np.any(starts-np.asarray(self.lower) <= margin) or np.any(np.asarray(self.upper)-starts <= margin):
            raise ValueError("initial entities must remain inside region")
        if any(np.linalg.norm(starts[a]-starts[b]) <= self.contact_distance+1e-9
               for a in range(len(starts)) for b in range(a+1,len(starts))):
            raise ValueError("initial entity overlap")
        if not np.all(np.isfinite(self.intruder_ends)):
            raise ValueError("invalid intruder endpoints")

    @property
    def num_uavs(self):
        return DEFENDER_SLOTS

    @property
    def defender_count(self):
        return SCALES[self.scale][0]

    @property
    def intruder_count(self):
        return SCALES[self.scale][1]


def zuo_paper_profile(**overrides):
    """Optional paper-like numerical profile, separate from project defaults."""
    values=dict(dt=.1,defender_speeds=(25.,)*5,intruder_speeds=(30.,)*2,
                contact_distance=.1)
    return InterceptionConfig(**(values|overrides))
