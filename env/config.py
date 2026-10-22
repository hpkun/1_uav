"""Validated configuration; all units are SI and angles are radians."""
from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
import yaml

ENVIRONMENT_VERSION = 'coop3d-v3.0.0'
SCALES = {'3U3P': (3, 3), '5U5P': (5, 5), '3U5P': (3, 5)}


@dataclass
class ScenarioConfig:
    formation_half_width: float = 500.0
    uav_altitude: tuple = (2500.0, 3500.0)
    initial_speed: tuple = (200.0, 250.0)
    initial_gamma: tuple = (-math.pi / 36, math.pi / 36)
    heading_jitter: float = math.pi / 6
    waypoint_x: tuple = (5500.0, 11000.0)
    waypoint_y: tuple = (-2000.0, 2000.0)
    waypoint_z: tuple = (2000.0, 4000.0)
    initial_min_separation: float = 200.0
    max_sampling_attempts: int = 10000


@dataclass
class EnvConfig:
    scale: str = '3U3P'
    assignment_mode: str = 'hungarian'
    random_scenario: bool = True
    gravity: float = 9.81
    dt: float = 0.25
    v_min: float = 100.0
    v_max: float = 400.0
    denominator_epsilon: float = 1e-6
    max_steps: int = 250
    completion_radius: float = 300.0
    safe_distance: float = 100.0
    distance_scale: float = 10000.0
    altitude_scale: float = 5000.0
    failure_events: dict = field(default_factory=dict)  # {one_based_step: [uav_ids]}
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)

    def __post_init__(self):
        if isinstance(self.scenario, dict):
            self.scenario = ScenarioConfig(**self.scenario)
        for k in self.failure_events:
            if not (type(k) is int or isinstance(k,str) and k.isdecimal()):
                raise ValueError('failure event steps must be integer IDs')
        self.failure_events = {int(k): list(v) for k, v in self.failure_events.items()}
        if self.scale not in SCALES or self.assignment_mode not in ('hungarian', 'nearest'):
            raise ValueError('unsupported scale or assignment mode')
        for key in ('gravity', 'dt', 'v_min', 'v_max', 'denominator_epsilon', 'completion_radius', 'safe_distance', 'distance_scale', 'altitude_scale'):
            if isinstance(getattr(self,key),bool) or not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if not self.v_min < self.v_max or self.denominator_epsilon >= 1:
            raise ValueError('invalid speed limits or singularity tolerance')
        if type(self.max_steps) is not int or not 1 <= self.max_steps <= 250:
            raise ValueError('max_steps must be an integer in [1,250]')
        s = self.scenario
        if type(self.random_scenario) is not bool:
            raise ValueError('random_scenario must be boolean')
        for key in ('uav_altitude', 'initial_speed', 'initial_gamma', 'waypoint_x', 'waypoint_y', 'waypoint_z'):
            bounds = getattr(s, key)
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ValueError(f'{key} must have two bounds')
            low, high = bounds
            if not all(map(math.isfinite, (low, high))) or low > high:
                raise ValueError(f'invalid sampling range: {key}')
        if not self.v_min <= s.initial_speed[0] <= s.initial_speed[1] <= self.v_max:
            raise ValueError('initial_speed outside speed limits')
        for key in ('formation_half_width', 'heading_jitter', 'initial_min_separation'):
            if not math.isfinite(getattr(s, key)) or getattr(s, key) < 0:
                raise ValueError(f'invalid scenario {key}')
        if type(s.max_sampling_attempts) is not int or s.max_sampling_attempts < 1:
            raise ValueError('max_sampling_attempts must be positive')
        for step, ids in self.failure_events.items():
            if step < 1 or step > self.max_steps or any(type(i) is not int or not 0 <= i < self.num_uavs for i in ids):
                raise ValueError('invalid deterministic failure event')

    @property
    def num_uavs(self):
        return SCALES[self.scale][0]

    @property
    def num_waypoints(self):
        return SCALES[self.scale][1]

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path):
        data = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError('environment configuration must be a mapping')
        return cls(**data)
