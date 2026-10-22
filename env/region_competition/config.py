from dataclasses import asdict, dataclass
from pathlib import Path
import math
import yaml

ENVIRONMENT_VERSION = 'region-competition-v1.0.0'


@dataclass
class RegionConfig:
    scale: str = '3v3'
    blue_policy: str = 'region_safe'
    gravity: float = 9.81
    dt: float = .25
    v_min: float = 100.
    v_max: float = 400.
    gamma_limit: float = math.radians(85)
    denominator_epsilon: float = 1e-6
    residence_fraction_epsilon: float = 1e-12
    max_steps: int = 250
    num_regions: int = 3
    region_radius: float = 900.
    region_spacing: float = 2200.
    region_y_amplitude: float = 250.
    region_z_amplitude: float = 150.
    region_period: float = 40.
    start_distance: float = 2600.
    start_altitude: float = 3000.
    distance_scale: float = 5000.
    altitude_scale: float = 5000.
    safe_distance: float = 100.
    control_weight: float = 1.
    tracking_weight: float = 1.
    separation_weight: float = 1.
    numerical_weight: float = .1
    result_weight: float = 5.

    def __post_init__(self):
        if self.scale not in ('3v3','5v5') or self.blue_policy not in ('cruise','region','region_safe'):
            raise ValueError('unsupported scale or Blue policy')
        if type(self.max_steps) is not int or not 1 <= self.max_steps <= 250:
            raise ValueError('max_steps must be in [1,250]')
        if type(self.num_regions) is not int or not 1 <= self.num_regions <= 5:
            raise ValueError('num_regions must be in [1,5], never an empty task')
        for name,value in asdict(self).items():
            if name in ('scale','blue_policy','max_steps','num_regions'):
                continue
            if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value):
                raise ValueError(f'invalid {name}')
            if name.endswith('_amplitude') or name.endswith('_weight'):
                if value < 0: raise ValueError(f'negative {name}')
            elif value <= 0:
                raise ValueError(f'{name} must be positive')
        if not self.v_min < self.v_max or not 0 < self.gamma_limit < math.pi/2:
            raise ValueError('invalid flight envelope')
        if self.dt != .25 or self.v_min != 100 or self.v_max != 400:
            raise ValueError('this baseline fixes the published dt and speed envelope')
        if self.region_spacing <= 2*self.region_radius:
            raise ValueError('region spheres must be disjoint at all times')
        if self.residence_fraction_epsilon>1e-9:
            raise ValueError('residence resolution is a numerical guard, not a task threshold')

    @property
    def num_uavs(self):
        return 3 if self.scale == '3v3' else 5

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls,path):
        data=yaml.safe_load(Path(path).read_text(encoding='utf-8'))
        if not isinstance(data,dict): raise ValueError('configuration must be a mapping')
        return cls(**data)
