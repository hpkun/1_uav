from dataclasses import dataclass
from algorithm.mappo.config import TrainConfig

ALGORITHM_VERSION='mappo-region-v1.0.0'
PROTOCOL=dict(observation=12,state=60,action=3,value_heads=5,deadline_bootstrap=False)


@dataclass
class RegionTrainConfig(TrainConfig):
    horizon: int = 1500
    evaluation_episodes: int = 5
    checkpoint_interval_episodes: int = 50

    def __post_init__(self):
        super().__post_init__()
        if type(self.checkpoint_interval_episodes) is not int or self.checkpoint_interval_episodes<1:
            raise ValueError('checkpoint interval must be positive episodes')
