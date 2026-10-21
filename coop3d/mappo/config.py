from dataclasses import asdict, dataclass
import math


@dataclass
class TrainConfig:
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.954
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy_coef: float = 0.001
    epochs: int = 10
    minibatch_size: int = 128
    num_envs: int = 10
    horizon: int = 250
    seed: int = 1000
    evaluation_seeds: tuple = (10000000, 10000001, 10000002)
    # Additional engineering defaults, not specified by the task/paper:
    adam_epsilon: float = 1e-5
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True
    beta_epsilon: float = 1e-6
    torch_threads: int = 1

    def __post_init__(self):
        for key in ('actor_lr', 'critic_lr', 'clip', 'adam_epsilon', 'max_grad_norm', 'beta_epsilon'):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f'invalid {key}')
        for key in ('gamma', 'gae_lambda'):
            if not 0 <= getattr(self, key) <= 1:
                raise ValueError(f'invalid {key}')
        for key in ('epochs', 'minibatch_size', 'num_envs', 'horizon', 'torch_threads'):
            if not isinstance(getattr(self, key), int) or getattr(self, key) < 1:
                raise ValueError(f'invalid {key}')
        if not 0 < self.beta_epsilon < .01 or not math.isfinite(self.entropy_coef) or self.entropy_coef < 0:
            raise ValueError('invalid distribution or entropy settings')
        self.evaluation_seeds = tuple(map(int, self.evaluation_seeds))
        if not self.evaluation_seeds or len(set(self.evaluation_seeds)) != len(self.evaluation_seeds):
            raise ValueError('evaluation seeds must be nonempty and unique')
        if self.seed in self.evaluation_seeds:
            raise ValueError('training and evaluation seeds must be independent')

    def to_dict(self):
        return asdict(self)
