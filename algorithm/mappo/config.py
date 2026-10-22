from dataclasses import asdict, dataclass
import math
from pathlib import Path
import yaml


@dataclass
class TrainConfig:
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.954
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy_coef: float = 0.001
    ppo_epochs: int = 10
    minibatch_size: int = 128
    num_envs: int = 10
    horizon: int = 1500  # Xu Table III rollout; not the 250-step episode limit.
    seed: int = 1000
    evaluation_seeds: tuple = (10000000, 10000001, 10000002)
    # Additional engineering defaults, not specified by the task/paper:
    adam_epsilon: float = 1e-5
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True
    beta_epsilon: float = 1e-6
    torch_threads: int = 1
    initialization_mode: str = 'trim_centered'
    beta_concentration: float = 16.0
    output_weight_gain: float = 0.01
    evaluation_interval_sampled_steps: int = 25000
    evaluation_episodes: int = 5
    checkpoint_interval_sampled_steps: int = 25000
    console_interval_sampled_steps: int = 2500

    def __post_init__(self):
        for key in ('actor_lr', 'critic_lr', 'clip', 'adam_epsilon', 'max_grad_norm', 'beta_epsilon'):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f'invalid {key}')
        for key in ('gamma', 'gae_lambda'):
            if not 0 <= getattr(self, key) <= 1:
                raise ValueError(f'invalid {key}')
        for key in ('ppo_epochs', 'minibatch_size', 'num_envs', 'horizon', 'torch_threads'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f'invalid {key}')
        if not 0 < self.beta_epsilon < .01 or not math.isfinite(self.entropy_coef) or self.entropy_coef < 0:
            raise ValueError('invalid distribution or entropy settings')
        if any(type(s) is not int or s<0 for s in self.evaluation_seeds):
            raise ValueError('evaluation seeds must be nonnegative integers')
        self.evaluation_seeds = tuple(self.evaluation_seeds)
        if not self.evaluation_seeds or len(set(self.evaluation_seeds)) != len(self.evaluation_seeds):
            raise ValueError('evaluation seeds must be nonempty and unique')
        if self.seed in self.evaluation_seeds:
            raise ValueError('training and evaluation seeds must be independent')
        if self.initialization_mode not in ('trim_centered', 'legacy', 'symmetric'):
            raise ValueError('invalid initialization_mode')
        if not math.isfinite(self.beta_concentration) or self.beta_concentration <= 8:
            raise ValueError('beta_concentration must exceed 8 for softplus+1 trim parameters')
        if not math.isfinite(self.output_weight_gain) or self.output_weight_gain <= 0:
            raise ValueError('invalid output_weight_gain')
        for key in ('evaluation_interval_sampled_steps','evaluation_episodes','checkpoint_interval_sampled_steps','console_interval_sampled_steps'):
            if type(getattr(self,key)) is not int or getattr(self,key) < 1:
                raise ValueError(f'invalid {key}')
        if type(self.seed) is not int or self.seed < 0 or type(self.normalize_advantages) is not bool:
            raise ValueError('invalid seed or normalize_advantages')
        if self.clip >= 1:
            raise ValueError('clip must be below 1')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path):
        data = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError('algorithm configuration must be a mapping')
        return cls(**data)


@dataclass
class ProtocolTrainConfig(TrainConfig):
    """12/60 updater and Xu air-combat training settings.

    Extra settings are engineering assumptions inherited explicitly from
    TrainConfig, not author-published Beta/normalization/Adam details.
    """
    checkpoint_interval_episodes: int = 50
    evaluation_episodes: int = 200
    training_seed_index: int = 1
    initialization_mode: str = 'symmetric'
    preserve_singleton_advantages: bool = True
    training_episodes: int = 5000
    independent_training_seeds: int = 5
    ppo_masking: bool = True

    def __post_init__(self):
        super().__post_init__()
        if type(self.checkpoint_interval_episodes) is not int or self.checkpoint_interval_episodes < 1:
            raise ValueError('invalid checkpoint_interval_episodes')
        if type(self.training_seed_index) is not int or self.training_seed_index < 0:
            raise ValueError('invalid training_seed_index')
        if type(self.preserve_singleton_advantages) is not bool:
            raise ValueError('invalid preserve_singleton_advantages')
        if type(self.ppo_masking) is not bool:
            raise ValueError('invalid ppo_masking')
        for key in ('training_episodes', 'independent_training_seeds'):
            if type(getattr(self,key)) is not int or getattr(self,key)<1:
                raise ValueError(f'invalid {key}')

    def reproduction_assumptions(self):
        return dict(initialization=self.initialization_mode, beta_parameters='softplus(raw)+1; small symmetric output weights',
                    output_weight_gain=self.output_weight_gain, adam_epsilon=self.adam_epsilon,
                    max_grad_norm=self.max_grad_norm, normalize_advantages=self.normalize_advantages,
                    advantage_normalization='per agent population std+1e-8; singleton kept raw' if self.preserve_singleton_advantages else 'singleton centered to zero',
                    beta_epsilon=self.beta_epsilon, loss_reduction='sum of per-agent valid means; MSE without half',
                    timeout='bootstrap real final state; stop GAE at episode boundary',
                    checkpoint_timing='exact completed episode; retain partial rollout/wave and pre-sampled actions; no extra PPO update',
                    masking_ablation='loss-only; physical/GAE/normalization masks retained; dead actions=.5 with exact logp, GAE target=0',
                    budget='collect full per-worker horizons; exact episode budget via final active-worker waves')

    def worker_seeds(self):
        """Published initial-worker rule (Xu p.8); no per-episode reseeding."""
        return tuple(1000*self.training_seed_index+r for r in range(self.num_envs))
