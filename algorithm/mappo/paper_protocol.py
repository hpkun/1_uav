"""Xu p.7 Eq.(25)/p.8 Table III: independent array/network protocol.

No target selection, geometry-to-observation chain, or reward is implemented.
The Coop3D runner/checkpoints remain a distinct 12/90/5 protocol.
"""
from dataclasses import dataclass

import numpy as np

ALGORITHM_VERSION = 'mappo-array-12x60-v1.0.0'
PROTOCOL = dict(actor_input=12, actor_action=3, critic_input=60, critic_output=5,
                state_layout='five explicitly supplied Eq25 raw slots, UAV ID order, zero padded',
                value_loss='sum of per-UAV valid means of squared errors, no half',
                actor_loss='sum of per-UAV valid mean clipped PPO and entropy',
                observation_source='external raw slots; no task/geometry/reward adapter',
                shared_parameters='shared actor and five-head critic: engineering assumption')


# Index, original symbol, raw SI unit. Observation normalization is unpublished.
OBSERVATION_FIELDS = (
    (0, 'O_i_jhat', '1'),
    (1, 'v_i', 'm/s'), (2, 'gamma_i', 'rad'), (3, 'psi_i', 'rad'),
    (4, 'alpha_i', 'rad'), (5, 'z_i', 'm'),
    (6, 'v_jhat', 'm/s'), (7, 'gamma_jhat', 'rad'), (8, 'psi_jhat', 'rad'),
    (9, 'alpha_jhat', 'rad'), (10, 'z_jhat', 'm'), (11, 'd_i_jhat', 'm'),
)


def validate_observation_slots(slots):
    """Validate already supplied raw slots; do not infer targets or normalize.

    Eq.(25)'s O flag and the current-friendly validity mask in Eq.(26) are
    related but not interchangeable definitions. This validator checks only
    the binary slot flag and p.8's all-zero invalid-slot convention.
    """
    a = np.asarray(slots, dtype=np.float64)
    if a.ndim < 1 or a.shape[-1] != 12 or not np.isfinite(a).all():
        raise ValueError('expected finite raw 12D slots')
    valid = a[..., 0]
    if not np.isin(valid, (0, 1)).all():
        raise ValueError('slot validity must be binary')
    if np.any(a[valid == 0] != 0):
        raise ValueError('invalid slots must be entirely zero')
    return a.copy()


def pack_centralized_slots(slots, masks):
    """Explicit [...,N,12] -> [...,60], N<=5; never accept or slice 90D.

    Packing five slots is a declared interface assumption: Eq.(14)'s complete
    relationship set to Table III's 60D mapping is not fully published.
    """
    a = validate_observation_slots(slots)
    m = np.asarray(masks)
    if a.ndim < 2 or not 1 <= a.shape[-2] <= 5 or m.shape != a.shape[:-1]:
        raise ValueError('expected 1..5 local slots with matching UAV masks')
    if not np.isin(m, (0, 1)).all() or not np.array_equal(a[..., 0], m):
        raise ValueError('current UAV masks must match explicit slot flags')
    result = np.zeros((*a.shape[:-2], 5, 12), dtype=a.dtype)
    result[..., :a.shape[-2], :] = a
    return result.reshape(*a.shape[:-2], 60)


@dataclass(frozen=True)
class TableIII:
    """Published constants only; deliberately not a runnable TrainConfig."""
    actor_hidden: tuple = (128, 128)
    critic_hidden: tuple = (128, 128)
    activation: str = 'Tanh'
    actor_distribution: str = 'Beta'
    actor_input: int = 12
    critic_input: int = 60
    action_dimension: int = 3
    optimizer: str = 'Adam'
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.954
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy_coef: float = 0.001
    ppo_epochs: int = 10
    minibatch_size: int = 128
    num_envs: int = 10
    horizon: int = 1500
    max_episode_steps: int = 250
    checkpoint_interval_episodes: int = 50
