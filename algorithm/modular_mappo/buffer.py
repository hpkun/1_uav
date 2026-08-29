"""Rollout schema retaining raw rewards, wave identity and recurrent state."""
from dataclasses import dataclass
import numpy as np

@dataclass
class ModularRolloutBatch:
    observations:np.ndarray; actions:np.ndarray; raw_actions:np.ndarray
    old_log_probs:np.ndarray; rewards:np.ndarray; raw_environment_rewards:np.ndarray
    dones:np.ndarray; alive_masks:np.ndarray; next_observations:np.ndarray
    next_alive_masks:np.ndarray; wave_indices:np.ndarray; total_waves:np.ndarray
    contexts:np.ndarray; next_contexts:np.ndarray
    actor_hidden_before_step:np.ndarray|None=None
    critic_hidden_before_step:np.ndarray|None=None
    episode_masks:np.ndarray|None=None

def contiguous_chunks(time_steps:int,num_envs:int,sequence_length:int):
    return [(env,start,min(start+sequence_length,time_steps)) for env in range(num_envs) for start in range(0,time_steps,sequence_length)]

__all__=["ModularRolloutBatch","contiguous_chunks"]
