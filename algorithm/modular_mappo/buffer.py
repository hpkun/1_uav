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

def recurrent_batch_plan(time_steps:int,num_envs:int,sequence_length:int,minibatch_size:int,ppo_epochs:int=1):
    chunks=len(contiguous_chunks(time_steps,num_envs,sequence_length))
    sequences_per_minibatch=max(1,minibatch_size//sequence_length)
    minibatches_per_epoch=int(np.ceil(chunks/sequences_per_minibatch))
    return {"sequence_chunks":chunks,"sequences_per_minibatch":sequences_per_minibatch,
            "recurrent_minibatches_per_epoch":minibatches_per_epoch,
            "optimizer_steps":minibatches_per_epoch*ppo_epochs}

def recurrent_alive_mean(values,alive_mask,valid_time_mask):
    mask=alive_mask*valid_time_mask[...,None]
    return (values*mask).sum()/mask.sum().clamp_min(1.0)

__all__=["ModularRolloutBatch","contiguous_chunks","recurrent_batch_plan","recurrent_alive_mean"]
