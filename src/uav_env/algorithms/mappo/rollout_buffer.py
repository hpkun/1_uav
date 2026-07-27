"""Fixed-size on-policy MAPPO rollout storage."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class RolloutBuffer:
    rollout_length: int; num_envs: int; num_agents: int; obs_dim: int; state_dim: int

    def __post_init__(self) -> None:
        t,e,a = self.rollout_length,self.num_envs,self.num_agents
        self.observations=np.zeros((t+1,e,a,self.obs_dim),np.float32); self.global_states=np.zeros((t+1,e,self.state_dim),np.float32)
        self.available_action_masks=np.ones((t+1,e,a,15),bool); self.actions=np.zeros((t,e,a),np.int64)
        self.old_log_probs=np.zeros((t,e,a),np.float32); self.values=np.zeros((t+1,e,a),np.float32); self.rewards=np.zeros((t,e,a),np.float32)
        self.terminated=np.zeros((t,e),bool); self.truncated=np.zeros((t,e),bool); self.actor_active_masks=np.ones((t,e,a),np.float32)
        self.critic_masks=np.ones((t,e,a),np.float32); self.terminal_values=np.zeros((t,e,a),np.float32)
        self.truncation_bootstrap_masks=np.zeros((t,e),np.float32)
        self.next_values=np.zeros((t,e,a),np.float32)
        self.advantages=np.zeros((t,e,a),np.float32); self.returns=np.zeros((t,e,a),np.float32); self.step=0

    def set_initial(self, obs, states, available) -> None:
        self.observations[0]=obs; self.global_states[0]=states; self.available_action_masks[0]=available

    def insert(self, actions, log_probs, values, rewards, terminated, truncated, actor_masks, critic_masks,
               next_obs, next_states, next_available, terminal_values, truncation_bootstrap_mask) -> None:
        if self.step >= self.rollout_length: raise RuntimeError("Rollout buffer is full")
        i=self.step; self.actions[i]=actions; self.old_log_probs[i]=log_probs; self.values[i]=values; self.rewards[i]=rewards
        self.terminated[i]=terminated; self.truncated[i]=truncated; self.actor_active_masks[i]=actor_masks; self.critic_masks[i]=critic_masks
        self.observations[i+1]=next_obs; self.global_states[i+1]=next_states; self.available_action_masks[i+1]=next_available
        self.terminal_values[i]=terminal_values; self.truncation_bootstrap_masks[i]=truncation_bootstrap_mask; self.step+=1

    def finish(self, last_values, gamma: float, gae_lambda: float) -> None:
        from uav_env.algorithms.mappo.returns import compute_gae
        self.values[-1]=last_values
        self.next_values[:]=self.values[1:]
        self.next_values[self.terminated]=0.0
        truncated_indices = np.nonzero(self.truncated)
        if truncated_indices[0].size:
            self.next_values[truncated_indices] = (
                self.terminal_values[truncated_indices]
                * self.truncation_bootstrap_masks[truncated_indices][:, None]
            )
        self.advantages,self.returns=compute_gae(self.rewards,self.values,self.terminated,self.truncated,self.terminal_values,self.truncation_bootstrap_masks,gamma,gae_lambda)
