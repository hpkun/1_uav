"""Fixed-size rollout storage for joint-reward scalar-critic HAPPO."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def compute_scalar_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    terminal_values: np.ndarray,
    truncation_bootstrap_mask: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """GAE for scalar team rewards and scalar centralized values."""

    if (
        values.shape != (rewards.shape[0] + 1, rewards.shape[1])
        or terminated.shape != rewards.shape
        or truncated.shape != rewards.shape
        or terminal_values.shape != rewards.shape
        or truncation_bootstrap_mask.shape != rewards.shape
    ):
        raise ValueError("HAPPO scalar GAE shape mismatch")
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = np.zeros(rewards.shape[1], dtype=np.float32)
    for step in range(rewards.shape[0] - 1, -1, -1):
        bootstrap = np.where(
            terminated[step],
            0.0,
            np.where(truncated[step], terminal_values[step] * truncation_bootstrap_mask[step], values[step + 1]),
        )
        delta = rewards[step] + gamma * bootstrap - values[step]
        continuation = (~(terminated[step] | truncated[step])).astype(np.float32)
        gae = delta + gamma * gae_lambda * continuation * gae
        advantages[step] = gae
    return advantages, advantages + values[:-1]


@dataclass
class HAPPORolloutBuffer:
    rollout_length: int
    num_envs: int
    num_agents: int
    obs_dim: int
    state_dim: int
    action_dim: int = 15

    def __post_init__(self) -> None:
        t, e, n = self.rollout_length, self.num_envs, self.num_agents
        self.observations = np.zeros((t + 1, e, n, self.obs_dim), np.float32)
        self.global_states = np.zeros((t + 1, e, self.state_dim), np.float32)
        self.available_action_masks = np.ones((t + 1, e, n, self.action_dim), bool)
        self.alive_masks = np.ones((t + 1, e, n), np.float32)
        self.actions = np.zeros((t, e, n), np.int64)
        self.old_log_probs = np.zeros((t, e, n), np.float32)
        self.team_rewards = np.zeros((t, e), np.float32)
        self.agent_rewards = np.zeros((t, e, n), np.float32)
        self.values = np.zeros((t + 1, e), np.float32)
        self.returns = np.zeros((t, e), np.float32)
        self.advantages = np.zeros((t, e), np.float32)
        self.terminated = np.zeros((t, e), bool)
        self.truncated = np.zeros((t, e), bool)
        self.terminal_values = np.zeros((t, e), np.float32)
        self.truncation_bootstrap_masks = np.zeros((t, e), np.float32)
        self.step = 0

    def set_initial(self, obs: np.ndarray, states: np.ndarray, available: np.ndarray, alive: np.ndarray) -> None:
        self.observations[0] = obs
        self.global_states[0] = states
        self.available_action_masks[0] = available
        self.alive_masks[0] = alive

    def insert(
        self,
        actions: np.ndarray,
        log_probs: np.ndarray,
        values: np.ndarray,
        team_rewards: np.ndarray,
        agent_rewards: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        alive_masks: np.ndarray,
        next_obs: np.ndarray,
        next_states: np.ndarray,
        next_available: np.ndarray,
        next_alive: np.ndarray,
        terminal_values: np.ndarray,
        truncation_bootstrap_mask: np.ndarray,
    ) -> None:
        if self.step >= self.rollout_length:
            raise RuntimeError("HAPPO rollout buffer is full")
        i = self.step
        self.actions[i] = actions
        self.old_log_probs[i] = log_probs
        self.values[i] = values
        self.team_rewards[i] = team_rewards
        self.agent_rewards[i] = agent_rewards
        self.terminated[i] = terminated
        self.truncated[i] = truncated
        self.alive_masks[i] = alive_masks
        self.observations[i + 1] = next_obs
        self.global_states[i + 1] = next_states
        self.available_action_masks[i + 1] = next_available
        self.alive_masks[i + 1] = next_alive
        self.terminal_values[i] = terminal_values
        self.truncation_bootstrap_masks[i] = truncation_bootstrap_mask
        self.step += 1

    def finish(self, last_values: np.ndarray, gamma: float, gae_lambda: float) -> None:
        self.values[-1] = last_values
        self.advantages, self.returns = compute_scalar_gae(
            self.team_rewards,
            self.values,
            self.terminated,
            self.truncated,
            self.terminal_values,
            self.truncation_bootstrap_masks,
            gamma,
            gae_lambda,
        )
