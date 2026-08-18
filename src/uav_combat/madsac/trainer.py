"""MADSAC optimization for Equations (18)-(21)."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn

from .actor import SharedSquashedGaussianActor
from .attention_critic import AttentionCritic
from .replay_buffer import ReplayBuffer


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_parameter, source_parameter in zip(target.parameters(), source.parameters()):
            target_parameter.mul_(1.0 - tau).add_(source_parameter, alpha=tau)


def masked_slot_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over alive slots for diagnostics, not an Eq. (19)/(20) loss."""
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def batch_mean_agent_sum(values: torch.Tensor, alive_mask: torch.Tensor) -> torch.Tensor:
    """Eq. (19)/(20): mean over replay batch after summing alive agents."""
    if values.ndim != 2 or values.shape != alive_mask.shape:
        raise ValueError("values and alive_mask must have the same [batch, agents] shape")
    if values.shape[0] == 0:
        raise ValueError("objective batch must not be empty")
    return (values * alive_mask).sum(dim=1).mean()


def joint_actions_with_own_gradient(actions: torch.Tensor, agent_index: int) -> torch.Tensor:
    """Keep only agent ``i``'s action path differentiable for Eq. (21)."""
    if actions.ndim != 3 or not 0 <= agent_index < actions.shape[1]:
        raise ValueError("actions must be [batch, agents, action_dim] with a valid agent index")
    detached = actions.detach()
    own_mask = torch.zeros_like(actions)
    own_mask[:, agent_index, :] = 1.0
    return detached + own_mask * (actions - detached)


class MADSACTrainer:
    """Shared actor, double centralized attention critics, and target networks."""

    def __init__(
        self,
        observation_dim: int = 54,
        action_dim: int = 3,
        num_agents: int = 4,
        hidden_dim: int = 256,
        attention_heads: int = 2,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        tau: float = 0.001,
        alpha: float = 0.1,
        replay_capacity: int = 1_000_000,
        batch_size: int = 1024,
        device: str = "cpu",
        seed: int = 0,
        actor_activation: str = "relu",
        critic_activation: str = "leaky_relu",
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        self.gamma, self.tau, self.alpha = float(gamma), float(tau), float(alpha)
        self.batch_size = int(batch_size)
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        self.actor = SharedSquashedGaussianActor(
            observation_dim, action_dim, hidden_dim, log_std_min, log_std_max, actor_activation
        ).to(self.device)
        self.critic1 = AttentionCritic(
            observation_dim, action_dim, hidden_dim, attention_heads, critic_activation
        ).to(self.device)
        self.critic2 = AttentionCritic(
            observation_dim, action_dim, hidden_dim, attention_heads, critic_activation
        ).to(self.device)
        self.target_actor = copy.deepcopy(self.actor).eval()
        self.target_critic1 = copy.deepcopy(self.critic1).eval()
        self.target_critic2 = copy.deepcopy(self.critic2).eval()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=learning_rate)
        self.replay = ReplayBuffer(replay_capacity, num_agents, observation_dim, action_dim)
        self.critic_update_count = 0
        self.actor_update_count = 0
        self.target_update_count = 0
        self.sampled_steps = 0
        self.vector_steps = 0

    @torch.no_grad()
    def act(self, observations: np.ndarray, alive_mask: np.ndarray | None = None, deterministic: bool = False) -> np.ndarray:
        tensor = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        actions = self.actor.deterministic(tensor) if deterministic else self.actor.sample(tensor)[0]
        if alive_mask is not None:
            actions = actions * torch.as_tensor(alive_mask, dtype=torch.float32, device=self.device).unsqueeze(-1)
        return actions.cpu().numpy()

    def compute_target(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Equation (18): target actor, target double-Q minimum, and entropy."""
        with torch.no_grad():
            mask = batch["next_alive_masks"]
            next_actions, next_log_prob = self.target_actor.sample(batch["next_observations"])
            next_actions = next_actions * mask.unsqueeze(-1)
            q_min = torch.minimum(
                self.target_critic1(batch["next_observations"], next_actions, mask),
                self.target_critic2(batch["next_observations"], next_actions, mask),
            )
            bootstrap = mask * (q_min - self.alpha * next_log_prob)
            return batch["rewards"] + self.gamma * (1.0 - batch["dones"]) * bootstrap

    def update_critics(self, batch_size: int | None = None) -> dict[str, float]:
        """Equations (18)-(19): independently update both centralized critics."""
        batch = self.replay.sample(batch_size or self.batch_size, self.rng, self.device)
        alive = batch["alive_masks"]
        target = self.compute_target(batch)
        q1 = self.critic1(batch["observations"], batch["actions"], alive)
        q2 = self.critic2(batch["observations"], batch["actions"], alive)
        q1_loss = batch_mean_agent_sum((q1 - target).square(), alive)
        q2_loss = batch_mean_agent_sum((q2 - target).square(), alive)
        self.critic1_optimizer.zero_grad()
        q1_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad()
        q2_loss.backward()
        self.critic2_optimizer.step()
        self.critic_update_count += 1
        metrics = {
            "critic1_loss": float(q1_loss.detach()),
            "critic2_loss": float(q2_loss.detach()),
            "q_value": float(masked_slot_mean(torch.minimum(q1, q2), alive).detach()),
        }
        self._require_finite(metrics)
        return metrics

    def update_actor(self, batch_size: int | None = None) -> dict[str, float]:
        """Eq. (20)-(21): sum shared-policy gradients through each own action."""
        batch = self.replay.sample(batch_size or self.batch_size, self.rng, self.device)
        alive = batch["alive_masks"]
        for critic in (self.critic1, self.critic2):
            critic.requires_grad_(False)
        try:
            actions, log_prob = self.actor.sample(batch["observations"])
            actions = actions * alive.unsqueeze(-1)
            q_by_agent = []
            for agent_index in range(actions.shape[1]):
                joint_actions = joint_actions_with_own_gradient(actions, agent_index)
                q1_i = self.critic1(batch["observations"], joint_actions, alive)[:, agent_index]
                q2_i = self.critic2(batch["observations"], joint_actions, alive)[:, agent_index]
                q_by_agent.append(torch.minimum(q1_i, q2_i))
            min_q = torch.stack(q_by_agent, dim=1)
            actor_loss = batch_mean_agent_sum(self.alpha * log_prob - min_q, alive)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
        finally:
            for critic in (self.critic1, self.critic2):
                critic.requires_grad_(True)
        self.actor_update_count += 1
        metrics = {
            "actor_loss": float(actor_loss.detach()),
            "entropy": float(masked_slot_mean(-log_prob.detach(), alive)),
        }
        self._require_finite(metrics)
        return metrics

    def update_targets(self) -> None:
        """Algorithm 1 target update, called only in the delayed actor branch."""
        soft_update(self.target_actor, self.actor, self.tau)
        soft_update(self.target_critic1, self.critic1, self.tau)
        soft_update(self.target_critic2, self.critic2, self.tau)
        self.target_update_count += 1

    @staticmethod
    def _require_finite(metrics: dict[str, float]) -> None:
        if not np.all(np.isfinite(list(metrics.values()))):
            raise FloatingPointError(f"non-finite MADSAC update: {metrics}")

    def checkpoint_state(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """Small checkpoint; replay is deliberately not copied or serialized."""
        return {
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "target_critic1": self.target_critic1.state_dict(),
            "target_critic2": self.target_critic2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic1_optimizer": self.critic1_optimizer.state_dict(),
            "critic2_optimizer": self.critic2_optimizer.state_dict(),
            "critic_updates": self.critic_update_count,
            "actor_updates": self.actor_update_count,
            "target_updates": self.target_update_count,
            "sampled_steps": self.sampled_steps,
            "vector_steps": self.vector_steps,
            "extra": extra or {},
        }

    def save(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_state(extra), path)

    def load(self, path: str | Path) -> dict[str, Any]:
        state = torch.load(path, map_location=self.device, weights_only=False)
        for key, module in (
            ("actor", self.actor), ("critic1", self.critic1), ("critic2", self.critic2),
            ("target_actor", self.target_actor), ("target_critic1", self.target_critic1),
            ("target_critic2", self.target_critic2),
        ):
            module.load_state_dict(state[key])
        for key, optimizer in (
            ("actor_optimizer", self.actor_optimizer),
            ("critic1_optimizer", self.critic1_optimizer),
            ("critic2_optimizer", self.critic2_optimizer),
        ):
            optimizer.load_state_dict(state[key])
        self.critic_update_count = int(state.get("critic_updates", 0))
        self.actor_update_count = int(state.get("actor_updates", 0))
        self.target_update_count = int(state.get("target_updates", 0))
        self.sampled_steps = int(state.get("sampled_steps", 0))
        self.vector_steps = int(state.get("vector_steps", 0))
        return dict(state.get("extra", {}))


__all__ = [
    "MADSACTrainer", "batch_mean_agent_sum", "joint_actions_with_own_gradient",
    "masked_slot_mean", "soft_update",
]
