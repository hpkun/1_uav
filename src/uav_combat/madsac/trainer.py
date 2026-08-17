"""Equations (18)-(21) MADSAC optimization."""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .actor import SharedSquashedGaussianActor
from .attention_critic import AttentionCritic
from .replay_buffer import ReplayBuffer


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()): tp.mul_(1.0 - tau).add_(sp, alpha=tau)


class MADSACTrainer:
    def __init__(self, observation_dim: int = 45, action_dim: int = 3, num_agents: int = 4, hidden_dim: int = 256, attention_heads: int = 2, learning_rate: float = 1e-4, gamma: float = 0.99, tau: float = 0.001, alpha: float = 0.1, policy_delay: int = 2, replay_capacity: int = 1_000_000, batch_size: int = 1024, device: str = "cpu", seed: int = 0) -> None:
        self.device = torch.device(device); self.gamma, self.tau, self.alpha = gamma, tau, alpha
        self.policy_delay, self.batch_size = policy_delay, batch_size
        self.rng = np.random.default_rng(seed); torch.manual_seed(seed)
        self.actor = SharedSquashedGaussianActor(observation_dim, action_dim, hidden_dim).to(self.device)
        self.critic1 = AttentionCritic(observation_dim, action_dim, hidden_dim, attention_heads).to(self.device)
        self.critic2 = AttentionCritic(observation_dim, action_dim, hidden_dim, attention_heads).to(self.device)
        self.target_actor = copy.deepcopy(self.actor).eval()
        self.target_critic1, self.target_critic2 = copy.deepcopy(self.critic1).eval(), copy.deepcopy(self.critic2).eval()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=learning_rate)
        self.replay = ReplayBuffer(replay_capacity, num_agents, observation_dim, action_dim)
        self.update_count = self.actor_update_count = self.target_update_count = 0

    @torch.no_grad()
    def act(self, observations: np.ndarray, deterministic: bool = False) -> np.ndarray:
        tensor = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        return (self.actor.deterministic(tensor) if deterministic else self.actor.sample(tensor)[0]).cpu().numpy()

    def compute_target(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            next_actions, next_log_probs = self.target_actor.sample(batch["next_observations"])
            q = torch.minimum(self.target_critic1(batch["next_observations"], next_actions), self.target_critic2(batch["next_observations"], next_actions))
            return batch["rewards"] + self.gamma * (1.0 - batch["dones"]) * (q - self.alpha * next_log_probs)

    def update(self, batch_size: int | None = None) -> dict[str, float | bool]:
        batch = self.replay.sample(batch_size or self.batch_size, self.rng, self.device); target = self.compute_target(batch)
        q1, q2 = self.critic1(batch["observations"], batch["actions"]), self.critic2(batch["observations"], batch["actions"])
        q1_loss, q2_loss = (q1 - target).square().mean(), (q2 - target).square().mean()
        self.critic1_optimizer.zero_grad(); q1_loss.backward(); self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad(); q2_loss.backward(); self.critic2_optimizer.step(); self.update_count += 1
        actor_loss_value = None; actor_updated = self.update_count % self.policy_delay == 0
        if actor_updated:
            for critic in (self.critic1, self.critic2): critic.requires_grad_(False)
            actions, log_prob = self.actor.sample(batch["observations"])
            min_q = torch.minimum(self.critic1(batch["observations"], actions), self.critic2(batch["observations"], actions))
            actor_loss = (self.alpha * log_prob - min_q).mean()
            self.actor_optimizer.zero_grad(); actor_loss.backward(); self.actor_optimizer.step(); actor_loss_value = float(actor_loss.detach())
            for critic in (self.critic1, self.critic2): critic.requires_grad_(True)
            soft_update(self.target_actor, self.actor, self.tau); soft_update(self.target_critic1, self.critic1, self.tau); soft_update(self.target_critic2, self.critic2, self.tau)
            self.actor_update_count += 1; self.target_update_count += 1
        entropy = float((-self.actor.sample(batch["observations"])[1]).mean().detach())
        metrics = {"Q1_loss": float(q1_loss.detach()), "Q2_loss": float(q2_loss.detach()), "actor_loss": actor_loss_value, "entropy": entropy, "mean_Q1": float(q1.mean().detach()), "mean_Q2": float(q2.mean().detach()), "mean_min_Q": float(torch.minimum(q1, q2).mean().detach()), "replay_size": float(self.replay.size), "actor_updated": actor_updated, "target_updated": actor_updated}
        checked = [v for v in metrics.values() if isinstance(v, float)]
        if not np.all(np.isfinite(checked)): raise FloatingPointError(f"non-finite MADSAC update: {metrics}")
        return metrics

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"actor": self.actor.state_dict(), "critic1": self.critic1.state_dict(), "critic2": self.critic2.state_dict(), "target_actor": self.target_actor.state_dict(), "target_critic1": self.target_critic1.state_dict(), "target_critic2": self.target_critic2.state_dict(), "actor_optimizer": self.actor_optimizer.state_dict(), "critic1_optimizer": self.critic1_optimizer.state_dict(), "critic2_optimizer": self.critic2_optimizer.state_dict(), "updates": self.update_count, "actor_updates": self.actor_update_count, "target_updates": self.target_update_count}, path)

    def load(self, path: str | Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=True)
        for key, module in (("actor", self.actor), ("critic1", self.critic1), ("critic2", self.critic2), ("target_actor", self.target_actor), ("target_critic1", self.target_critic1), ("target_critic2", self.target_critic2)): module.load_state_dict(state[key])
        self.update_count = int(state.get("updates", 0))
        self.actor_update_count = int(state.get("actor_updates", 0)); self.target_update_count = int(state.get("target_updates", 0))
        for key, optimizer in (("actor_optimizer", self.actor_optimizer), ("critic1_optimizer", self.critic1_optimizer), ("critic2_optimizer", self.critic2_optimizer)):
            if key in state: optimizer.load_state_dict(state[key])
