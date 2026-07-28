"""Sequential HAPPO updater with cumulative probability-ratio factor."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from uav_env.algorithms.happo.networks import IndependentActorSet, JointCentralizedCritic
from uav_env.algorithms.happo.rollout_buffer import HAPPORolloutBuffer
from uav_env.algorithms.mappo.trainer import (
    explained_variance,
    masked_mean,
    normalize_masked_advantages,
    ppo_value_loss,
    value_loss_inputs,
)
from uav_env.algorithms.mappo.value_normalizer import ValueNormalizer


def happo_policy_loss(
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantage: torch.Tensor,
    active_mask: torch.Tensor,
    factor: torch.Tensor,
    clip_param: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Equation (11)-style HAPPO clipped surrogate for one agent.

    ``factor`` is the detached product of previous agents' updated policy
    ratios and is not clipped. The current agent still uses PPO clipping.
    """

    ratio = torch.exp(new_log_prob - old_log_prob)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
    objective = factor.detach() * torch.minimum(ratio * advantage, clipped_ratio * advantage)
    loss = -masked_mean(objective, active_mask)
    clip_fraction = masked_mean(((ratio - 1.0).abs() > clip_param).to(ratio.dtype), active_mask)
    approx_kl = masked_mean(old_log_prob - new_log_prob, active_mask)
    return loss, ratio, clip_fraction, approx_kl


def update_happo_factor(
    factor: torch.Tensor,
    updated_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    """Multiply factor by an updated actor ratio, using ratio 1 for inactive rows."""

    ratio = torch.exp(updated_log_prob - old_log_prob)
    ratio_for_factor = torch.where(active_mask.bool(), ratio, torch.ones_like(ratio))
    return (factor * ratio_for_factor).detach()


def flatten_agent(data: np.ndarray | torch.Tensor, agent_id: int) -> torch.Tensor:
    """Flatten [T,E,N,...] data for one agent into [T*E,...]."""

    tensor = data if isinstance(data, torch.Tensor) else torch.as_tensor(data)
    return tensor[:, :, int(agent_id)].reshape(-1, *tensor.shape[3:])


class HAPPOTrainer:
    """HAPPO trainer with independent actor optimizers and one scalar critic."""

    def __init__(
        self,
        actors: IndependentActorSet,
        critic: JointCentralizedCritic,
        config: dict[str, Any],
        normalizer: ValueNormalizer,
        device: torch.device,
    ) -> None:
        self.actors = actors
        self.critic = critic
        self.config = config
        self.normalizer = normalizer
        self.device = device
        self.actor_optimizers = [
            torch.optim.Adam(actor.parameters(), lr=float(config["actor_lr"]))
            for actor in actors.actors
        ]
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=float(config["critic_lr"]))
        seed = int(config.get("seed", 0))
        self.order_rng = np.random.default_rng(seed + 271_828)
        self.actor_minibatch_rngs = [np.random.default_rng(seed + 104_729 + i * 997) for i in range(len(actors))]
        self.critic_minibatch_rng = np.random.default_rng(seed + 314_159)

    def next_update_order(self) -> list[int]:
        """Return fixed or reproducible random agent update order."""

        order = list(range(len(self.actors)))
        if not bool(self.config.get("fixed_agent_order", False)):
            order = self.order_rng.permutation(order).astype(int).tolist()
        return order

    def _actor_full_log_probs(self, buffer: HAPPORolloutBuffer, agent_id: int) -> torch.Tensor:
        actor = self.actors[int(agent_id)]
        obs = torch.as_tensor(buffer.observations[:-1, :, agent_id], device=self.device)
        available = torch.as_tensor(buffer.available_action_masks[:-1, :, agent_id], device=self.device)
        actions = torch.as_tensor(buffer.actions[:, :, agent_id], device=self.device)
        dist = Categorical(logits=actor(obs.reshape(-1, obs.shape[-1]), available.reshape(-1, available.shape[-1])))
        return dist.log_prob(actions.reshape(-1)).reshape(buffer.rollout_length, buffer.num_envs)

    def _update_actor(
        self,
        buffer: HAPPORolloutBuffer,
        agent_id: int,
        factor: torch.Tensor,
        advantages: torch.Tensor,
    ) -> dict[str, float]:
        c = self.config
        obs = torch.as_tensor(buffer.observations[:-1, :, agent_id], device=self.device).reshape(-1, buffer.obs_dim)
        available = torch.as_tensor(buffer.available_action_masks[:-1, :, agent_id], device=self.device).reshape(-1, buffer.action_dim)
        actions = torch.as_tensor(buffer.actions[:, :, agent_id], device=self.device).reshape(-1)
        old_log = torch.as_tensor(buffer.old_log_probs[:, :, agent_id], device=self.device).reshape(-1)
        active = torch.as_tensor(buffer.alive_masks[:-1, :, agent_id], device=self.device).reshape(-1)
        adv = advantages.reshape(-1)
        factor_flat = factor.reshape(-1)
        total = actions.numel()
        mini_batches = int(c["actor_num_mini_batches"])
        if mini_batches <= 0 or mini_batches > total:
            raise ValueError(f"actor_num_mini_batches must be in [1, {total}]")
        records: list[list[float]] = []
        actor = self.actors[int(agent_id)]
        optimizer = self.actor_optimizers[int(agent_id)]
        for _ in range(int(c["ppo_epochs"])):
            indices = self.actor_minibatch_rngs[int(agent_id)].permutation(total)
            for batch in np.array_split(indices, mini_batches):
                idx = torch.as_tensor(batch, device=self.device)
                dist = Categorical(logits=actor(obs[idx], available[idx]))
                new_log = dist.log_prob(actions[idx])
                loss, ratio, clip_fraction, approx_kl = happo_policy_loss(
                    new_log, old_log[idx], adv[idx], active[idx], factor_flat[idx], float(c["clip_param"])
                )
                entropy = masked_mean(dist.entropy(), active[idx])
                if bool(active[idx].sum() > 0):
                    optimizer.zero_grad()
                    (loss - float(c["entropy_coef"]) * entropy).backward()
                    grad_norm = nn.utils.clip_grad_norm_(actor.parameters(), float(c["max_grad_norm"]))
                    optimizer.step()
                else:
                    grad_norm = torch.zeros((), device=self.device)
                records.append([loss.item(), entropy.item(), approx_kl.item(), clip_fraction.item(), masked_mean(ratio, active[idx]).item(), float(grad_norm)])
        values = np.asarray(records, dtype=np.float64)
        return {
            f"actor_{agent_id}_policy_loss": float(values[:, 0].mean()),
            f"actor_{agent_id}_entropy": float(values[:, 1].mean()),
            f"actor_{agent_id}_approx_kl": float(values[:, 2].mean()),
            f"actor_{agent_id}_clip_fraction": float(values[:, 3].mean()),
            f"actor_{agent_id}_ratio_mean": float(values[:, 4].mean()),
            f"actor_{agent_id}_grad_norm": float(values[:, 5].mean()),
        }

    def _update_critic(self, buffer: HAPPORolloutBuffer) -> dict[str, float]:
        c = self.config
        states = torch.as_tensor(buffer.global_states[:-1], device=self.device).reshape(-1, buffer.state_dim)
        old_values_physical = torch.as_tensor(buffer.values[:-1], device=self.device).reshape(-1)
        returns_physical = torch.as_tensor(buffer.returns, device=self.device).reshape(-1)
        mask = torch.ones_like(returns_physical)
        if bool(c.get("use_value_normalization", True)):
            self.normalizer.update(returns_physical)
            _, old_values, returns = value_loss_inputs(old_values_physical, old_values_physical, returns_physical, self.normalizer, True)
        else:
            old_values, returns = old_values_physical, returns_physical
        total = states.shape[0]
        mini_batches = int(c["critic_num_mini_batches"])
        records: list[list[float]] = []
        for _ in range(int(c["critic_epochs"])):
            indices = self.critic_minibatch_rng.permutation(total)
            for batch in np.array_split(indices, mini_batches):
                idx = torch.as_tensor(batch, device=self.device)
                new_physical = self.critic(states[idx])
                new_value, _, _ = value_loss_inputs(
                    new_physical, old_values_physical[idx], returns_physical[idx],
                    self.normalizer, bool(c.get("use_value_normalization", True)),
                )
                loss = ppo_value_loss(
                    new_value, old_values[idx], returns[idx], mask[idx], float(c["value_clip_param"]),
                    bool(c.get("use_clipped_value_loss", True)), bool(c.get("use_huber_loss", True)),
                    float(c["huber_delta"]),
                )
                self.critic_optimizer.zero_grad()
                (float(c["value_loss_coef"]) * loss).backward()
                grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), float(c["max_grad_norm"]))
                self.critic_optimizer.step()
                records.append([loss.item(), float(grad_norm)])
        arr = np.asarray(records, dtype=np.float64)
        with torch.no_grad():
            predicted = self.critic(states)
            ev = explained_variance(predicted, returns_physical)
        return {
            "critic_value_loss": float(arr[:, 0].mean()),
            "critic_grad_norm": float(arr[:, 1].mean()),
            "explained_variance": float(ev),
            "return_mean": float(returns_physical.mean()),
            "return_std": float(returns_physical.std(unbiased=False)),
        }

    def update(self, buffer: HAPPORolloutBuffer) -> dict[str, float]:
        """Run one HAPPO update: all actors sequentially, then critic."""

        active_any = torch.as_tensor(buffer.alive_masks[:-1].sum(axis=2) > 0, device=self.device)
        advantages = torch.as_tensor(buffer.advantages, device=self.device)
        normalized, adv_mean, adv_std = normalize_masked_advantages(advantages, active_any)
        if bool(self.config.get("normalize_advantages", True)):
            advantages = normalized
        factor = torch.ones((buffer.rollout_length, buffer.num_envs), dtype=torch.float32, device=self.device)
        order = self.next_update_order()
        result: dict[str, float] = {
            "agent_update_order": "-".join(str(i) for i in order),
            "joint_advantage_mean": float(adv_mean),
            "joint_advantage_std": float(adv_std),
        }
        factor_update_count = 0
        for agent_id in order:
            result[f"factor_mean_before_actor_{agent_id}"] = float(factor.mean())
            result[f"factor_std_before_actor_{agent_id}"] = float(factor.std(unbiased=False))
            result[f"factor_min_before_actor_{agent_id}"] = float(factor.min())
            result[f"factor_max_before_actor_{agent_id}"] = float(factor.max())
            result.update(self._update_actor(buffer, int(agent_id), factor, advantages))
            with torch.no_grad():
                new_log = self._actor_full_log_probs(buffer, int(agent_id))
                old_log = torch.as_tensor(buffer.old_log_probs[:, :, agent_id], device=self.device)
                active = torch.as_tensor(buffer.alive_masks[:-1, :, agent_id], device=self.device)
                factor = update_happo_factor(factor, new_log, old_log, active)
                factor_update_count += 1
        result["factor_update_count"] = float(factor_update_count)
        result.update(self._update_critic(buffer))
        if not all(np.isfinite(v) for v in result.values() if isinstance(v, (int, float))):
            raise FloatingPointError("Non-finite HAPPO metric")
        return result
