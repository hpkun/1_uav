"""Standard masked feed-forward MAPPO/PPO updater and testable loss primitives."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from uav_env.algorithms.mappo.networks import CentralizedCritic, SharedActor
from uav_env.algorithms.mappo.rollout_buffer import RolloutBuffer
from uav_env.algorithms.mappo.value_normalizer import ValueNormalizer


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return a finite mask-weighted mean, including for an all-zero mask."""

    weights = mask.to(dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def normalize_masked_advantages(advantages: torch.Tensor, mask: torch.Tensor, epsilon: float = 1.0e-8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize using active samples while retaining all entries for Critic bookkeeping."""

    active = mask.bool()
    if not active.any():
        zero = torch.zeros((), dtype=advantages.dtype, device=advantages.device)
        return advantages.clone(), zero, zero
    mean = advantages[active].mean()
    std = advantages[active].std(unbiased=False)
    return (advantages - mean) / (std + epsilon), mean, std


def ppo_policy_loss(
    new_log_prob: torch.Tensor, old_log_prob: torch.Tensor, advantage: torch.Tensor,
    active_mask: torch.Tensor, clip_param: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the clipped PPO objective and diagnostics on active Actor samples."""

    ratio = torch.exp(new_log_prob - old_log_prob)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
    policy_loss = -masked_mean(torch.minimum(ratio * advantage, clipped_ratio * advantage), active_mask)
    clip_fraction = masked_mean(((ratio - 1.0).abs() > clip_param).to(ratio.dtype), active_mask)
    approx_kl = masked_mean(old_log_prob - new_log_prob, active_mask)
    return policy_loss, ratio, clipped_ratio, clip_fraction, approx_kl


def _pointwise_value_loss(error: torch.Tensor, use_huber_loss: bool, huber_delta: float) -> torch.Tensor:
    if not use_huber_loss:
        return 0.5 * error.square()
    absolute = error.abs()
    return torch.where(absolute <= huber_delta, 0.5 * error.square(), huber_delta * (absolute - 0.5 * huber_delta))


def huber_loss(error: torch.Tensor, delta: float) -> torch.Tensor:
    """Backward-compatible public pointwise Huber helper."""

    return _pointwise_value_loss(error, True, delta)


def ppo_value_loss(
    new_value: torch.Tensor, old_value: torch.Tensor, target: torch.Tensor,
    critic_mask: torch.Tensor, value_clip_param: float, use_clipped_value_loss: bool,
    use_huber_loss: bool, huber_delta: float,
) -> torch.Tensor:
    """Compute masked PPO value loss with independently effective clipping/loss switches."""

    loss = _pointwise_value_loss(new_value - target, use_huber_loss, huber_delta)
    if use_clipped_value_loss:
        clipped = old_value + torch.clamp(new_value - old_value, -value_clip_param, value_clip_param)
        clipped_loss = _pointwise_value_loss(clipped - target, use_huber_loss, huber_delta)
        loss = torch.maximum(loss, clipped_loss)
    return masked_mean(loss, critic_mask)


def compute_approx_kl(new_log_prob: torch.Tensor, old_log_prob: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return masked_mean(old_log_prob - new_log_prob, mask)


def compute_clip_fraction(ratio: torch.Tensor, mask: torch.Tensor, clip_param: float) -> torch.Tensor:
    return masked_mean(((ratio - 1.0).abs() > clip_param).to(ratio.dtype), mask)


def explained_variance(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Return explained variance in the caller's (physical return) space."""

    if mask is not None:
        selected = mask.bool()
        prediction, target = prediction[selected], target[selected]
    if target.numel() == 0:
        return torch.zeros((), dtype=prediction.dtype, device=prediction.device)
    variance = torch.var(target, unbiased=False)
    if variance <= 1.0e-12:
        return torch.zeros((), dtype=prediction.dtype, device=prediction.device)
    return 1.0 - torch.var(target - prediction, unbiased=False) / variance


def value_loss_inputs(
    new_physical: torch.Tensor,
    old_physical: torch.Tensor,
    target_physical: torch.Tensor,
    normalizer: ValueNormalizer,
    use_value_normalization: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Put physical new/old/target values in one loss space using one snapshot."""

    if not use_value_normalization:
        return new_physical, old_physical, target_physical
    return (
        normalizer.normalize(new_physical),
        normalizer.normalize(old_physical),
        normalizer.normalize(target_physical),
    )


class MAPPOTrainer:
    def __init__(self, actor: SharedActor, critic: CentralizedCritic, config: dict[str, Any], normalizer: ValueNormalizer, device: torch.device) -> None:
        self.actor, self.critic, self.config, self.normalizer, self.device = actor, critic, config, normalizer, device
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=float(config["actor_lr"]))
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=float(config["critic_lr"]))
        self.minibatch_rng = np.random.default_rng(int(config.get("seed", 0)) + 104729)

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        c = self.config
        obs = torch.as_tensor(buffer.observations[:-1], device=self.device)
        states = torch.as_tensor(buffer.global_states[:-1], device=self.device)
        actions = torch.as_tensor(buffer.actions, device=self.device)
        available = torch.as_tensor(buffer.available_action_masks[:-1], device=self.device)
        old_log = torch.as_tensor(buffer.old_log_probs, device=self.device)
        old_values = torch.as_tensor(buffer.values[:-1], device=self.device)
        returns = torch.as_tensor(buffer.returns, device=self.device)
        advantages = torch.as_tensor(buffer.advantages, device=self.device)
        actor_mask = torch.as_tensor(buffer.actor_active_masks, device=self.device)
        critic_mask = torch.as_tensor(buffer.critic_masks, device=self.device)
        normalized_advantages, adv_mean, adv_std = normalize_masked_advantages(advantages, actor_mask)
        if c.get("normalize_advantages", True):
            advantages = normalized_advantages
        if c.get("use_value_normalization", True):
            self.normalizer.update(returns[critic_mask.bool()])
            _, norm_old_values, norm_returns = value_loss_inputs(
                old_values, old_values, returns, self.normalizer, True,
            )
        else:
            norm_returns, norm_old_values = returns, old_values
        total = actions.numel()
        mini_batches = int(c["num_mini_batches"])
        if mini_batches <= 0 or mini_batches > total:
            raise ValueError(f"num_mini_batches must be in [1, {total}]")
        records: list[list[float]] = []
        for _ in range(int(c["ppo_epochs"])):
            indices = self.minibatch_rng.permutation(total)
            for batch in np.array_split(indices, mini_batches):
                idx = torch.as_tensor(batch, device=self.device)
                ob = obs.reshape(-1, obs.shape[-1])[idx]
                av = available.reshape(-1, 15)[idx]
                ac = actions.reshape(-1)[idx]
                ol = old_log.reshape(-1)[idx]
                ad = advantages.reshape(-1)[idx]
                am = actor_mask.reshape(-1)[idx]
                st = states[:, :, None, :].expand(-1, -1, buffer.num_agents, -1).reshape(-1, states.shape[-1])[idx]
                ov = norm_old_values.reshape(-1)[idx]
                target = norm_returns.reshape(-1)[idx]
                cm = critic_mask.reshape(-1)[idx]
                agent_ids = torch.arange(buffer.num_agents, device=self.device).repeat(buffer.rollout_length * buffer.num_envs)[idx]

                dist = Categorical(logits=self.actor(ob, av))
                new_log = dist.log_prob(ac)
                policy_loss, ratio, _, clip_fraction, approx_kl = ppo_policy_loss(new_log, ol, ad, am, float(c["clip_param"]))
                entropy = masked_mean(dist.entropy(), am)
                if bool(am.sum() > 0):
                    self.actor_optimizer.zero_grad()
                    (policy_loss - float(c["entropy_coef"]) * entropy).backward()
                    actor_grad = nn.utils.clip_grad_norm_(self.actor.parameters(), float(c["max_grad_norm"]))
                    self.actor_optimizer.step()
                else:
                    actor_grad = torch.zeros(())

                all_values = self.critic(st)
                new_physical = all_values.gather(-1, agent_ids[:, None]).squeeze(-1)
                new_value, _, _ = value_loss_inputs(
                    new_physical,
                    old_values.reshape(-1)[idx],
                    returns.reshape(-1)[idx],
                    self.normalizer,
                    bool(c.get("use_value_normalization", True)),
                )
                value_loss = ppo_value_loss(
                    new_value, ov, target, cm, float(c["value_clip_param"]),
                    bool(c.get("use_clipped_value_loss", True)), bool(c.get("use_huber_loss", True)),
                    float(c["huber_delta"]),
                )
                self.critic_optimizer.zero_grad()
                (float(c["value_loss_coef"]) * value_loss).backward()
                critic_grad = nn.utils.clip_grad_norm_(self.critic.parameters(), float(c["max_grad_norm"]))
                self.critic_optimizer.step()
                records.append([
                    policy_loss.item(), value_loss.item(), entropy.item(), approx_kl.item(), clip_fraction.item(),
                    float(actor_grad), float(critic_grad), masked_mean(ratio, am).item(),
                ])
        values = np.asarray(records)
        names = ["policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "actor_grad_norm", "critic_grad_norm", "ratio_mean"]
        result = {name: float(values[:, i].mean()) for i, name in enumerate(names)}
        with torch.no_grad():
            predicted = self.critic(states.reshape(-1, states.shape[-1]))
            predicted = predicted.reshape(*states.shape[:2], buffer.num_agents)
            explained = explained_variance(predicted, returns, critic_mask)
        result.update(
            advantage_mean=float(adv_mean), advantage_std=float(adv_std),
            return_mean=float(returns.mean()), return_std=float(returns.std(unbiased=False)),
            normalized_return_mean=float(norm_returns.mean()), normalized_return_std=float(norm_returns.std(unbiased=False)),
            explained_variance=float(explained),
        )
        if not all(np.isfinite(list(result.values()))):
            raise FloatingPointError("Non-finite MAPPO metric")
        return result
