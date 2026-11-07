"""Shared Beta policy and centralized MAPPO for interception."""
from dataclasses import dataclass
import math
import numpy as np
import torch
from torch import nn
from torch.distributions import Beta
from torch.nn.functional import softplus

from algorithm.common.runtime import require_cuda
from env.interception.config import OBS_DIM, STATE_DIM, MAX_PURSUERS, MAX_EVADERS, SLOT, AXES
from .gae import Rollout, gae


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 17
    workers: int = 4
    horizon: int = 128
    epochs: int = 4
    minibatch: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 5e-4
    gamma: float = .99
    gae_lambda: float = .95
    clip: float = .2
    entropy: float = .01
    max_grad_norm: float = .5

    def __post_init__(self):
        for name in ("workers", "horizon", "epochs", "minibatch"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"invalid {name}")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be a uint32")
        for name in ("actor_lr", "critic_lr", "clip", "max_grad_norm"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"invalid {name}")
        if not (0 < self.gamma <= 1 and 0 <= self.gae_lambda <= 1 and self.clip < 1):
            raise ValueError("invalid PPO coefficients")
        if not math.isfinite(self.entropy) or self.entropy < 0:
            raise ValueError("invalid entropy coefficient")


def network(inputs, outputs, gain):
    layers = nn.Sequential(nn.Linear(inputs, 128), nn.Tanh(), nn.Linear(128, 128),
                           nn.Tanh(), nn.Linear(128, outputs))
    for layer in (layers[0], layers[2]):
        nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
        nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(layers[-1].weight, gain=gain)
    nn.init.zeros_(layers[-1].bias)
    return layers


def clean_central(states):
    slots = states[..., : (MAX_PURSUERS + MAX_EVADERS) * SLOT]
    slots = slots.reshape(*states.shape[:-1], MAX_PURSUERS + MAX_EVADERS, SLOT)
    flags = slots[..., -1:]
    if not ((flags == 0) | (flags == 1)).all():
        raise ValueError("invalid central masks")
    slots = torch.where(flags.bool(), slots, 0.)
    tail = states[..., (MAX_PURSUERS + MAX_EVADERS) * SLOT:]
    return torch.cat((slots.flatten(-2), tail), dim=-1)


class MAPPO:
    def __init__(self, config):
        self.config = config
        self.device = require_cuda()
        torch.set_num_threads(1)
        self.actor = network(OBS_DIM, 2 * AXES, .01).to(self.device)
        self.critic = network(STATE_DIM, MAX_PURSUERS, 1.).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.updates = self.steps = self.episodes = 0

    def tensor(self, array):
        return torch.as_tensor(array, device=self.device)

    def distribution(self, observations):
        alpha, beta = (1 + softplus(self.actor(observations))).chunk(2, dim=-1)
        return Beta(alpha, beta)

    @torch.no_grad()
    def act(self, observations, mask, deterministic=False):
        mask = self.tensor(mask).bool()
        obs = self.tensor(observations).float()
        dist = self.distribution(obs[mask])
        values = dist.mean if deterministic else dist.sample()
        values = values.clamp(1e-6, 1 - 1e-6)
        action = torch.full((*mask.shape, AXES), .5, device=self.device)
        logp = torch.zeros(mask.shape, device=self.device)
        action[mask] = values
        logp[mask] = dist.log_prob(values).sum(-1)
        return action, logp

    @torch.no_grad()
    def collect(self, vector):
        rows, episodes = [], []
        for _ in range(self.config.horizon):
            obs = self.tensor(vector.observations).float()
            states = clean_central(self.tensor(vector.states).float())
            mask = self.tensor(vector.masks).bool()
            action, logp = self.act(obs, mask)
            result = vector.step(action.cpu().numpy())
            final = clean_central(self.tensor(result.transition_states).float())
            rows.append((obs, states, action, logp, self.critic(states), self.critic(final),
                         self.tensor(result.rewards), mask, self.tensor(result.next_masks),
                         self.tensor(result.terminated), self.tensor(result.truncated)))
            episodes.extend(result.episodes)
        return Rollout(*(torch.stack(column) for column in zip(*rows))), episodes

    def update(self, rollout):
        c = self.config
        with torch.no_grad():
            advantage, returns = gae(rollout.rewards, rollout.values, rollout.next_values,
                                     rollout.current_masks, rollout.next_masks, rollout.terminated,
                                     rollout.truncated, c.gamma, c.gae_lambda)
        valid = rollout.current_masks.bool()
        obs, actions = rollout.observations[valid], rollout.actions[valid]
        old_log, advantage, returns = rollout.log_probs[valid], advantage[valid], returns[valid]
        slots = torch.where(valid)[-1]
        states = rollout.states.unsqueeze(-2).expand(*valid.shape, STATE_DIM)[valid]
        states = clean_central(states)
        if not len(obs) or not all(torch.isfinite(t).all() for t in (obs, actions, old_log, states)):
            raise FloatingPointError("invalid PPO batch")
        if not ((actions > 0) & (actions < 1)).all():
            raise ValueError("invalid Beta actions")
        if len(advantage) > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
        metrics = []
        for _ in range(c.epochs):
            for idx in torch.randperm(len(obs), device=self.device).split(c.minibatch):
                dist = self.distribution(obs[idx])
                log_ratio = dist.log_prob(actions[idx]).sum(-1) - old_log[idx]
                ratio = log_ratio.exp()
                policy = -torch.minimum(ratio * advantage[idx],
                                        ratio.clamp(1-c.clip, 1+c.clip) * advantage[idx]).mean()
                entropy = dist.entropy().sum(-1).mean()
                actor_loss = policy - c.entropy * entropy
                critic_loss = (self.critic(states[idx])[torch.arange(len(idx), device=self.device),
                                                        slots[idx]] - returns[idx]).square().mean()
                if not torch.isfinite(actor_loss + critic_loss):
                    raise FloatingPointError("nonfinite PPO loss")
                norms = []
                for model, optimizer, loss in ((self.actor, self.actor_optimizer, actor_loss),
                                               (self.critic, self.critic_optimizer, critic_loss)):
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    norms.append(nn.utils.clip_grad_norm_(model.parameters(), c.max_grad_norm,
                                                         error_if_nonfinite=True))
                    optimizer.step()
                metrics.append([policy.item(), critic_loss.item(), entropy.item(),
                                ((ratio - 1) - log_ratio).mean().item(), actor_loss.item(),
                                ((ratio - 1).abs() > c.clip).float().mean().item(),
                                norms[0].item(), norms[1].item()])
        self.updates += 1
        self.steps += int(np.prod(rollout.terminated.shape))
        return dict(zip(("policy_loss", "critic_loss", "entropy", "approx_kl", "actor_total_loss",
                         "clip_fraction", "actor_grad_norm", "critic_grad_norm"),
                        np.mean(metrics, axis=0).tolist()), valid_samples=len(obs))
