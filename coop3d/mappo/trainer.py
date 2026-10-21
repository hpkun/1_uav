"""MAPPO, per-agent returns with shared centralized scalar state value."""
from dataclasses import dataclass
import numpy as np
import torch
from .config import TrainConfig
from .networks import Actor, Critic
from ..observation import centralized


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is mandatory for training and evaluation; CPU fallback is disabled')
    return torch.device('cuda')


def gae(rewards, values, next_values, current_masks, next_masks, terminated, truncated, gamma, lam):
    """Timeout bootstraps terminal observation, but never recurses into reset episodes."""
    advantages = torch.zeros_like(rewards)
    accumulator = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        bootstrap = (~terminated[t]).unsqueeze(-1) * next_masks[t]
        continuation = (~(terminated[t] | truncated[t])).unsqueeze(-1) * next_masks[t]
        delta = current_masks[t] * (rewards[t] + gamma * bootstrap * next_values[t] - values[t])
        accumulator = current_masks[t] * (delta + gamma * lam * continuation * accumulator)
        advantages[t] = accumulator
    return advantages, (advantages + values) * current_masks


@dataclass
class Rollout:
    observations: torch.Tensor
    states: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    next_values: torch.Tensor
    rewards: torch.Tensor
    current_masks: torch.Tensor
    next_masks: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor


class MAPPOTrainer:
    def __init__(self, config=None, *, device=None):
        self.config = config or TrainConfig()
        # CPU is exposed only for network/unit tests. Executable workflows require CUDA.
        self.device = torch.device(device) if device is not None else require_cuda()
        torch.set_num_threads(self.config.torch_threads)
        torch.manual_seed(self.config.seed)
        self.actor, self.critic = Actor().to(self.device), Critic().to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr, eps=self.config.adam_epsilon)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_lr, eps=self.config.adam_epsilon)
        self.updates = 0

    def tensor(self, a, dtype=torch.float32):
        return torch.as_tensor(a, device=self.device, dtype=dtype)

    def collect(self, vector):
        rows, summaries = [], []
        self.rollout_components = dict.fromkeys(('progress', 'completion', 'separation', 'efficiency', 'terminal'), 0.0)
        self.rollout_counters = dict(speed_clips=0, gamma_clips=0, nonfinite_states=0)
        for _ in range(self.config.horizon):
            observations = self.tensor(vector.observations)
            states = self.tensor(centralized(vector.observations))
            with torch.no_grad():
                actions, _ = self.actor.act(observations)
                actions = actions.clamp(self.config.beta_epsilon, 1 - self.config.beta_epsilon)
                log_probs, _ = self.actor.evaluate(observations, actions)
                values = self.critic(states).unsqueeze(-1).expand(*observations.shape[:2])
            transition = vector.step(actions.cpu().numpy())
            for info in transition.infos:
                for key, value in info['reward_components'].items():
                    self.rollout_components[key] += float(value.sum())
                for key in self.rollout_counters:
                    self.rollout_counters[key] = max(self.rollout_counters[key], info['numerical_counters'][key])
            with torch.no_grad():
                next_values = self.critic(self.tensor(centralized(transition.transition_observations))).unsqueeze(-1).expand_as(values)
            rows.append((observations, states, actions, log_probs, values, next_values,
                         self.tensor(transition.rewards), self.tensor(transition.current_masks), self.tensor(transition.next_masks),
                         self.tensor(transition.terminated, torch.bool), self.tensor(transition.truncated, torch.bool)))
            summaries.extend(info for info, done in zip(transition.infos, transition.terminated | transition.truncated) if done)
        return Rollout(*(torch.stack([r[i] for r in rows]) for i in range(11))), summaries

    def update(self, rollout):
        c = self.config
        with torch.no_grad():
            advantages, returns = gae(rollout.rewards, rollout.values, rollout.next_values, rollout.current_masks,
                                      rollout.next_masks, rollout.terminated, rollout.truncated, c.gamma, c.gae_lambda)
        n = rollout.observations.shape[-2]
        valid = rollout.current_masks.reshape(-1).bool()
        if not valid.any():
            return dict(actor_loss=0.0, critic_loss=0.0, entropy=0.0, valid_samples=0, optimizer_steps=0)
        obs = rollout.observations.reshape(-1, 12)[valid]
        states = rollout.states.unsqueeze(-2).expand(-1, -1, n, -1).reshape(-1, 60)[valid]
        actions = rollout.actions.reshape(-1, 3)[valid]
        old_log = rollout.log_probs.reshape(-1)[valid]
        adv, target = advantages.reshape(-1)[valid], returns.reshape(-1)[valid]
        if c.normalize_advantages:
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        metrics = []
        for _ in range(c.epochs):
            indices = torch.randperm(len(obs), device=self.device)
            for idx in indices.split(c.minibatch_size):
                log_probs, entropy = self.actor.evaluate(obs[idx], actions[idx])
                ratio = (log_probs - old_log[idx]).exp()
                policy = -torch.minimum(ratio * adv[idx], ratio.clamp(1 - c.clip, 1 + c.clip) * adv[idx]).mean()
                actor_loss = policy - c.entropy_coef * entropy.mean()
                critic_loss = .5 * (self.critic(states[idx]) - target[idx]).square().mean()
                if not torch.isfinite(actor_loss + critic_loss):
                    raise FloatingPointError('nonfinite PPO loss')
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), c.max_grad_norm, error_if_nonfinite=True)
                self.actor_optimizer.step()
                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), c.max_grad_norm, error_if_nonfinite=True)
                self.critic_optimizer.step()
                metrics.append([policy.item(), critic_loss.item(), entropy.mean().item(), (old_log[idx] - log_probs).mean().item()])
        self.updates += 1
        result = dict(zip(('actor_loss', 'critic_loss', 'entropy', 'approx_kl'), np.mean(metrics, axis=0).tolist()))
        result.update(valid_samples=len(obs), optimizer_steps=len(metrics), update=self.updates)
        return result
