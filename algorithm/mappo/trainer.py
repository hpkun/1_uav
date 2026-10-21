"""MAPPO with individual centralized values, NaN-safe GAE and exact joint budgets."""
from dataclasses import dataclass
import math
import numpy as np
import torch
from .config import TrainConfig
from .networks import Actor, Critic
from algorithm.common.runtime import require_cuda
from algorithm.common.metrics import COMPONENTS, target_distances


def finite_selected(name, value, mask):
    if not torch.isfinite(value[mask]).all():
        raise FloatingPointError(f'nonfinite {name} in valid samples')


def clean_global(states, masks):
    """Exclude inactive/nonexistent local slots before any critic forward pass."""
    n = masks.shape[-1]
    full = torch.zeros((*masks.shape[:-1],5), device=masks.device, dtype=torch.bool)
    full[...,:n] = masks.bool()
    local = states[...,:60].reshape(*states.shape[:-1],5,12)
    local = torch.where(full.unsqueeze(-1),local,torch.zeros_like(local))
    points = states[...,60:].reshape(*states.shape[:-1],5,6)
    exists = points[...,5] == 1
    # The exists flag itself is protocol data, not a discardable sample value.
    if not torch.isfinite(points[...,5]).all():
        raise FloatingPointError('nonfinite waypoint existence flag')
    if not ((points[...,5]==0)|(points[...,5]==1)).all():
        raise ValueError('invalid waypoint existence flag')
    points = torch.where(exists.unsqueeze(-1),points,torch.zeros_like(points))
    return torch.cat((local.flatten(-2),points.flatten(-2)),dim=-1)


def gae(rewards, values, next_values, current_masks, next_masks, terminated, truncated, gamma, lam):
    if rewards.shape != values.shape or rewards.shape != next_values.shape or rewards.shape != current_masks.shape or rewards.shape != next_masks.shape:
        raise ValueError('GAE tensor dimensions mismatch')
    if not torch.isfinite(current_masks).all() or not torch.isfinite(next_masks).all():
        raise FloatingPointError('nonfinite GAE masks')
    if not (((current_masks==0)|(current_masks==1)).all() and ((next_masks==0)|(next_masks==1)).all()):
        raise ValueError('GAE masks must be binary')
    current, nxt = current_masks.bool(), next_masks.bool()
    bootstrap = current & nxt & (~terminated).unsqueeze(-1)
    continuation = bootstrap & (~truncated).unsqueeze(-1)
    finite_selected('reward', rewards, current)
    finite_selected('value', values, current)
    finite_selected('successor value', next_values, bootstrap)
    r = torch.where(current,rewards,torch.zeros_like(rewards))
    v = torch.where(current,values,torch.zeros_like(values))
    nv = torch.where(bootstrap,next_values,torch.zeros_like(next_values))
    advantages = torch.zeros_like(r)
    accumulator = torch.zeros_like(r[0])
    for t in reversed(range(len(r))):
        delta = r[t] + gamma*nv[t] - v[t]
        carry = torch.where(continuation[t],accumulator,torch.zeros_like(accumulator))
        accumulator = torch.where(current[t],delta+gamma*lam*carry,torch.zeros_like(delta))
        advantages[t] = accumulator
    returns = torch.where(current,advantages+v,torch.zeros_like(v))
    finite_selected('advantage',advantages,current)
    finite_selected('return',returns,current)
    return advantages, returns


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
        self.device = require_cuda(device or 'cuda')
        torch.set_num_threads(self.config.torch_threads)
        torch.manual_seed(self.config.seed)
        self.actor = Actor(self.config.initialization_mode,self.config.beta_concentration,self.config.output_weight_gain).to(self.device)
        self.critic = Critic().to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=self.config.actor_lr,eps=self.config.adam_epsilon)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=self.config.critic_lr,eps=self.config.adam_epsilon)
        self.updates, self.sampled_steps = 0, 0
        self.training_episode_count = 0

    def tensor(self, value, dtype=torch.float32):
        return torch.as_tensor(value,device=self.device,dtype=dtype)

    def collect(self, vector, joint_steps=None):
        budget = self.config.horizon*vector.num_envs if joint_steps is None else joint_steps
        if type(budget) is not int or budget < 1 or budget > self.config.horizon*vector.num_envs:
            raise ValueError('rollout budget outside [1,horizon*num_envs]')
        rows, episodes, e, n = [], [], vector.num_envs, vector.config.num_uavs
        totals = dict(valid_uav_steps=0, gamma_clips=0, speed_clips=0, assignment_switches=0, distance_sum=0.,distance_count=0,distance_reduction=0.)
        self.rollout_components = dict.fromkeys(COMPONENTS,0.)
        executed_actions = []
        remaining = budget
        while remaining:
            ids = np.arange(min(e,remaining))
            obs = self.tensor(vector.observations[ids])
            masks = self.tensor(vector.masks[ids]).bool()
            clean_obs = torch.where(masks.unsqueeze(-1),obs,torch.zeros_like(obs))
            finite_selected('observation',clean_obs,masks)
            states = clean_global(self.tensor(vector.states[ids]),masks)
            with torch.no_grad():
                actions,_ = self.actor.act(clean_obs)
                actions = actions.clamp(self.config.beta_epsilon,1-self.config.beta_epsilon)
                logp,_ = self.actor.evaluate(clean_obs,actions)
                values = self.critic(states)[...,:n]
            transition = vector.step(actions.cpu().numpy(),indices=ids)
            next_masks = self.tensor(transition.next_masks).bool()
            with torch.no_grad():
                next_states = clean_global(self.tensor(transition.transition_states),next_masks)
                next_values = self.critic(next_states)[...,:n]
            # Only the final vector step may be partial. Padded workers never
            # execute a transition, and have zero current masks in the buffer.
            shape = (e,n)
            row = [torch.zeros((*shape,12),device=self.device),torch.zeros((e,90),device=self.device),
                   torch.zeros((*shape,3),device=self.device)]
            row.extend(torch.zeros(shape,device=self.device) for _ in range(6))
            row.extend((torch.ones(e,device=self.device,dtype=torch.bool),torch.zeros(e,device=self.device,dtype=torch.bool)))
            selected = [obs,states,actions,logp,values,next_values,self.tensor(transition.rewards),masks.float(),next_masks.float(),
                        self.tensor(transition.terminated,torch.bool),self.tensor(transition.truncated,torch.bool)]
            for dst,src in zip(row,selected): dst[:len(ids)] = src
            rows.append(row)
            episodes.extend(transition.episodes)
            executed_actions.append(actions[masks].cpu().numpy())
            for info in transition.infos:
                totals['valid_uav_steps'] += int(info['current_active_mask'].sum())
                for key in ('gamma_clips','speed_clips'): totals[key] += info['numerical_delta'][key]
                for key in COMPONENTS: self.rollout_components[key] += float(info['reward_components'][key].sum())
                totals['assignment_switches'] += int(((info['transition_assignments'] != info['assignments']) & info['next_active_mask']).sum())
                before,after = target_distances(info,previous=True),target_distances(info)
                totals['distance_sum'] += float(after.sum()); totals['distance_count'] += len(after)
                totals['distance_reduction'] += float((before-after).sum())
            remaining -= len(ids)
        self.sampled_steps += budget
        self.training_episode_count += len(episodes)
        denominator = max(1,totals['valid_uav_steps'])
        acts = np.concatenate(executed_actions,axis=0)
        self.rollout_metrics = dict(sampled_steps=budget, valid_uav_steps=totals['valid_uav_steps'],
            gamma_clip_rate=totals['gamma_clips']/denominator,speed_clip_rate=totals['speed_clips']/denominator,
            numerical_events={k:totals[k] for k in ('gamma_clips','speed_clips')},
            assignment_switches=totals['assignment_switches'],mean_target_distance=totals['distance_sum']/max(1,totals['distance_count']),
            target_distance_reduction=totals['distance_reduction'],reward_components_mean={k:v/denominator for k,v in self.rollout_components.items()},
            beta_action_mean=acts.mean(0).tolist() if len(acts) else [0.]*3,beta_action_std=acts.std(0).tolist() if len(acts) else [0.]*3)
        return Rollout(*(torch.stack([r[i] for r in rows]) for i in range(11))),episodes

    def update(self, rollout):
        c = self.config
        with torch.no_grad():
            advantages,returns = gae(rollout.rewards,rollout.values,rollout.next_values,rollout.current_masks,
                                    rollout.next_masks,rollout.terminated,rollout.truncated,c.gamma,c.gae_lambda)
        n = rollout.observations.shape[-2]
        valid = rollout.current_masks.flatten().bool()
        if not valid.any():
            return dict(actor_loss=0.,critic_loss=0.,entropy=0.,approx_kl=0.,clip_fraction=0.,gradient_norm=0.,
                        valid_samples=0,optimizer_steps=0,update=self.updates)
        obs = rollout.observations.reshape(-1,12)[valid]
        actions = rollout.actions.reshape(-1,3)[valid]
        old_log = rollout.log_probs.flatten()[valid]
        joint_valid = rollout.current_masks.bool().any(-1)
        # Invalid global states in padding must not enter a critic forward.
        state_clean = torch.zeros_like(rollout.states)
        state_clean[joint_valid] = clean_global(rollout.states[joint_valid],rollout.current_masks[joint_valid])
        states = state_clean.unsqueeze(-2).expand(-1,-1,n,-1).reshape(-1,90)[valid]
        slots = torch.arange(n,device=self.device).expand_as(rollout.current_masks).flatten()[valid]
        adv,target = advantages.flatten()[valid],returns.flatten()[valid]
        for name,value in [('observation',obs),('global state',states),('action',actions),('old log probability',old_log),('return',target)]:
            if not torch.isfinite(value).all(): raise FloatingPointError(f'nonfinite valid {name}')
        if not ((actions>0)&(actions<1)).all(): raise ValueError('PPO actions must be inside (0,1)')
        if c.normalize_advantages: adv = (adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
        metrics=[]
        for _ in range(c.ppo_epochs):
            for idx in torch.randperm(len(obs),device=self.device).split(c.minibatch_size):
                logp,entropy = self.actor.evaluate(obs[idx],actions[idx])
                log_ratio = logp-old_log[idx]
                ratio = log_ratio.exp()
                policy = -torch.minimum(ratio*adv[idx],ratio.clamp(1-c.clip,1+c.clip)*adv[idx]).mean()
                actor_loss = policy-c.entropy_coef*entropy.mean()
                values = self.critic(states[idx]).gather(1,slots[idx,None]).squeeze(1)
                critic_loss = .5*(values-target[idx]).square().mean()
                if not torch.isfinite(actor_loss+critic_loss): raise FloatingPointError('nonfinite PPO loss')
                self.actor_optimizer.zero_grad(set_to_none=True); actor_loss.backward()
                actor_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(),c.max_grad_norm,error_if_nonfinite=True)
                self.actor_optimizer.step()
                self.critic_optimizer.zero_grad(set_to_none=True); critic_loss.backward()
                critic_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(),c.max_grad_norm,error_if_nonfinite=True)
                self.critic_optimizer.step()
                metrics.append([policy.item(),critic_loss.item(),entropy.mean().item(),((ratio-1)-log_ratio).mean().item(),
                                ((ratio-1).abs()>c.clip).float().mean().item(),math.hypot(actor_norm.item(),critic_norm.item())])
        self.updates += 1
        result=dict(zip(('actor_loss','critic_loss','entropy','approx_kl','clip_fraction','gradient_norm'),np.mean(metrics,axis=0).tolist()))
        result.update(valid_samples=len(obs),optimizer_steps=len(metrics),update=self.updates)
        return result
