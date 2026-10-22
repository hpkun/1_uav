"""MAPPO with individual centralized values, NaN-safe GAE and exact joint budgets."""
from dataclasses import dataclass
from contextlib import nullcontext
import math
import numpy as np
import torch
from .config import TrainConfig, ProtocolTrainConfig
from .networks import Actor, Critic, ProtocolCritic
from algorithm.common.protocol import ALGORITHM_VERSION, PROTOCOL
from algorithm.common.runtime import require_cuda
from algorithm.common.metrics import COMPONENTS, target_distances


def finite_selected(name, value, mask):
    if not torch.isfinite(value[mask]).all():
        raise FloatingPointError(f'nonfinite {name} in valid samples')


@torch.no_grad()
def standardize_advantages(adv, slots, *, per_agent):
    """Existing population-std+1e-8 rule, including singleton -> zero."""
    if not per_agent:
        return (adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
    adv = adv.clone()
    for slot in slots.unique():
        selected = slots == slot
        adv[selected] = (adv[selected]-adv[selected].mean())/(adv[selected].std(unbiased=False)+1e-8)
    return adv


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
    if rewards.ndim != 3 or terminated.shape != rewards.shape[:-1] or truncated.shape != rewards.shape[:-1]:
        raise ValueError('GAE requires [T,E,N] samples and [T,E] boundaries')
    if terminated.dtype != torch.bool or truncated.dtype != torch.bool:
        raise ValueError('GAE boundaries must be boolean')
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


def validate_rollout_shape(rollout, state_dim):
    """Validate batch axes/protocol, including the no-valid early-return path."""
    obs = rollout.observations
    if obs.ndim != 4 or obs.shape[-1] != 12 or not (obs.shape[0] and obs.shape[1] and 1 <= obs.shape[2] <= 5):
        raise ValueError('rollout requires nonempty [T,E,N<=5,12] observations')
    shape = obs.shape[:-1]
    if rollout.states.shape != (*shape[:2], state_dim):
        raise ValueError(f'rollout requires matching {state_dim}D centralized states')
    if rollout.actions.shape != (*shape, 3):
        raise ValueError('rollout action axes disagree with observations')
    for name in ('log_probs','values','next_values','rewards','current_masks','next_masks'):
        if getattr(rollout,name).shape != shape:
            raise ValueError(f'rollout {name} axes disagree with observations')


class MAPPOTrainer:
    algorithm_version = ALGORITHM_VERSION
    protocol = PROTOCOL

    def __init__(self, config=None, *, device=None):
        if isinstance(config, ProtocolTrainConfig):
            raise TypeError('12/60 configuration requires the independent ProtocolMAPPOTrainer')
        self._initialize(config or TrainConfig(), device, Critic)

    def _initialize(self, config, device, critic_class):
        self.config = config
        self.device = require_cuda(device or 'cuda')
        torch.set_num_threads(self.config.torch_threads)
        torch.manual_seed(self.config.seed)
        self.actor = Actor(self.config.initialization_mode,self.config.beta_concentration,self.config.output_weight_gain).to(self.device)
        self.critic = critic_class().to(self.device)
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
        totals = dict(valid_uav_steps=0, gamma_clips=0, speed_clips=0, integration_failures=0, assignment_switches=0, distance_sum=0.,distance_count=0,distance_reduction=0.)
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
                for key in ('gamma_clips','speed_clips','integration_failures'): totals[key] += info['numerical_delta'][key]
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
            numerical_events={k:totals[k] for k in ('gamma_clips','speed_clips','integration_failures')},
            assignment_switches=totals['assignment_switches'],mean_target_distance=totals['distance_sum']/max(1,totals['distance_count']),
            target_distance_reduction=totals['distance_reduction'],reward_components_mean={k:v/denominator for k,v in self.rollout_components.items()},
            beta_action_mean=acts.mean(0).tolist() if len(acts) else [0.]*3,beta_action_std=acts.std(0).tolist() if len(acts) else [0.]*3)
        return Rollout(*(torch.stack([r[i] for r in rows]) for i in range(11))),episodes

    def update(self, rollout):
        return self._update(rollout, 90, clean_global, .5, False)

    def _update(self, rollout, state_dim, clean_state, value_scale, per_agent, diagnostics=False):
        validate_rollout_shape(rollout, state_dim)
        c = self.config
        with torch.no_grad():
            advantages,returns = gae(rollout.rewards,rollout.values,rollout.next_values,rollout.current_masks,
                                    rollout.next_masks,rollout.terminated,rollout.truncated,c.gamma,c.gae_lambda)
        n = rollout.observations.shape[-2]
        valid = rollout.current_masks.flatten().bool()
        diagnostic = None
        if diagnostics:
            from .diagnostics import UpdateObserver, advantage_statistics
            all_slots = torch.arange(n,device=self.device).expand_as(rollout.current_masks).flatten()[valid]
            counts = torch.bincount(all_slots,minlength=n).tolist()
            diagnostic = dict(per_uav_valid_samples=counts, active_uav_count=sum(v>0 for v in counts),
                              advantage_before=advantage_statistics(advantages.flatten()[valid],all_slots,n),
                              singleton_uavs=[i for i,v in enumerate(counts) if v==1],
                              normalized=c.normalize_advantages,
                              excluded_invalid_positions=int((~valid).sum()), invalid_samples_in_loss=0)
        if not valid.any():
            result = dict(policy_loss=0.,actor_total_loss=0.,actor_loss=0.,critic_loss=0.,entropy=0.,approx_kl=0.,clip_fraction=0.,gradient_norm=0.,
                        valid_samples=0,optimizer_steps=0,update=self.updates)
            if diagnostic is not None:
                diagnostic.update(status='skipped_no_valid_samples', advantage_after=diagnostic['advantage_before'],
                                  **UpdateObserver(self.actor,self.critic,n,c.max_grad_norm).report())
                result['numerical_diagnostics'] = diagnostic
            return result
        obs = rollout.observations.reshape(-1,12)[valid]
        actions = rollout.actions.reshape(-1,3)[valid]
        old_log = rollout.log_probs.flatten()[valid]
        joint_valid = rollout.current_masks.bool().any(-1)
        # Invalid global states in padding must not enter a critic forward.
        state_clean = torch.zeros_like(rollout.states)
        state_clean[joint_valid] = clean_state(rollout.states[joint_valid],rollout.current_masks[joint_valid])
        states = state_clean.unsqueeze(-2).expand(-1,-1,n,-1).reshape(-1,state_dim)[valid]
        slots = torch.arange(n,device=self.device).expand_as(rollout.current_masks).flatten()[valid]
        adv,target = advantages.flatten()[valid],returns.flatten()[valid]
        for name,value in [('observation',obs),('global state',states),('action',actions),('old log probability',old_log),('return',target)]:
            if not torch.isfinite(value).all(): raise FloatingPointError(f'nonfinite valid {name}')
        if not ((actions>0)&(actions<1)).all(): raise ValueError('PPO actions must be inside (0,1)')
        if c.normalize_advantages:
            adv = standardize_advantages(adv, slots, per_agent=per_agent)
        observer = None
        if diagnostic is not None:
            diagnostic.update(status='ok', advantage_after=advantage_statistics(adv,slots,n),
                              valid_inputs_finite=True,
                              beta_actions=dict(finite=bool(torch.isfinite(actions).all()),min=actions.min().item(),max=actions.max().item()))
            observer = UpdateObserver(self.actor,self.critic,n,c.max_grad_norm)
        metrics=[]
        with observer if observer is not None else nullcontext():
            for _ in range(c.ppo_epochs):
                for idx in torch.randperm(len(obs),device=self.device).split(c.minibatch_size):
                    logp,entropy = self.actor.evaluate(obs[idx],actions[idx])
                    log_ratio = logp-old_log[idx]
                    ratio = log_ratio.exp()
                    def reduce_valid(samples):
                        if not per_agent:
                            return samples.mean()
                        # Eq.(26) defines L_i separately. Sum the valid-agent means
                        # for shared parameters; this aggregation is an assumption.
                        return torch.stack([samples[slots[idx] == slot].mean() for slot in slots[idx].unique()]).sum()
                    policy = -reduce_valid(torch.minimum(ratio*adv[idx],ratio.clamp(1-c.clip,1+c.clip)*adv[idx]))
                    entropy_reduced = reduce_valid(entropy)
                    actor_loss = policy-c.entropy_coef*entropy_reduced
                    values = self.critic(states[idx]).gather(1,slots[idx,None]).squeeze(1)
                    critic_loss = value_scale*reduce_valid((values-target[idx]).square())
                    if not torch.isfinite(actor_loss+critic_loss): raise FloatingPointError('nonfinite PPO loss')
                    self.actor_optimizer.zero_grad(set_to_none=True); actor_loss.backward()
                    actor_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(),c.max_grad_norm,error_if_nonfinite=True)
                    self.actor_optimizer.step()
                    self.critic_optimizer.zero_grad(set_to_none=True); critic_loss.backward()
                    if per_agent and isinstance(self.critic, ProtocolCritic):
                        present = set(slots[idx].tolist())
                        for slot, head in enumerate(self.critic.value_heads):
                            if slot not in present:
                                for parameter in head.parameters():
                                    parameter.grad = None
                    critic_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(),c.max_grad_norm,error_if_nonfinite=True)
                    self.critic_optimizer.step()
                    if observer is not None:
                        observer.minibatch(slots[idx],actor_norm,critic_norm)
                    metrics.append([policy.item(),actor_loss.item(),critic_loss.item(),entropy_reduced.item(),((ratio-1)-log_ratio).mean().item(),
                                    ((ratio-1).abs()>c.clip).float().mean().item(),math.hypot(actor_norm.item(),critic_norm.item())])
        self.updates += 1
        result=dict(zip(('policy_loss','actor_total_loss','critic_loss','entropy','approx_kl','clip_fraction','gradient_norm'),np.mean(metrics,axis=0).tolist()))
        result['actor_loss']=result['actor_total_loss']  # Compatibility alias; old logs used policy_loss here.
        result.update(valid_samples=len(obs),optimizer_steps=len(metrics),update=self.updates)
        if diagnostic is not None:
            diagnostic.update(observer.report())
            result['numerical_diagnostics'] = diagnostic
        return result


def clean_protocol_state(states, masks):
    if states.shape[-1] != 60 or not 1 <= masks.shape[-1] <= 5 or states.shape[:-1] != masks.shape[:-1]:
        raise ValueError('12/60 protocol state/mask dimensions mismatch')
    full = torch.zeros((*masks.shape[:-1], 5), dtype=torch.bool, device=masks.device)
    full[..., :masks.shape[-1]] = masks.bool()
    slots = states.reshape(*states.shape[:-1], 5, 12)
    slots = torch.where(full.unsqueeze(-1), slots, torch.zeros_like(slots))
    if not torch.isfinite(slots).all() or not torch.equal(slots[..., 0], full.to(states.dtype)):
        raise ValueError('protocol state requires finite valid slots with flag=1')
    return slots.flatten(-2)


class ProtocolMAPPOTrainer(MAPPOTrainer):
    """Executable generic 12/60 optimizer; accepts externally supplied Rollout.

    No simulator collector, target/geometry construction or reward computation.
    This is not a complete Xu task/training reproduction.
    """
    from .paper_protocol import ALGORITHM_VERSION as algorithm_version, PROTOCOL as protocol

    def __init__(self, config=None, *, device=None):
        if config is not None and not isinstance(config, ProtocolTrainConfig):
            raise TypeError('12/60 updater requires ProtocolTrainConfig')
        self._initialize(config or ProtocolTrainConfig(), device, ProtocolCritic)

    def collect(self, *args, **kwargs):
        raise RuntimeError('12/60 protocol has no task/reward collector; supply explicit Rollout arrays')

    @torch.no_grad()
    def sample_actions(self, observations, masks, *, deterministic=False):
        """Return the SAME interior Beta actions used for stored logp.

        Externally supplied raw slots only; invalid slots return zero/no logp.
        Clamping and Beta initialization are explicit engineering assumptions.
        """
        obs, m = self.tensor(observations), self.tensor(masks)
        if obs.shape[:-1] != m.shape or obs.shape[-1] != 12 or not (((m == 0) | (m == 1)).all()):
            raise ValueError('invalid actor protocol dimensions/masks')
        clean = torch.where(m.bool().unsqueeze(-1), obs, torch.zeros_like(obs))
        if not torch.equal(clean[..., 0], m):
            raise ValueError('actor slot flags disagree with current masks')
        actions, _ = self.actor.act(clean, deterministic=deterministic)
        actions = actions.clamp(self.config.beta_epsilon, 1-self.config.beta_epsilon)
        logp, _ = self.actor.evaluate(clean, actions)
        return (torch.where(m.bool().unsqueeze(-1), actions, torch.zeros_like(actions)),
                torch.where(m.bool(), logp, torch.zeros_like(logp)))

    def update(self, rollout, *, diagnostics=True):
        try:
            validate_rollout_shape(rollout,60)
            obs, masks = rollout.observations, rollout.current_masks
            states = clean_protocol_state(rollout.states, masks)
            local = states.reshape(*states.shape[:-1], 5, 12)[..., :obs.shape[-2], :]
            if not torch.equal(obs[masks.bool()], local[masks.bool()]):
                raise ValueError('actor observations and explicit centralized slots disagree')
            result = self._update(rollout, 60, clean_protocol_state, 1., True, diagnostics)
            if diagnostics:
                self.last_update_diagnostics = result['numerical_diagnostics']
            return result
        except (ValueError, FloatingPointError, RuntimeError) as exc:
            if diagnostics:
                self.last_update_diagnostics = dict(status='rejected',error=type(exc).__name__,reason=str(exc))
            raise
