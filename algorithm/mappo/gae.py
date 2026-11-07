"""Masked generalized advantage estimation shared by the rollout and PPO."""
from dataclasses import dataclass
import torch

def finite_selected(name, value, mask):
    if not torch.isfinite(value[mask]).all():
        raise FloatingPointError(f'nonfinite {name} in valid samples')
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
