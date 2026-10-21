"""Frozen pre-maintenance update, only for parameter-equivalence testing."""
import math
import numpy as np
import torch
from algorithm.mappo.trainer import gae, clean_global

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
