"""Standard masked feed-forward MAPPO/PPO updater."""

from __future__ import annotations

from typing import Any
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from uav_env.algorithms.mappo.networks import CentralizedCritic, SharedActor
from uav_env.algorithms.mappo.rollout_buffer import RolloutBuffer
from uav_env.algorithms.mappo.value_normalizer import ValueNormalizer


def huber_loss(error: torch.Tensor, delta: float) -> torch.Tensor:
    absolute=error.abs(); return torch.where(absolute<=delta,0.5*error.square(),delta*(absolute-0.5*delta))


class MAPPOTrainer:
    def __init__(self, actor: SharedActor, critic: CentralizedCritic, config: dict[str,Any], normalizer: ValueNormalizer, device: torch.device) -> None:
        self.actor,self.critic,self.config,self.normalizer,self.device=actor,critic,config,normalizer,device
        self.actor_optimizer=torch.optim.Adam(actor.parameters(),lr=float(config["actor_lr"])); self.critic_optimizer=torch.optim.Adam(critic.parameters(),lr=float(config["critic_lr"]))

    def update(self, buffer: RolloutBuffer) -> dict[str,float]:
        c=self.config; obs=torch.as_tensor(buffer.observations[:-1],device=self.device); states=torch.as_tensor(buffer.global_states[:-1],device=self.device)
        actions=torch.as_tensor(buffer.actions,device=self.device); available=torch.as_tensor(buffer.available_action_masks[:-1],device=self.device)
        old_log=torch.as_tensor(buffer.old_log_probs,device=self.device); old_values=torch.as_tensor(buffer.values[:-1],device=self.device)
        returns=torch.as_tensor(buffer.returns,device=self.device); advantages=torch.as_tensor(buffer.advantages,device=self.device)
        actor_mask=torch.as_tensor(buffer.actor_active_masks,device=self.device); critic_mask=torch.as_tensor(buffer.critic_masks,device=self.device)
        valid=actor_mask.bool(); adv_mean=advantages[valid].mean() if valid.any() else torch.tensor(0.,device=self.device); adv_std=advantages[valid].std(unbiased=False) if valid.any() else torch.tensor(1.,device=self.device)
        if c.get("normalize_advantages",True): advantages=(advantages-adv_mean)/(adv_std+1e-8)
        if c.get("use_value_normalization",True):
            self.normalizer.update(returns[critic_mask.bool()]); norm_returns=self.normalizer.normalize(returns); norm_old_values=self.normalizer.normalize(old_values)
        else:
            norm_returns=returns;norm_old_values=old_values
        total=actions.numel(); indices=np.arange(total); records=[]
        for _ in range(int(c["ppo_epochs"])):
            np.random.shuffle(indices)
            for batch in np.array_split(indices,int(c["num_mini_batches"])):
                idx=torch.as_tensor(batch,device=self.device); flat=lambda x:x.reshape(-1,*x.shape[3:]) if x.ndim>3 else x.reshape(-1)
                ob=obs.reshape(-1,obs.shape[-1])[idx]; av=available.reshape(-1,15)[idx]; ac=actions.reshape(-1)[idx]; ol=old_log.reshape(-1)[idx]
                ad=advantages.reshape(-1)[idx]; am=actor_mask.reshape(-1)[idx]; st=states[:,:,None,:].expand(-1,-1,buffer.num_agents,-1).reshape(-1,states.shape[-1])[idx]
                ov=norm_old_values.reshape(-1)[idx]; rt=norm_returns.reshape(-1)[idx]; cm=critic_mask.reshape(-1)[idx]
                dist=Categorical(logits=self.actor(ob,av)); new_log=dist.log_prob(ac); ratio=torch.exp(new_log-ol)
                clipped=torch.clamp(ratio,1-float(c["clip_param"]),1+float(c["clip_param"])); denom=am.sum().clamp_min(1)
                policy_loss=-(torch.minimum(ratio*ad,clipped*ad)*am).sum()/denom; entropy=(dist.entropy()*am).sum()/denom
                if am.sum()>0:
                    self.actor_optimizer.zero_grad(); (policy_loss-float(c["entropy_coef"])*entropy).backward(); actor_grad=nn.utils.clip_grad_norm_(self.actor.parameters(),float(c["max_grad_norm"])); self.actor_optimizer.step()
                else: actor_grad=torch.tensor(0.)
                new_value=self.critic(st).diagonal(dim1=-2,dim2=-1) if st.ndim>2 else None
                # Critic receives flattened states; select values by flattened agent identity explicitly.
                ids=torch.arange(buffer.num_agents,device=self.device).repeat(buffer.rollout_length*buffer.num_envs)[idx]
                all_values=self.critic(st); new_value=all_values.gather(-1,ids[:,None]).squeeze(-1)
                target=rt; unclipped=huber_loss(new_value-target,float(c["huber_delta"]))
                clipped_value=ov+torch.clamp(new_value-ov,-float(c["value_clip_param"]),float(c["value_clip_param"])); clipped_loss=huber_loss(clipped_value-target,float(c["huber_delta"]))
                value_loss=(torch.maximum(unclipped,clipped_loss)*cm).sum()/cm.sum().clamp_min(1)
                self.critic_optimizer.zero_grad(); (float(c["value_loss_coef"])*value_loss).backward(); critic_grad=nn.utils.clip_grad_norm_(self.critic.parameters(),float(c["max_grad_norm"])); self.critic_optimizer.step()
                approx_kl=((ol-new_log)*am).sum()/denom
                clip_fraction=((((ratio-1).abs()>float(c["clip_param"])).float()*am).sum()/denom)
                ratio_mean=(ratio*am).sum()/denom
                records.append([policy_loss.item(),value_loss.item(),entropy.item(),approx_kl.item(),clip_fraction.item(),float(actor_grad),float(critic_grad),ratio_mean.item()])
        values=np.asarray(records); names=["policy_loss","value_loss","entropy","approx_kl","clip_fraction","actor_grad_norm","critic_grad_norm","ratio_mean"]
        result={name:float(values[:,i].mean()) for i,name in enumerate(names)}
        with torch.no_grad():
            predicted=self.critic(states.reshape(-1,states.shape[-1]))
            if c.get("use_value_normalization",True): predicted=self.normalizer.denormalize(predicted)
            predicted=predicted.reshape(*states.shape[:2],buffer.num_agents)
            target=returns
            target_variance=torch.var(target,unbiased=False)
            explained=1.0-torch.var(target-predicted,unbiased=False)/target_variance if target_variance>1e-12 else torch.tensor(0.0,device=self.device)
        result.update(
            advantage_mean=float(adv_mean),advantage_std=float(adv_std),
            return_mean=float(returns.mean()),return_std=float(returns.std(unbiased=False)),
            normalized_return_mean=float(norm_returns.mean()),normalized_return_std=float(norm_returns.std(unbiased=False)),
            explained_variance=float(explained),
        )
        if not all(np.isfinite(list(result.values()))): raise FloatingPointError("Non-finite MAPPO metric")
        return result
