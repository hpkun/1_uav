"""60D adapter; all PPO loss/optimizer mathematics inherited from the audited trainer."""
import numpy as np
import torch
from torch import nn
from algorithm.common.runtime import require_cuda
from algorithm.mappo.networks import Actor, mlp
from algorithm.mappo.trainer import MAPPOTrainer, Rollout, gae
from .config import RegionTrainConfig


def clean_region_state(states,masks):
    if states.shape[-1]!=60: raise ValueError('region state must be 60D')
    full=torch.zeros((*masks.shape[:-1],5),device=masks.device,dtype=torch.bool)
    full[...,:masks.shape[-1]]=masks.bool()
    return torch.where(full.unsqueeze(-1),states.reshape(*states.shape[:-1],5,12),0).flatten(-2)


class RegionCritic(nn.Module):
    def __init__(self):
        super().__init__(); self.network=mlp(60,5)

    def forward(self,states):
        if states.shape[-1]!=60 or not torch.isfinite(states).all(): raise FloatingPointError('finite 60D state required')
        return self.network(states)


class RegionTrainer(MAPPOTrainer):
    state_dim=60
    clean_state=staticmethod(clean_region_state)

    def __init__(self,config=None,*,device='cuda'):
        self.config=config or RegionTrainConfig(); self.device=require_cuda(device)
        c=self.config; torch.set_num_threads(c.torch_threads); torch.manual_seed(c.seed)
        self.actor=Actor(c.initialization_mode,c.beta_concentration,c.output_weight_gain).to(self.device)
        self.critic=RegionCritic().to(self.device)
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=c.actor_lr,eps=c.adam_epsilon)
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=c.critic_lr,eps=c.adam_epsilon)
        self.sampled_steps=self.updates=self.training_episode_count=0

    def estimate_advantages(self,batch):
        c=self.config
        # Physical termination and deadline remain separate in the buffer.
        # BOTH have zero future value for this finite-horizon task.
        value_terminal=batch.terminated | batch.truncated
        return gae(batch.rewards,batch.values,batch.next_values,batch.current_masks,batch.next_masks,
                   value_terminal,batch.truncated,c.gamma,c.gae_lambda)

    def collect(self,vector,joint_steps=None):
        e,n=vector.num_envs,vector.config.num_uavs
        budget=self.config.horizon*e if joint_steps is None else joint_steps
        if type(budget) is not int or not 1<=budget<=self.config.horizon*e: raise ValueError('invalid joint budget')
        rows=[]; episodes=[]; remaining=budget; valid_count=0
        while remaining:
            ids=np.arange(min(e,remaining)); count=len(ids)
            masks=self.tensor(vector.masks[ids]).bool()
            obs=torch.where(masks.unsqueeze(-1),self.tensor(vector.observations[ids]),0)
            states=self.clean_state(self.tensor(vector.states[ids]),masks)
            with torch.no_grad():
                actions,_=self.actor.act(obs)
                actions=actions.clamp(self.config.beta_epsilon,1-self.config.beta_epsilon)
                logp,_=self.actor.evaluate(obs,actions)
                values=self.critic(states)[...,:n]
            transition=vector.step(actions.cpu().numpy(),indices=ids)
            next_mask=self.tensor(transition.next_masks).bool()
            with torch.no_grad():
                successor=self.clean_state(self.tensor(transition.transition_states),next_mask)
                next_values=self.critic(successor)[...,:n]
            selected=[obs,states,actions,logp,values,next_values,self.tensor(transition.rewards),masks.float(),next_mask.float(),
                      self.tensor(transition.terminated,torch.bool),self.tensor(transition.truncated,torch.bool)]
            row=[]
            for index,value in enumerate(selected):
                shape=(e,*value.shape[1:])
                destination=torch.ones(shape,device=self.device,dtype=value.dtype) if index==9 else torch.zeros(shape,device=self.device,dtype=value.dtype)
                destination[:count]=value; row.append(destination)
            rows.append(row); episodes.extend(transition.episodes)
            valid_count+=int(masks.sum().item()); remaining-=count
        self.sampled_steps+=budget; self.training_episode_count+=len(episodes)
        self.rollout_metrics=dict(joint_transitions=budget,valid_uav_samples=valid_count,completed_episodes=len(episodes))
        return Rollout(*(torch.stack([r[k] for r in rows]) for k in range(11))),episodes
