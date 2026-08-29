"""Modular actor/critic; the disabled topology is state-dict compatible with MAPPO."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.distributions import Normal
from algorithm.mappo.networks import SharedMAPPOActor, CentralizedValueCritic


class ModularMAPPOActor(SharedMAPPOActor):
    def __init__(self, observation_dim=52, action_dim=3, hidden_dim=256,
                 log_std_min=-5.0, log_std_max=2.0, activation="relu",
                 context_dim=0, recurrent_hidden_dim=0):
        self.base_observation_dim=int(observation_dim); self.context_dim=int(context_dim)
        self.recurrent_hidden_dim=int(recurrent_hidden_dim)
        super().__init__(observation_dim+self.context_dim, action_dim, hidden_dim,
                         log_std_min, log_std_max, activation)
        if self.recurrent_hidden_dim:
            self.gru=nn.GRUCell(hidden_dim, self.recurrent_hidden_dim)
            self.mean=nn.Linear(self.recurrent_hidden_dim, action_dim)
            self.log_std=nn.Linear(self.recurrent_hidden_dim, action_dim)

    def _input(self, obs, context):
        if not self.context_dim:return obs
        if context is None: raise ValueError("actor wave context is required")
        if context.ndim==obs.ndim-1: context=context.unsqueeze(-2).expand(*obs.shape[:-1],-1)
        return torch.cat((obs,context),-1)

    def distribution_step(self, observations, context=None, hidden=None, episode_mask=None, alive_mask=None):
        encoded=self.backbone(self._input(observations,context))
        new_hidden=hidden
        if self.recurrent_hidden_dim:
            if hidden is None:hidden=torch.zeros(*encoded.shape[:-1],self.recurrent_hidden_dim,device=encoded.device)
            if episode_mask is not None:
                reset=episode_mask[...,None,None] if episode_mask.ndim==hidden.ndim-2 else episode_mask[...,None]
                hidden=hidden*reset
            new_hidden=self.gru(encoded.reshape(-1,encoded.shape[-1]),hidden.reshape(-1,hidden.shape[-1])).view(*encoded.shape[:-1],-1)
            if alive_mask is not None:new_hidden=new_hidden*alive_mask[...,None]
            encoded=new_hidden
        mean=self.mean(encoded); std=self.log_std(encoded).clamp(self.log_std_min,self.log_std_max).exp()
        return Normal(mean,std),new_hidden

    def distribution(self, observations, context=None):
        if self.recurrent_hidden_dim: raise RuntimeError("recurrent actor requires distribution_step")
        return self.distribution_step(observations,context)[0]


class ModularCentralizedCritic(CentralizedValueCritic):
    def __init__(self, observation_dim=52, hidden_dim=256, attention_heads=2,
                 activation="relu", context_dim=0, recurrent_hidden_dim=0):
        self.base_observation_dim=int(observation_dim); self.context_dim=int(context_dim)
        self.recurrent_hidden_dim=int(recurrent_hidden_dim)
        super().__init__(observation_dim+self.context_dim,hidden_dim,attention_heads,activation)
        if self.recurrent_hidden_dim:
            act={"relu":nn.ReLU,"leaky_relu":nn.LeakyReLU}[activation]
            self.value_network=nn.Sequential(nn.Linear(hidden_dim*2,hidden_dim),act())
            self.gru=nn.GRUCell(hidden_dim,self.recurrent_hidden_dim)
            self.value_head=nn.Linear(self.recurrent_hidden_dim,1)

    @property
    def output_layer(self):
        return self.value_head if self.recurrent_hidden_dim else self.value_network[-1]

    def _input(self,obs,context):
        if not self.context_dim:return obs
        if context is None:raise ValueError("critic wave context is required")
        if context.ndim==obs.ndim-1:context=context.unsqueeze(-2).expand(*obs.shape[:-1],-1)
        return torch.cat((obs,context),-1)

    def forward_step(self,observations,alive_mask=None,context=None,hidden=None,episode_mask=None,return_attention=False):
        observations=self._input(observations,context)
        embedding=self.embedding(observations); batch,agents,_=embedding.shape
        def heads(x):return x.view(batch,agents,self.attention_heads,self.head_dim).transpose(1,2)
        q,k,v=heads(self.wq(embedding)),heads(self.wk(embedding)),heads(self.wv(embedding))
        logits=q@k.transpose(-2,-1)/math.sqrt(self.head_dim)
        if alive_mask is None:alive_mask=torch.ones(batch,agents,device=logits.device)
        alive=alive_mask>.5; not_self=~torch.eye(agents,dtype=torch.bool,device=logits.device).view(1,1,agents,agents)
        valid=alive[:,None,None,:].expand(batch,self.attention_heads,agents,agents)&not_self
        weights=torch.softmax(logits.masked_fill(~valid,-1e9),-1)*valid.to(logits.dtype)
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-12)
        weights=weights*alive[:,None,:,None]
        ctx=(weights@v).transpose(1,2).contiguous().view(batch,agents,self.hidden_dim)
        feature=self.value_network(torch.cat((embedding,ctx),-1))
        new_hidden=hidden
        if self.recurrent_hidden_dim:
            if hidden is None:hidden=torch.zeros(batch,agents,self.recurrent_hidden_dim,device=feature.device)
            if episode_mask is not None:
                reset=episode_mask[...,None,None] if episode_mask.ndim==hidden.ndim-2 else episode_mask[...,None]
                hidden=hidden*reset
            new_hidden=self.gru(feature.reshape(-1,feature.shape[-1]),hidden.reshape(-1,hidden.shape[-1])).view(batch,agents,-1)
            new_hidden=new_hidden*alive_mask[...,None]; values=self.value_head(new_hidden).squeeze(-1)
        else: values=feature.squeeze(-1)
        values=values*alive_mask
        return (values,new_hidden,weights) if return_attention else (values,new_hidden)

    def forward(self,observations,alive_mask=None,return_attention=False,context=None):
        if self.recurrent_hidden_dim:raise RuntimeError("recurrent critic requires forward_step")
        values,_,weights=self.forward_step(observations,alive_mask,context,return_attention=True)
        return (values,weights) if return_attention else values


__all__=["ModularMAPPOActor","ModularCentralizedCritic"]
