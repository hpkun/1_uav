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
                 context_dim=0, recurrent_hidden_dim=0,
                 entity_attention_config=None):
        self.base_observation_dim=int(observation_dim); self.context_dim=int(context_dim)
        self.recurrent_hidden_dim=int(recurrent_hidden_dim)
        super().__init__(observation_dim+self.context_dim, action_dim, hidden_dim,
                         log_std_min, log_std_max, activation)
        config=dict(entity_attention_config or {})
        self.entity_attention_enabled=bool(config.get("enabled",False))
        self.entity_attention_mode=str(config.get("mode","replacement"))
        self.entity_dim=int(config.get("entity_dim",32));self.entity_attention_heads=int(config.get("attention_heads",2))
        if self.entity_attention_enabled:
            if self.entity_attention_mode not in {"replacement","residual","gated_residual"}:
                raise ValueError("entity attention mode must be replacement, residual, or gated_residual")
            if self.base_observation_dim!=52:raise ValueError("entity attention requires the fixed 52D observation layout")
            if self.context_dim:raise ValueError("entity attention cannot be combined with wave context")
            if self.recurrent_hidden_dim:raise ValueError("entity attention cannot be combined with recurrent memory")
            if hidden_dim!=256:raise ValueError("entity attention fusion hidden dimension must be 256")
            if self.entity_dim!=32 or self.entity_attention_heads!=2:raise ValueError("entity attention v1 requires entity_dim=32 and attention_heads=2")
            encode=lambda size:nn.Sequential(nn.Linear(size,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU())
            self.self_encoder=encode(7)
            self.ally_encoder=encode(6)
            self.enemy_encoder=encode(5)
            self.ally_attention=nn.MultiheadAttention(32,2,batch_first=True)
            self.enemy_attention=nn.MultiheadAttention(32,2,batch_first=True)
            self.entity_fusion=nn.Sequential(nn.Linear(96,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU())
            if self.entity_attention_mode=="replacement":
                # Preserve the V1 topology exactly: no legacy backbone, adapter,
                # or gate parameters are present in replacement checkpoints.
                del self.backbone
            else:
                self.entity_residual_adapter=nn.Linear(256,256)
                nn.init.zeros_(self.entity_residual_adapter.weight)
                nn.init.zeros_(self.entity_residual_adapter.bias)
                if self.entity_attention_mode=="gated_residual":
                    initial_gate=float(config.get("initial_gate",.05))
                    if not 0.<initial_gate<1.:
                        raise ValueError("gated_residual initial_gate must be in (0, 1)")
                    self.initial_entity_gate=initial_gate
                    self.entity_gate=nn.Linear(512,1)
                    nn.init.zeros_(self.entity_gate.weight)
                    nn.init.constant_(self.entity_gate.bias,math.log(initial_gate/(1.-initial_gate)))
        if self.recurrent_hidden_dim:
            self.gru=nn.GRUCell(hidden_dim, self.recurrent_hidden_dim)
            self.mean=nn.Linear(self.recurrent_hidden_dim, action_dim)
            self.log_std=nn.Linear(self.recurrent_hidden_dim, action_dim)

    @staticmethod
    def split_entities(observations):
        """Split the frozen 52D contract; alive flags are excluded from encoders."""
        if observations.shape[-1]!=52:raise ValueError("entity observation must end in 52 features")
        self_features=observations[...,0:7]
        allies=observations[...,7:28].reshape(*observations.shape[:-1],3,7)
        enemies=observations[...,28:52].reshape(*observations.shape[:-1],4,6)
        return self_features,allies[...,:6],allies[...,6],enemies[...,:5],enemies[...,5]

    @staticmethod
    def _masked_entity_attention(query,entities,alive,module,prefix):
        flat_q=query.reshape(-1,1,query.shape[-1]);flat_e=entities.reshape(-1,entities.shape[-2],entities.shape[-1])
        flat_alive=alive.reshape(-1,alive.shape[-1])>.5;any_alive=flat_alive.any(-1)
        safe_mask=~flat_alive
        if torch.any(~any_alive):
            safe_mask=safe_mask.clone();safe_mask[~any_alive,0]=False
        context,weights=module(flat_q,flat_e,flat_e,key_padding_mask=safe_mask,
                               need_weights=True,average_attn_weights=False)
        valid=flat_alive[:,None,None,:].to(weights.dtype)
        weights=weights*valid
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-12)
        context=context*any_alive[:,None,None].to(context.dtype)
        context=context.reshape(*prefix,context.shape[-1])
        weights=weights.squeeze(-2).reshape(*prefix,weights.shape[1],weights.shape[-1])
        return context,weights

    def _entity_encode(self,observations):
        prefix=observations.shape[:-1]
        own,ally_features,ally_alive,enemy_features,enemy_alive=self.split_entities(observations)
        h_self=self.self_encoder(own);h_ally=self.ally_encoder(ally_features);h_enemy=self.enemy_encoder(enemy_features)
        c_ally,ally_weights=self._masked_entity_attention(h_self,h_ally,ally_alive,self.ally_attention,prefix)
        c_enemy,enemy_weights=self._masked_entity_attention(h_self,h_enemy,enemy_alive,self.enemy_attention,prefix)
        encoded=self.entity_fusion(torch.cat((h_self,c_ally,c_enemy),-1))
        diagnostics={"ally_attention_weights":ally_weights,"enemy_attention_weights":enemy_weights,
                     "ally_entity_alive":ally_alive,"enemy_entity_alive":enemy_alive}
        return encoded,diagnostics

    def _input(self, obs, context):
        if not self.context_dim:return obs
        if context is None: raise ValueError("actor wave context is required")
        if context.ndim==obs.ndim-1: context=context.unsqueeze(-2).expand(*obs.shape[:-1],-1)
        return torch.cat((obs,context),-1)

    def distribution_step(self, observations, context=None, hidden=None, episode_mask=None, alive_mask=None,return_attention=False):
        diagnostics=None
        if self.entity_attention_enabled:
            h_entity,diagnostics=self._entity_encode(observations)
            diagnostics["entity_mode"]=self.entity_attention_mode
            diagnostics["entity_feature_norm"]=torch.linalg.vector_norm(h_entity,dim=-1)
            if self.entity_attention_mode=="replacement":
                encoded=h_entity
            else:
                h_base=self.backbone(self._input(observations,context))
                delta=self.entity_residual_adapter(h_entity)
                base_norm=torch.linalg.vector_norm(h_base,dim=-1)
                delta_norm=torch.linalg.vector_norm(delta,dim=-1)
                diagnostics.update({"entity_base_feature_norm":base_norm,"entity_delta_norm":delta_norm,
                                    "entity_delta_to_base_ratio":delta_norm/base_norm.clamp_min(1e-12)})
                if self.entity_attention_mode=="gated_residual":
                    gate=torch.sigmoid(self.entity_gate(torch.cat((h_base,h_entity),-1)))
                    diagnostics["entity_gate"]=gate.squeeze(-1)
                    encoded=h_base+gate*delta
                else:
                    encoded=h_base+delta
        else:encoded=self.backbone(self._input(observations,context))
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
        result=(Normal(mean,std),new_hidden)
        return (*result,diagnostics) if return_attention else result

    def distribution(self, observations, context=None,return_attention=False):
        if self.recurrent_hidden_dim: raise RuntimeError("recurrent actor requires distribution_step")
        result=self.distribution_step(observations,context,return_attention=return_attention)
        return (result[0],result[2]) if return_attention else result[0]


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
