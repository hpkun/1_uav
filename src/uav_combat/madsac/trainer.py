"""Mask-correct Equations (18)-(21) MADSAC optimization and resume state."""
from __future__ import annotations
import copy
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn
from .actor import SharedSquashedGaussianActor
from .attention_critic import AttentionCritic
from .replay_buffer import ReplayBuffer


def soft_update(target:nn.Module,source:nn.Module,tau:float)->None:
    with torch.no_grad():
        for tp,sp in zip(target.parameters(),source.parameters()): tp.mul_(1-tau).add_(sp,alpha=tau)


def masked_mean(values:torch.Tensor,mask:torch.Tensor)->torch.Tensor:
    return (values*mask).sum()/mask.sum().clamp_min(1.0)


class MADSACTrainer:
    def __init__(self,observation_dim:int=45,action_dim:int=3,num_agents:int=4,hidden_dim:int=256,attention_heads:int=2,learning_rate:float=1e-4,gamma:float=.99,tau:float=.001,alpha:float=.1,policy_delay:int=2,replay_capacity:int=1_000_000,batch_size:int=1024,device:str="cpu",seed:int=0,actor_activation:str="relu",critic_activation:str="leaky_relu",log_std_min:float=-5,log_std_max:float=2,config_signature:str="")->None:
        self.device=torch.device(device); self.gamma,self.tau,self.alpha=gamma,tau,alpha; self.policy_delay,self.batch_size=policy_delay,batch_size
        self.rng=np.random.default_rng(seed); torch.manual_seed(seed)
        if self.device.type=="cuda":
            if not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
            torch.cuda.manual_seed_all(seed)
        self.actor=SharedSquashedGaussianActor(observation_dim,action_dim,hidden_dim,log_std_min,log_std_max,actor_activation).to(self.device)
        self.critic1=AttentionCritic(observation_dim,action_dim,hidden_dim,attention_heads,critic_activation).to(self.device)
        self.critic2=AttentionCritic(observation_dim,action_dim,hidden_dim,attention_heads,critic_activation).to(self.device)
        self.target_actor=copy.deepcopy(self.actor).eval(); self.target_critic1=copy.deepcopy(self.critic1).eval(); self.target_critic2=copy.deepcopy(self.critic2).eval()
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=learning_rate); self.critic1_optimizer=torch.optim.Adam(self.critic1.parameters(),lr=learning_rate); self.critic2_optimizer=torch.optim.Adam(self.critic2.parameters(),lr=learning_rate)
        self.replay=ReplayBuffer(replay_capacity,num_agents,observation_dim,action_dim)
        self.update_count=self.actor_update_count=self.target_update_count=0
        self.sampled_env_steps=self.vector_steps=0; self.episode_counters:dict[str,int]={}; self.evaluation_history:list[dict[str,Any]]=[]
        self.run_metadata:dict[str,Any]={"seed":seed}; self.config_signature=config_signature

    @torch.no_grad()
    def act(self,observations:np.ndarray,alive_mask:np.ndarray|None=None,deterministic:bool=False)->np.ndarray:
        tensor=torch.as_tensor(observations,dtype=torch.float32,device=self.device)
        actions=self.actor.deterministic(tensor) if deterministic else self.actor.sample(tensor)[0]
        if alive_mask is not None: actions=actions*torch.as_tensor(alive_mask,dtype=torch.float32,device=self.device).unsqueeze(-1)
        return actions.cpu().numpy()

    def compute_target(self,batch:dict[str,torch.Tensor])->torch.Tensor:
        with torch.no_grad():
            mask=batch["next_alive_masks"]
            next_actions,next_log_probs=self.target_actor.sample(batch["next_observations"]); next_actions=next_actions*mask.unsqueeze(-1); next_log_probs=next_log_probs*mask
            q=torch.minimum(self.target_critic1(batch["next_observations"],next_actions,mask),self.target_critic2(batch["next_observations"],next_actions,mask))
            bootstrap=mask*(q-self.alpha*next_log_probs)
            return batch["rewards"]+self.gamma*(1-batch["dones"])*bootstrap

    def update(self,batch_size:int|None=None)->dict[str,float|bool|None]:
        batch=self.replay.sample(batch_size or self.batch_size,self.rng,self.device); alive=batch["alive_masks"]; target=self.compute_target(batch)
        q1=self.critic1(batch["observations"],batch["actions"],alive); q2=self.critic2(batch["observations"],batch["actions"],alive)
        q1_loss=masked_mean((q1-target).square(),alive); q2_loss=masked_mean((q2-target).square(),alive)
        self.critic1_optimizer.zero_grad(); q1_loss.backward(); self.critic1_optimizer.step(); self.critic2_optimizer.zero_grad(); q2_loss.backward(); self.critic2_optimizer.step(); self.update_count+=1
        actor_loss_value=None; actor_updated=self.update_count%self.policy_delay==0
        if actor_updated:
            for critic in (self.critic1,self.critic2): critic.requires_grad_(False)
            actions,log_prob=self.actor.sample(batch["observations"]); actions=actions*alive.unsqueeze(-1); log_prob=log_prob*alive
            min_q=torch.minimum(self.critic1(batch["observations"],actions,alive),self.critic2(batch["observations"],actions,alive))
            actor_loss=masked_mean(self.alpha*log_prob-min_q,alive); self.actor_optimizer.zero_grad(); actor_loss.backward(); self.actor_optimizer.step(); actor_loss_value=float(actor_loss.detach())
            for critic in (self.critic1,self.critic2): critic.requires_grad_(True)
            soft_update(self.target_actor,self.actor,self.tau); soft_update(self.target_critic1,self.critic1,self.tau); soft_update(self.target_critic2,self.critic2,self.tau); self.actor_update_count+=1; self.target_update_count+=1
        with torch.no_grad():
            _,logp=self.actor.sample(batch["observations"]); entropy=float(masked_mean(-logp,alive)); _,weights=self.critic1(batch["observations"],batch["actions"],alive,return_attention=True)
            attention_entropy=-(weights.clamp_min(1e-12)*weights.clamp_min(1e-12).log()).sum(-1).mean(1); mean_attention_entropy=float(masked_mean(attention_entropy,alive))
        metrics={"Q1_loss":float(q1_loss.detach()),"Q2_loss":float(q2_loss.detach()),"actor_loss":actor_loss_value,"entropy":entropy,"mean_Q1":float(masked_mean(q1,alive).detach()),"mean_Q2":float(masked_mean(q2,alive).detach()),"mean_min_Q":float(masked_mean(torch.minimum(q1,q2),alive).detach()),"mean_attention_entropy":mean_attention_entropy,"alive_sample_fraction":float(alive.mean()),"replay_size":float(self.replay.size),"actor_updated":actor_updated,"target_updated":actor_updated}
        if not np.all(np.isfinite([v for v in metrics.values() if isinstance(v,float)])): raise FloatingPointError(f"non-finite MADSAC update: {metrics}")
        return metrics

    def checkpoint_state(self,include_replay:bool=True)->dict[str,Any]:
        return {"actor":self.actor.state_dict(),"critic1":self.critic1.state_dict(),"critic2":self.critic2.state_dict(),"target_actor":self.target_actor.state_dict(),"target_critic1":self.target_critic1.state_dict(),"target_critic2":self.target_critic2.state_dict(),"actor_optimizer":self.actor_optimizer.state_dict(),"critic1_optimizer":self.critic1_optimizer.state_dict(),"critic2_optimizer":self.critic2_optimizer.state_dict(),"updates":self.update_count,"actor_updates":self.actor_update_count,"target_updates":self.target_update_count,"sampled_env_steps":self.sampled_env_steps,"vector_steps":self.vector_steps,"episode_counters":self.episode_counters,"evaluation_history":self.evaluation_history,"run_metadata":self.run_metadata,"config_signature":self.config_signature,"numpy_rng_state":self.rng.bit_generator.state,"torch_cpu_rng_state":torch.get_rng_state(),"torch_cuda_rng_state":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,"replay":self.replay.state_dict() if include_replay else None}

    def save(self,path:str|Path,include_replay:bool=True)->None:
        Path(path).parent.mkdir(parents=True,exist_ok=True); torch.save(self.checkpoint_state(include_replay),path)

    def load(self,path:str|Path,require_replay:bool=False)->None:
        state=torch.load(path,map_location=self.device,weights_only=False)
        if self.config_signature and state.get("config_signature") and self.config_signature!=state["config_signature"]: raise RuntimeError("checkpoint config signature mismatch")
        for key,module in (("actor",self.actor),("critic1",self.critic1),("critic2",self.critic2),("target_actor",self.target_actor),("target_critic1",self.target_critic1),("target_critic2",self.target_critic2)): module.load_state_dict(state[key])
        for key,opt in (("actor_optimizer",self.actor_optimizer),("critic1_optimizer",self.critic1_optimizer),("critic2_optimizer",self.critic2_optimizer)): opt.load_state_dict(state[key])
        self.update_count=int(state["updates"]); self.actor_update_count=int(state["actor_updates"]); self.target_update_count=int(state["target_updates"]); self.sampled_env_steps=int(state.get("sampled_env_steps",0)); self.vector_steps=int(state.get("vector_steps",0)); self.episode_counters=dict(state.get("episode_counters",{})); self.evaluation_history=list(state.get("evaluation_history",[])); self.run_metadata=dict(state.get("run_metadata",{}))
        self.rng.bit_generator.state=state["numpy_rng_state"]; torch.set_rng_state(state["torch_cpu_rng_state"])
        if self.device.type=="cuda" and state.get("torch_cuda_rng_state") is not None: torch.cuda.set_rng_state_all(state["torch_cuda_rng_state"])
        if state.get("replay") is not None: self.replay.load_state_dict(state["replay"])
        elif require_replay: raise RuntimeError("full resume checkpoint has no replay state")


__all__=["MADSACTrainer","masked_mean","soft_update"]
