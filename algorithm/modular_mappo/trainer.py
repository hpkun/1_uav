"""MAPPO with opt-in capability modules and a genuine contiguous-sequence GRU path."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from typing import Any
import hashlib,json
import numpy as np
import torch
from torch import nn
from torch.distributions import kl_divergence
from algorithm.mappo.trainer import compute_gae,masked_mean,MAPPO_IMPL_VERSION
from algorithm.modules import (WaveContextModule,RecurrentMemoryModule,PopArtValueNormalizer,
 MultiWaveRewardAdapter,WaveBalancingModule,WarmStartInitializer,CurriculumController,PolicyAnchorRegularizer,enabled_module_names)
from .networks import ModularMAPPOActor,ModularCentralizedCritic
from .buffer import ModularRolloutBatch,contiguous_chunks,recurrent_alive_mean

MODULAR_MAPPO_IMPL_VERSION=1

class ModularMAPPOTrainer:
 def __init__(self,observation_dim=52,action_dim=3,num_agents=4,hidden_dim=256,attention_heads=2,
  actor_learning_rate=3e-4,critic_learning_rate=3e-4,gamma=.99,gae_lambda=.95,clip_ratio=.2,
  value_loss_coefficient=.5,entropy_coefficient=.01,max_grad_norm=.5,ppo_epochs=10,minibatch_size=512,
  normalize_advantages=True,clip_value_loss=True,device="cpu",seed=0,actor_activation="relu",
  critic_activation="relu",log_std_min=-5.,log_std_max=2.,modules_config=None):
  self.device=torch.device(device)
  if self.device.type=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA requested but unavailable")
  self.num_agents=int(num_agents);self.gamma=float(gamma);self.gae_lambda=float(gae_lambda);self.clip_ratio=float(clip_ratio)
  self.value_loss_coefficient=float(value_loss_coefficient);self.entropy_coefficient=float(entropy_coefficient);self.max_grad_norm=float(max_grad_norm)
  self.ppo_epochs=int(ppo_epochs);self.minibatch_size=int(minibatch_size);self.normalize_advantages=bool(normalize_advantages);self.clip_value_loss=bool(clip_value_loss)
  self.rng=np.random.default_rng(seed);torch.manual_seed(seed)
  if self.device.type=="cuda":torch.cuda.manual_seed_all(seed)
  self.modules_config=deepcopy(modules_config or {})
  self.wave_context=WaveContextModule(self.modules_config.get("wave_context"));self.recurrent=RecurrentMemoryModule(self.modules_config.get("recurrent_memory"))
  self.popart=PopArtValueNormalizer(self.modules_config.get("popart")).to(self.device);self.reward_adapter=MultiWaveRewardAdapter(self.modules_config.get("multi_wave_reward"))
  self.wave_balance=WaveBalancingModule(self.modules_config.get("wave_balancing"));self.warm_start=WarmStartInitializer(self.modules_config.get("warm_start"))
  self.curriculum=CurriculumController(self.modules_config.get("curriculum"));self.anchor=PolicyAnchorRegularizer(self.modules_config.get("policy_anchor"))
  ac=self.wave_context.context_dim if self.wave_context.actor_enabled else 0;cc=self.wave_context.context_dim if self.wave_context.critic_enabled else 0
  ar=self.recurrent.hidden_dim if self.recurrent.actor_enabled else 0;cr=self.recurrent.hidden_dim if self.recurrent.critic_enabled else 0
  self.actor=ModularMAPPOActor(observation_dim,action_dim,hidden_dim,log_std_min,log_std_max,actor_activation,ac,ar).to(self.device)
  self.critic=ModularCentralizedCritic(observation_dim,hidden_dim,attention_heads,critic_activation,cc,cr).to(self.device)
  self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=actor_learning_rate);self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=critic_learning_rate)
  self.ppo_update_count=self.actor_update_count=self.critic_update_count=self.sampled_steps=self.vector_steps=0
  self.warm_start_provenance={};self.anchor_provenance={}

 def context_numpy(self,wave,total):return self.wave_context.encode_numpy(wave,total) if self.wave_context.enabled else np.zeros((*np.asarray(wave).shape,0),np.float32)
 def initial_hidden(self,num_envs):return self.recurrent.zeros(num_envs,self.num_agents,True),self.recurrent.zeros(num_envs,self.num_agents,False)
 def _ctx(self,c,actor):
  active=self.wave_context.actor_enabled if actor else self.wave_context.critic_enabled
  return c if active else None
 @torch.no_grad()
 def act(self,observations,alive_mask=None,deterministic=False,return_policy_data=False,context=None,hidden=None,episode_mask=None):
  obs=torch.as_tensor(observations,dtype=torch.float32,device=self.device);mask=torch.as_tensor(alive_mask,dtype=torch.float32,device=self.device) if alive_mask is not None else None
  ctx=torch.as_tensor(context,dtype=torch.float32,device=self.device) if context is not None else None;hid=torch.as_tensor(hidden,dtype=torch.float32,device=self.device) if hidden is not None else None
  ep=torch.as_tensor(episode_mask,dtype=torch.float32,device=self.device) if episode_mask is not None else None
  dist,new_h=self.actor.distribution_step(obs,self._ctx(ctx,True),hid,ep,mask);raw=dist.mean if deterministic else dist.rsample();actions=torch.tanh(raw);log=self.actor._squashed_log_prob(dist,raw,actions)
  if mask is not None:actions*=mask[...,None];raw*=mask[...,None];log*=mask
  out=(actions.cpu().numpy(),raw.cpu().numpy(),log.cpu().numpy(),None if new_h is None else new_h.cpu().numpy())
  return out if return_policy_data else (out[0],out[3])
 @torch.no_grad()
 def values_step(self,obs,alive,context=None,hidden=None,episode_mask=None,raw=True):
  conv=lambda x:None if x is None else torch.as_tensor(x,dtype=torch.float32,device=self.device)
  value,new_h=self.critic.forward_step(conv(obs),conv(alive),self._ctx(conv(context),False),conv(hidden),conv(episode_mask))
  if raw and self.popart.enabled:value=self.popart.denormalize_values(value)
  return value.cpu().numpy(),None if new_h is None else new_h.cpu().numpy()
 def _value_rollout(self,r,obs,next_obs):
  T,E=obs.shape[:2];values=[];next_values=[]
  for t in range(T):
   h=None if r.critic_hidden_before_step is None else r.critic_hidden_before_step[t]
   v,nh=self.values_step(obs[t],r.alive_masks[t],r.contexts[t],h,r.episode_masks[t] if r.episode_masks is not None else None)
   nv,_=self.values_step(next_obs[t],r.next_alive_masks[t],r.next_contexts[t],nh,1-r.dones[t])
   values.append(v);next_values.append(nv)
  return torch.as_tensor(np.asarray(values),device=self.device),torch.as_tensor(np.asarray(next_values),device=self.device)
 def update(self,r:ModularRolloutBatch):
  tt=lambda x:torch.as_tensor(x,dtype=torch.float32,device=self.device)
  obs,act,raw,oldlog,rewards,dones,alive,nobs,nalive=map(tt,(r.observations,r.actions,r.raw_actions,r.old_log_probs,r.rewards,r.dones,r.alive_masks,r.next_observations,r.next_alive_masks))
  waves=torch.as_tensor(r.wave_indices,dtype=torch.long,device=self.device);ctx=tt(r.contexts);T,E=obs.shape[:2]
  with torch.no_grad():
   values,next_values=self._value_rollout(r,obs,nobs);adv,returns=compute_gae(rewards,values,next_values,dones,alive,nalive,self.gamma,self.gae_lambda)
   if self.normalize_advantages:
    live=adv[alive>.5];adv=((adv-live.mean())/live.std(unbiased=False).clamp_min(1e-8))*alive
   if self.popart.enabled:
    self.popart.update(returns[alive>.5],self.critic.output_layer); old_values=self.popart.normalize_targets(values);target_returns=self.popart.normalize_targets(returns)
   else:old_values=values;target_returns=returns
  # One stable set of weights is computed from the complete on-policy rollout.
  wave_w,wmetrics=self.wave_balance.compute_tensor(waves,alive)
  actor_before,critic_before=self.actor_update_count,self.critic_update_count
  metrics=self._update_recurrent(r,obs,act,raw,oldlog,alive,adv,old_values,target_returns,wave_w,ctx) if (self.recurrent.actor_enabled or self.recurrent.critic_enabled) else self._update_flat(obs,act,raw,oldlog,alive,adv,old_values,target_returns,wave_w,ctx)
  live_mask=alive>.5;rv=returns[live_mask];vv=values[live_mask];variance=torch.var(rv,unbiased=False)
  metrics["explained_variance"]=float((1-torch.var(rv-vv,unbiased=False)/variance.clamp_min(1e-8)).detach())
  metrics["actor_optimizer_steps_this_update"]=float(self.actor_update_count-actor_before);metrics["critic_optimizer_steps_this_update"]=float(self.critic_update_count-critic_before)
  metrics.update(self._policy_diagnostics(r,obs,act,alive,ctx))
  self.ppo_update_count+=1;metrics.update(wmetrics);metrics.update({"popart_mean":float(self.popart.mean),"popart_std":float(self.popart.std),"popart_count":float(self.popart.count)})
  if not np.all(np.isfinite(list(metrics.values()))):raise FloatingPointError(f"non-finite modular update: {metrics}")
  return metrics
 def _loss_step(self,obs,act,raw,oldlog,mask,adv,oldvalue,target,weights,ctx,ah=None,ch=None,ep=None):
  dist,newah=self.actor.distribution_step(obs,self._ctx(ctx,True),ah,ep,mask);newlog=self.actor._squashed_log_prob(dist,raw,act);sample_raw=dist.rsample();entropy=-self.actor._squashed_log_prob(dist,sample_raw,torch.tanh(sample_raw))
  ratio=(newlog-oldlog).exp();sur=torch.minimum(ratio*adv,ratio.clamp(1-self.clip_ratio,1+self.clip_ratio)*adv)
  weights=weights.unsqueeze(-1) if weights.ndim==mask.ndim-1 else weights
  aw=weights if self.wave_balance.actor_enabled else torch.ones_like(weights);actor_loss=-masked_mean(sur*aw,mask)
  value,newch=self.critic.forward_step(obs,mask,self._ctx(ctx,False),ch,ep)
  clipped=oldvalue+(value-oldvalue).clamp(-self.clip_ratio,self.clip_ratio);err=torch.maximum((value-target).square(),(clipped-target).square()) if self.clip_value_loss else (value-target).square()
  vw=weights if self.wave_balance.critic_enabled else torch.ones_like(weights);value_loss=.5*masked_mean(err*vw,mask);ent=masked_mean(entropy,mask)
  anchor_loss=torch.zeros((),device=self.device);akl=0.
  if self.anchor.enabled and self.anchor.reference_actor is not None:
   with torch.no_grad():ref=self.anchor.reference_actor.distribution(obs)
   anchor_loss,am=self.anchor.loss(dist,ref,self.sampled_steps,mask);akl=am["anchor_kl"]
  return actor_loss,value_loss,ent,anchor_loss,ratio,newlog,newah,newch,akl
 def _opt(self,losses):
  al,vl,en,anchor,ratio,newlog,*_=losses
  self.actor_optimizer.zero_grad();(al-self.entropy_coefficient*en+anchor).backward();ag=nn.utils.clip_grad_norm_(self.actor.parameters(),self.max_grad_norm);self.actor_optimizer.step()
  self.critic_optimizer.zero_grad();(self.value_loss_coefficient*vl).backward();cg=nn.utils.clip_grad_norm_(self.critic.parameters(),self.max_grad_norm);self.critic_optimizer.step();self.actor_update_count+=1;self.critic_update_count+=1
  return ag,cg
 def _row(self,losses,mask,ag,cg):
  al,vl,en,anchor,ratio,newlog,_,_,akl=losses;logratio=ratio.log()
  live=ratio[mask>.5]
  return {"actor_loss":float(al.detach()),"weighted_actor_loss":float(al.detach()),"value_loss":float(vl.detach()),"weighted_value_loss":float(vl.detach()),"entropy":float(en.detach()),"approx_kl":float(masked_mean((ratio-1)-logratio,mask).detach()),"clip_fraction":float(masked_mean((ratio.sub(1).abs()>self.clip_ratio).float(),mask).detach()),"ratio_mean":float(live.mean().detach()),"ratio_std":float(live.std(unbiased=False).detach()),"ratio_p1":float(torch.quantile(live,.01).detach()),"ratio_p50":float(torch.quantile(live,.5).detach()),"ratio_p99":float(torch.quantile(live,.99).detach()),"ratio_min":float(live.min().detach()),"ratio_max":float(live.max().detach()),"actor_grad_norm":float(ag),"critic_grad_norm":float(cg),"anchor_kl":float(akl),"anchor_loss":float(anchor.detach()),"anchor_effective_coefficient":float(self.anchor.effective_coefficient(self.sampled_steps))}
 def _update_flat(self,obs,act,raw,oldlog,alive,adv,oldvalue,target,weights,ctx):
  flat=lambda x:x.reshape(obs.shape[0]*obs.shape[1],*x.shape[2:]); arrays=list(map(flat,(obs,act,raw,oldlog,alive,adv,oldvalue,target,weights,ctx)));N=arrays[0].shape[0];rows=[]
  for _ in range(self.ppo_epochs):
   permutation=self.rng.permutation(N)
   for start in range(0,N,self.minibatch_size):
    ix=torch.as_tensor(permutation[start:start+self.minibatch_size],device=self.device); args=[x[ix] for x in arrays];loss=self._loss_step(*args);ag,cg=self._opt(loss);rows.append(self._row(loss,args[4],ag,cg))
  return {k:float(np.mean([r[k] for r in rows])) for k in rows[0]}
 def _update_recurrent(self,r,obs,act,raw,oldlog,alive,adv,oldvalue,target,weights,ctx):
  tt=lambda x:torch.as_tensor(x,dtype=torch.float32,device=self.device)
  chunks=contiguous_chunks(obs.shape[0],obs.shape[1],self.recurrent.sequence_length);rows=[]
  sequences_per_minibatch=max(1,self.minibatch_size//self.recurrent.sequence_length)
  for _ in range(self.ppo_epochs):
   order=self.rng.permutation(len(chunks))
   for start in range(0,len(chunks),sequences_per_minibatch):
    group=[chunks[int(i)] for i in order[start:start+sequences_per_minibatch]]
    rows.append(self._recurrent_minibatch(r,group,obs,act,raw,oldlog,alive,adv,oldvalue,target,weights,ctx,tt))
  out={k:float(np.mean([x[k] for x in rows])) for k in rows[0]};out.update({"sequence_chunks":float(len(chunks)),"sequences_per_minibatch":float(sequences_per_minibatch),"recurrent_minibatches_per_epoch":float(np.ceil(len(chunks)/sequences_per_minibatch))});return out

 def _recurrent_minibatch(self,r,group,obs,act,raw,oldlog,alive,adv,oldvalue,target,weights,ctx,tt):
  length=max(z-s for _,s,z in group);batch=len(group)
  def padded(source):
   out=torch.zeros((length,batch,*source.shape[2:]),dtype=source.dtype,device=self.device)
   for b,(e,s,z) in enumerate(group):out[:z-s,b]=source[s:z,e]
   return out
  O,A,R,L,M,ADV,OV,TG,W,C=map(padded,(obs,act,raw,oldlog,alive,adv,oldvalue,target,weights,ctx))
  valid=torch.zeros(length,batch,device=self.device)
  for b,(_,s,z) in enumerate(group):valid[:z-s,b]=1
  EP=torch.zeros(length,batch,device=self.device)
  if r.episode_masks is not None:
   episode=tt(r.episode_masks)
   for b,(e,s,z) in enumerate(group):EP[:z-s,b]=episode[s:z,e]
  def initial(saved):
   if saved is None:return None
   return torch.stack([tt(saved[s,e]) for e,s,_ in group]).detach()
  ah,ch=initial(r.actor_hidden_before_step),initial(r.critic_hidden_before_step)
  surrogates=[];errors=[];entropies=[];ratios=[];logratios=[];masks=[];waveweights=[];anchor_kls=[]
  for t in range(length):
   mask=M[t]*valid[t,:,None]
   dist,ah=self.actor.distribution_step(O[t],self._ctx(C[t],True),ah,EP[t],mask)
   newlog=self.actor._squashed_log_prob(dist,R[t],A[t]);logratio=newlog-L[t];ratio=logratio.exp()
   surrogate=torch.minimum(ratio*ADV[t],ratio.clamp(1-self.clip_ratio,1+self.clip_ratio)*ADV[t])
   sampled=dist.rsample();entropy=-self.actor._squashed_log_prob(dist,sampled,torch.tanh(sampled))
   value,ch=self.critic.forward_step(O[t],mask,self._ctx(C[t],False),ch,EP[t])
   clipped=OV[t]+(value-OV[t]).clamp(-self.clip_ratio,self.clip_ratio)
   error=torch.maximum((value-TG[t]).square(),(clipped-TG[t]).square()) if self.clip_value_loss else (value-TG[t]).square()
   if self.anchor.enabled and self.anchor.reference_actor is not None:
    with torch.no_grad():reference=self.anchor.reference_actor.distribution(O[t])
    anchor_kls.append(kl_divergence(dist,reference).sum(-1))
   surrogates.append(surrogate);errors.append(error);entropies.append(entropy);ratios.append(ratio);logratios.append(logratio);masks.append(mask);waveweights.append(W[t,:,None].expand_as(mask))
  surrogate,error,entropy,ratio,logratio,mask,ww=map(lambda x:torch.stack(x),(surrogates,errors,entropies,ratios,logratios,masks,waveweights))
  aw=ww if self.wave_balance.actor_enabled else torch.ones_like(ww);vw=ww if self.wave_balance.critic_enabled else torch.ones_like(ww)
  actor_loss=-recurrent_alive_mean(surrogate*aw,M,valid);value_loss=.5*recurrent_alive_mean(error*vw,M,valid);entropy_mean=recurrent_alive_mean(entropy,M,valid)
  anchor_mean=recurrent_alive_mean(torch.stack(anchor_kls),M,valid) if anchor_kls else torch.zeros((),device=self.device)
  anchor_loss=anchor_mean*self.anchor.effective_coefficient(self.sampled_steps)
  merged=(actor_loss,value_loss,entropy_mean,anchor_loss,ratio.reshape(-1,self.num_agents),logratio.reshape(-1,self.num_agents),ah,ch,float(anchor_mean.detach()))
  ag,cg=self._opt(merged);return self._row(merged,mask.reshape(-1,self.num_agents),ag,cg)

 @torch.no_grad()
 def _policy_diagnostics(self,r,obs,actions,alive,ctx):
  logstd=[]
  for t in range(obs.shape[0]):
   hidden=None if r.actor_hidden_before_step is None else torch.as_tensor(r.actor_hidden_before_step[t],dtype=torch.float32,device=self.device)
   ep=None if r.episode_masks is None else torch.as_tensor(r.episode_masks[t],dtype=torch.float32,device=self.device)
   dist,_=self.actor.distribution_step(obs[t],self._ctx(ctx[t],True),hidden,ep,alive[t]);logstd.append(dist.scale.log())
  logs=torch.stack(logstd);live=alive>.5;live_actions=actions[live];live_logs=logs[live];result={}
  for index,name in enumerate(("psi","theta","v")):
   result[f"policy_log_std_mean_{name}"]=float(live_logs[:,index].mean())
   for threshold,label in ((.9,"0_9"),(.99,"0_99"),(.999,"0_999")):result[f"action_abs_gt_{label}_fraction_{name}"]=float((live_actions[:,index].abs()>threshold).float().mean())
  return result
 def module_protocol(self):
  raw=json.dumps(self.modules_config,sort_keys=True,separators=(",",":"));return {"enabled_modules":enabled_module_names(self.modules_config),"module_config":deepcopy(self.modules_config),"module_config_sha256":hashlib.sha256(raw.encode()).hexdigest()}
 def checkpoint_state(self,extra=None):
  return {"algorithm":"modular_mappo","modular_mappo_impl_version":MODULAR_MAPPO_IMPL_VERSION,"baseline_mappo_impl_version":MAPPO_IMPL_VERSION,"actor":self.actor.state_dict(),"critic":self.critic.state_dict(),"actor_optimizer":self.actor_optimizer.state_dict(),"critic_optimizer":self.critic_optimizer.state_dict(),"popart":self.popart.state_dict(),"ppo_updates":self.ppo_update_count,"actor_updates":self.actor_update_count,"critic_updates":self.critic_update_count,"sampled_steps":self.sampled_steps,"vector_steps":self.vector_steps,**self.module_protocol(),"warm_start_provenance":self.warm_start_provenance,"anchor_provenance":self.anchor_provenance,"anchor_reference_actor_state":None if self.anchor.reference_actor is None else self.anchor.reference_actor.state_dict(),"extra":extra or {}}
 def save(self,path,extra=None):Path(path).parent.mkdir(parents=True,exist_ok=True);torch.save(self.checkpoint_state(extra),path)
 def load(self,path,strict_protocol=True):
  state=torch.load(path,map_location=self.device,weights_only=False)
  if state.get("algorithm")!="modular_mappo":raise RuntimeError("not a modular_mappo checkpoint")
  if strict_protocol and state.get("module_config_sha256")!=self.module_protocol()["module_config_sha256"]:raise RuntimeError("checkpoint module protocol mismatch")
  self.actor.load_state_dict(state["actor"]);self.critic.load_state_dict(state["critic"]);self.popart.load_state_dict(state.get("popart",{}),strict=False)
  self.actor_optimizer.load_state_dict(state["actor_optimizer"]);self.critic_optimizer.load_state_dict(state["critic_optimizer"])
  self.warm_start_provenance=state.get("warm_start_provenance",{});self.anchor_provenance=state.get("anchor_provenance",{})
  reference_state=state.get("anchor_reference_actor_state")
  if self.anchor.enabled:
   if reference_state is None:raise RuntimeError("policy-anchor checkpoint is not self-contained")
   reference=deepcopy(self.actor).to(self.device);reference.load_state_dict(reference_state);self.anchor.attach(reference,self.anchor_provenance.get("reference_checkpoint"))
  for key,attr in (("ppo_updates","ppo_update_count"),("actor_updates","actor_update_count"),("critic_updates","critic_update_count"),("sampled_steps","sampled_steps"),("vector_steps","vector_steps")):setattr(self,attr,int(state.get(key,0)))
  return state.get("extra",{})

__all__=["MODULAR_MAPPO_IMPL_VERSION","ModularMAPPOTrainer"]
