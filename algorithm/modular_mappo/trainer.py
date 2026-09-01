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
from algorithm.modules import (AdvantagePriorityModule,PPOStabilizationModule,
 ADVANTAGE_PRIORITY_VERSION,PPO_STABILIZATION_VERSION)
from .networks import ModularMAPPOActor,ModularCentralizedCritic
from .buffer import ModularRolloutBatch,contiguous_chunks,recurrent_alive_mean

# Version 2 is the formal hardened implementation. Version 1 checkpoints use
# prototype recurrent/weighting semantics and are diagnostic-only artifacts.
MODULAR_MAPPO_IMPL_VERSION=2

def stable_ratio_terms(new_log_prob,old_log_prob):
 """Return PPO log-ratio/ratio without reconstructing log-ratio from exp()."""
 if not torch.all(torch.isfinite(new_log_prob)):raise FloatingPointError("non-finite new_log_prob")
 if not torch.all(torch.isfinite(old_log_prob)):raise FloatingPointError("non-finite old_log_prob")
 log_ratio=new_log_prob-old_log_prob
 if not torch.all(torch.isfinite(log_ratio)):raise FloatingPointError("non-finite log_ratio")
 ratio=log_ratio.exp()
 if not torch.all(torch.isfinite(ratio)):raise FloatingPointError("non-finite ratio")
 return log_ratio,ratio

def aggregate_update_rows(rows,clip_ratio):
 """Aggregate sample diagnostics over every alive sample in the PPO update."""
 if not rows:raise RuntimeError("no modular PPO metric rows")
 counts=np.asarray([row["_valid_count"] for row in rows],dtype=np.float64);total=float(counts.sum())
 if total<=0:raise FloatingPointError("no valid alive PPO samples")
 private={"_valid_count","_ratio_values","_log_ratio_values"}
 weighted={"actor_loss","weighted_actor_loss","value_loss","weighted_value_loss","entropy","approx_kl","clip_fraction"}
 result={}
 for key in rows[0]:
  if key in private or key.startswith("ratio_") or key.startswith("log_ratio_") or key=="max_abs_log_ratio":continue
  values=np.asarray([row[key] for row in rows],dtype=np.float64)
  result[key]=float(np.sum(values*counts)/total) if key in weighted else float(values.mean())
 ratio=np.concatenate([row["_ratio_values"] for row in rows]).astype(np.float64,copy=False)
 log_ratio=np.concatenate([row["_log_ratio_values"] for row in rows]).astype(np.float64,copy=False)
 if ratio.size!=int(total) or log_ratio.size!=int(total):raise RuntimeError("ratio diagnostic sample-count mismatch")
 if not np.all(np.isfinite(log_ratio)):raise FloatingPointError("non-finite log_ratio diagnostics")
 if not np.all(np.isfinite(ratio)):raise FloatingPointError("non-finite ratio diagnostics")
 kl=(ratio-1.0)-log_ratio
 if not np.all(np.isfinite(kl)):raise FloatingPointError("non-finite approx_kl samples")
 underflow=(ratio==0.0)&np.isfinite(log_ratio)
 result.update({
  "approx_kl":float(kl.mean()),
  "clip_fraction":float((np.abs(ratio-1.0)>clip_ratio).mean()),
  "ratio_mean":float(ratio.mean()),"ratio_std":float(ratio.std()),
  "ratio_p1":float(np.quantile(ratio,.01)),"ratio_p50":float(np.quantile(ratio,.5)),"ratio_p99":float(np.quantile(ratio,.99)),
  "ratio_min":float(ratio.min()),"ratio_max":float(ratio.max()),
  "log_ratio_min":float(log_ratio.min()),"log_ratio_max":float(log_ratio.max()),"max_abs_log_ratio":float(np.abs(log_ratio).max()),
  "ratio_underflow_count":int(underflow.sum()),"ratio_underflow_fraction":float(underflow.mean()),"ratio_sample_count":int(total),
 })
 return result

class ModularMAPPOTrainer:
 def __init__(self,observation_dim=52,action_dim=3,num_agents=4,hidden_dim=256,attention_heads=2,
  actor_learning_rate=3e-4,critic_learning_rate=3e-4,gamma=.99,gae_lambda=.95,clip_ratio=.2,
  value_loss_coefficient=.5,entropy_coefficient=.01,max_grad_norm=.5,ppo_epochs=10,minibatch_size=512,
  normalize_advantages=True,clip_value_loss=True,device="cpu",seed=0,actor_activation="relu",
  critic_activation="relu",log_std_min=-5.,log_std_max=2.,modules_config=None,
  total_sampled_steps=1):
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
  self.advantage_priority=AdvantagePriorityModule(self.modules_config.get("advantage_priority"))
  self.ppo_stabilization=PPOStabilizationModule(self.modules_config.get("ppo_stabilization"))
  self.entity_attention_config=deepcopy(self.modules_config.get("entity_attention",{}))
  self.entity_attention_enabled=bool(self.entity_attention_config.get("enabled",False))
  self.total_sampled_steps=int(total_sampled_steps)
  if self.total_sampled_steps<=0:raise ValueError("total_sampled_steps must be positive")
  if self.entity_attention_enabled and (self.recurrent.enabled or self.wave_context.enabled):raise ValueError("entity attention v1 is incompatible with recurrent memory and wave context")
  if self.ppo_stabilization.enabled and self.recurrent.enabled:raise ValueError("PPO stabilization v1 requires the feed-forward update path")
  ac=self.wave_context.context_dim if self.wave_context.actor_enabled else 0;cc=self.wave_context.context_dim if self.wave_context.critic_enabled else 0
  ar=self.recurrent.hidden_dim if self.recurrent.actor_enabled else 0;cr=self.recurrent.hidden_dim if self.recurrent.critic_enabled else 0
  self.actor=ModularMAPPOActor(observation_dim,action_dim,hidden_dim,log_std_min,log_std_max,actor_activation,ac,ar,self.entity_attention_config).to(self.device)
  self.critic=ModularCentralizedCritic(observation_dim,hidden_dim,attention_heads,critic_activation,cc,cr).to(self.device)
  self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=actor_learning_rate);self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=critic_learning_rate)
  self.ppo_update_count=self.actor_update_count=self.critic_update_count=self.sampled_steps=self.vector_steps=0
  self.kl_hard_stop_count=0
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
   values,next_values=self._value_rollout(r,obs,nobs);adv,returns=compute_gae(rewards,values,next_values,dones,alive,nalive,self.gamma,self.gae_lambda);raw_adv=adv.clone()
   wave_w,wmetrics=self.wave_balance.compute_tensor(waves,alive)
   _,actor_w,pmetrics=self.advantage_priority.compute_tensor(raw_adv,waves,alive,wave_w if self.wave_balance.actor_enabled else torch.ones_like(wave_w))
   if self.normalize_advantages:
    live=adv[alive>.5];adv=((adv-live.mean())/live.std(unbiased=False).clamp_min(1e-8))*alive
   if self.popart.enabled:
    self.popart.update(returns[alive>.5],self.critic.output_layer); old_values=self.popart.normalize_targets(values);target_returns=self.popart.normalize_targets(returns)
   else:old_values=values;target_returns=returns
  # All actor priorities are frozen from the complete raw-GAE rollout.
  if self.ppo_stabilization.enabled:
   lr=self.ppo_stabilization.actor_learning_rate(self.sampled_steps,self.total_sampled_steps)
   for group in self.actor_optimizer.param_groups:group["lr"]=lr
  actor_before,critic_before=self.actor_update_count,self.critic_update_count
  if self.recurrent.actor_enabled or self.recurrent.critic_enabled:
   metrics=self._update_recurrent(r,obs,act,raw,oldlog,alive,adv,old_values,target_returns,wave_w,ctx)
  elif self.ppo_stabilization.enabled:
   metrics=self._update_flat_stabilized(obs,act,raw,oldlog,alive,adv,old_values,target_returns,wave_w,ctx,actor_w)
  else:
   metrics=self._update_flat(obs,act,raw,oldlog,alive,adv,old_values,target_returns,wave_w,ctx,actor_w if self.advantage_priority.enabled else None)
  live_mask=alive>.5;rv=returns[live_mask];vv=values[live_mask];variance=torch.var(rv,unbiased=False)
  metrics["explained_variance"]=float((1-torch.var(rv-vv,unbiased=False)/variance.clamp_min(1e-8)).detach())
  metrics["actor_optimizer_steps_this_update"]=float(self.actor_update_count-actor_before);metrics["critic_optimizer_steps_this_update"]=float(self.critic_update_count-critic_before)
  recurrent_steps=(self.actor_update_count-actor_before) if (self.recurrent.actor_enabled or self.recurrent.critic_enabled) else 0
  metrics["recurrent_optimizer_steps_this_update"]=float(recurrent_steps)
  metrics.update(self._policy_diagnostics(r,obs,act,alive,ctx))
  self.ppo_update_count+=1;metrics.update(wmetrics);metrics.update(pmetrics);metrics.update({"popart_mean":float(self.popart.mean),"popart_std":float(self.popart.std),"popart_count":float(self.popart.count),"actor_learning_rate":float(self.actor_optimizer.param_groups[0]["lr"]),"critic_learning_rate":float(self.critic_optimizer.param_groups[0]["lr"]),"kl_hard_stop_count":float(self.kl_hard_stop_count),"cumulative_kl_hard_stop_count":float(self.kl_hard_stop_count)})
  if not np.all(np.isfinite(list(metrics.values()))):raise FloatingPointError(f"non-finite modular update: {metrics}")
  return metrics
 def _loss_step(self,obs,act,raw,oldlog,mask,adv,oldvalue,target,weights,ctx,ah=None,ch=None,ep=None,actor_weights=None):
  dist,newah=self.actor.distribution_step(obs,self._ctx(ctx,True),ah,ep,mask);newlog=self.actor._squashed_log_prob(dist,raw,act);sample_raw=dist.rsample();entropy=-self.actor._squashed_log_prob(dist,sample_raw,torch.tanh(sample_raw))
  logratio,ratio=stable_ratio_terms(newlog,oldlog);sur=torch.minimum(ratio*adv,ratio.clamp(1-self.clip_ratio,1+self.clip_ratio)*adv)
  weights=weights.unsqueeze(-1) if weights.ndim==mask.ndim-1 else weights
  aw=actor_weights if actor_weights is not None else (weights if self.wave_balance.actor_enabled else torch.ones_like(weights))
  aw=aw.unsqueeze(-1) if aw.ndim==mask.ndim-1 else aw
  actor_loss=-masked_mean(sur*aw,mask)
  value,newch=self.critic.forward_step(obs,mask,self._ctx(ctx,False),ch,ep)
  clipped=oldvalue+(value-oldvalue).clamp(-self.clip_ratio,self.clip_ratio);err=torch.maximum((value-target).square(),(clipped-target).square()) if self.clip_value_loss else (value-target).square()
  vw=weights if self.wave_balance.critic_enabled else torch.ones_like(weights);value_loss=.5*masked_mean(err*vw,mask);ent=masked_mean(entropy,mask)
  anchor_loss=torch.zeros((),device=self.device);akl=0.
  if self.anchor.enabled and self.anchor.reference_actor is not None:
   with torch.no_grad():ref=self.anchor.reference_actor.distribution(obs)
   anchor_loss,am=self.anchor.loss(dist,ref,self.sampled_steps,mask);akl=am["anchor_kl"]
  return actor_loss,value_loss,ent,anchor_loss,ratio,logratio,newlog,oldlog,newah,newch,akl
 @staticmethod
 def _gradient_norm(parameters):
  total=torch.zeros((),dtype=torch.float64)
  for parameter in parameters:
   if parameter.grad is not None:total+=parameter.grad.detach().double().square().sum().cpu()
  return float(total.sqrt())
 def _opt(self,losses):
  al,vl,en,anchor,*_=losses
  self.actor_optimizer.zero_grad();(al-self.entropy_coefficient*en+anchor).backward();arg=self._gradient_norm(self.actor.gru.parameters()) if self.recurrent.actor_enabled else 0.;ag=nn.utils.clip_grad_norm_(self.actor.parameters(),self.max_grad_norm);self.actor_optimizer.step()
  self.critic_optimizer.zero_grad();(self.value_loss_coefficient*vl).backward();crg=self._gradient_norm(self.critic.gru.parameters()) if self.recurrent.critic_enabled else 0.;cg=nn.utils.clip_grad_norm_(self.critic.parameters(),self.max_grad_norm);self.critic_optimizer.step();self.actor_update_count+=1;self.critic_update_count+=1
  return ag,cg,arg,crg
 def _row(self,losses,mask,ag,cg,arg=0.,crg=0.):
  al,vl,en,anchor,ratio,logratio,newlog,oldlog,_,_,akl=losses
  stable_ratio_terms(newlog,oldlog)
  live_mask=mask>.5;live=ratio[live_mask];live_logratio=logratio[live_mask]
  if live.numel()==0:raise FloatingPointError("no valid alive PPO samples")
  kl=(ratio-1)-logratio
  return {"actor_loss":float(al.detach()),"weighted_actor_loss":float(al.detach()),"value_loss":float(vl.detach()),"weighted_value_loss":float(vl.detach()),"entropy":float(en.detach()),"approx_kl":float(masked_mean(kl,mask).detach()),"clip_fraction":float(masked_mean((ratio.sub(1).abs()>self.clip_ratio).float(),mask).detach()),"actor_grad_norm":float(ag),"critic_grad_norm":float(cg),"actor_gru_grad_norm":float(arg),"critic_gru_grad_norm":float(crg),"gru_gradient_norm":float(np.hypot(arg,crg)),"anchor_kl":float(akl),"anchor_loss":float(anchor.detach()),"anchor_effective_coefficient":float(self.anchor.effective_coefficient(self.sampled_steps)),"_valid_count":int(live.numel()),"_ratio_values":live.detach().cpu().numpy(),"_log_ratio_values":live_logratio.detach().cpu().numpy()}
 def _update_flat(self,obs,act,raw,oldlog,alive,adv,oldvalue,target,weights,ctx,actor_weights=None):
  flat=lambda x:x.reshape(obs.shape[0]*obs.shape[1],*x.shape[2:]); arrays=list(map(flat,(obs,act,raw,oldlog,alive,adv,oldvalue,target,weights,ctx)));flat_actor=None if actor_weights is None else flat(actor_weights);N=arrays[0].shape[0];rows=[]
  for _ in range(self.ppo_epochs):
   permutation=self.rng.permutation(N)
   for start in range(0,N,self.minibatch_size):
    ix=torch.as_tensor(permutation[start:start+self.minibatch_size],device=self.device); args=[x[ix] for x in arrays];loss=self._loss_step(*args,actor_weights=None if flat_actor is None else flat_actor[ix]);ag,cg,arg,crg=self._opt(loss);rows.append(self._row(loss,args[4],ag,cg,arg,crg))
  return aggregate_update_rows(rows,self.clip_ratio)

 def _update_flat_stabilized(self,obs,act,raw,oldlog,alive,adv,oldvalue,target,weights,ctx,actor_weights):
  """Split actor/critic epochs so a hard actor KL stop never truncates critic work."""
  flat=lambda x:x.reshape(obs.shape[0]*obs.shape[1],*x.shape[2:])
  arrays=list(map(flat,(obs,act,raw,oldlog,alive,adv,oldvalue,target,weights,ctx,actor_weights)));N=arrays[0].shape[0]
  actor_rows=[];critic_losses=[];critic_grad_norms=[];epoch_kls=[];actor_epochs=0;hard_stop=False
  for _ in range(self.ppo_epochs):
   permutation=self.rng.permutation(N)
   for start in range(0,N,self.minibatch_size):
    ix=torch.as_tensor(permutation[start:start+self.minibatch_size],device=self.device);args=[x[ix] for x in arrays[:10]]
    loss=self._loss_step(*args,actor_weights=arrays[10][ix]);al,_,en,anchor,*_=loss
    self.actor_optimizer.zero_grad();(al-self.entropy_coefficient*en+anchor).backward();ag=nn.utils.clip_grad_norm_(self.actor.parameters(),self.max_grad_norm);self.actor_optimizer.step();self.actor_update_count+=1
    actor_rows.append(self._row(loss,args[4],ag,0.))
   actor_epochs+=1;epoch_kl=self._full_rollout_kl(arrays[0],arrays[2],arrays[3],arrays[4],arrays[9]);epoch_kls.append(epoch_kl)
   if self.ppo_stabilization.should_stop_actor(epoch_kl):hard_stop=True;self.kl_hard_stop_count+=1;break
  for _ in range(self.ppo_epochs):
   permutation=self.rng.permutation(N)
   for start in range(0,N,self.minibatch_size):
    ix=torch.as_tensor(permutation[start:start+self.minibatch_size],device=self.device);args=[x[ix] for x in arrays[:10]]
    loss=self._loss_step(*args,actor_weights=arrays[10][ix]);vl=loss[1]
    self.critic_optimizer.zero_grad();(self.value_loss_coefficient*vl).backward();cg=nn.utils.clip_grad_norm_(self.critic.parameters(),self.max_grad_norm);self.critic_optimizer.step();self.critic_update_count+=1
    critic_losses.append(float(vl.detach()));critic_grad_norms.append(float(cg))
  out=aggregate_update_rows(actor_rows,self.clip_ratio)
  out.update({"value_loss":float(np.mean(critic_losses)),"weighted_value_loss":float(np.mean(critic_losses)),"critic_grad_norm":float(np.mean(critic_grad_norms)),"actor_epochs_planned":float(self.ppo_epochs),"actor_epochs_used":float(actor_epochs),"critic_epochs_used":float(self.ppo_epochs),"epoch_kl_last":float(epoch_kls[-1]),"epoch_kl_max":float(max(epoch_kls)),"kl_hard_stop_triggered":float(hard_stop)})
  return out

 @torch.no_grad()
 def _full_rollout_kl(self,obs,raw,oldlog,alive,ctx):
  values=[];batch=max(self.minibatch_size,1)
  for start in range(0,obs.shape[0],batch):
   dist,_=self.actor.distribution_step(obs[start:start+batch],self._ctx(ctx[start:start+batch],True),None,None,alive[start:start+batch]);newlog=self.actor._squashed_log_prob(dist,raw[start:start+batch],torch.tanh(raw[start:start+batch]));logratio,ratio=stable_ratio_terms(newlog,oldlog[start:start+batch]);mask=alive[start:start+batch]> .5;values.append(((ratio-1)-logratio)[mask])
  return float(torch.cat(values).double().mean())
 def _update_recurrent(self,r,obs,act,raw,oldlog,alive,adv,oldvalue,target,weights,ctx):
  tt=lambda x:torch.as_tensor(x,dtype=torch.float32,device=self.device)
  chunks=contiguous_chunks(obs.shape[0],obs.shape[1],self.recurrent.sequence_length);rows=[]
  sequences_per_minibatch=max(1,self.minibatch_size//self.recurrent.sequence_length)
  for _ in range(self.ppo_epochs):
   order=self.rng.permutation(len(chunks))
   for start in range(0,len(chunks),sequences_per_minibatch):
    group=[chunks[int(i)] for i in order[start:start+sequences_per_minibatch]]
    rows.append(self._recurrent_minibatch(r,group,obs,act,raw,oldlog,alive,adv,oldvalue,target,weights,ctx,tt))
  out=aggregate_update_rows(rows,self.clip_ratio);out.update({"sequence_chunks":float(len(chunks)),"sequences_per_minibatch":float(sequences_per_minibatch),"recurrent_minibatches_per_epoch":float(np.ceil(len(chunks)/sequences_per_minibatch))});return out

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
  surrogates=[];errors=[];entropies=[];ratios=[];logratios=[];newlogs=[];masks=[];waveweights=[];anchor_kls=[]
  for t in range(length):
   mask=M[t]*valid[t,:,None]
   dist,ah=self.actor.distribution_step(O[t],self._ctx(C[t],True),ah,EP[t],mask)
   newlog=self.actor._squashed_log_prob(dist,R[t],A[t]);logratio,ratio=stable_ratio_terms(newlog,L[t])
   surrogate=torch.minimum(ratio*ADV[t],ratio.clamp(1-self.clip_ratio,1+self.clip_ratio)*ADV[t])
   sampled=dist.rsample();entropy=-self.actor._squashed_log_prob(dist,sampled,torch.tanh(sampled))
   value,ch=self.critic.forward_step(O[t],mask,self._ctx(C[t],False),ch,EP[t])
   clipped=OV[t]+(value-OV[t]).clamp(-self.clip_ratio,self.clip_ratio)
   error=torch.maximum((value-TG[t]).square(),(clipped-TG[t]).square()) if self.clip_value_loss else (value-TG[t]).square()
   if self.anchor.enabled and self.anchor.reference_actor is not None:
    with torch.no_grad():reference=self.anchor.reference_actor.distribution(O[t])
    anchor_kls.append(kl_divergence(dist,reference).sum(-1))
   surrogates.append(surrogate);errors.append(error);entropies.append(entropy);ratios.append(ratio);logratios.append(logratio);newlogs.append(newlog);masks.append(mask);waveweights.append(W[t,:,None].expand_as(mask))
  surrogate,error,entropy,ratio,logratio,newlog,mask,ww=map(lambda x:torch.stack(x),(surrogates,errors,entropies,ratios,logratios,newlogs,masks,waveweights))
  aw=ww if self.wave_balance.actor_enabled else torch.ones_like(ww);vw=ww if self.wave_balance.critic_enabled else torch.ones_like(ww)
  actor_loss=-recurrent_alive_mean(surrogate*aw,M,valid);value_loss=.5*recurrent_alive_mean(error*vw,M,valid);entropy_mean=recurrent_alive_mean(entropy,M,valid)
  anchor_mean=recurrent_alive_mean(torch.stack(anchor_kls),M,valid) if anchor_kls else torch.zeros((),device=self.device)
  anchor_loss=anchor_mean*self.anchor.effective_coefficient(self.sampled_steps)
  merged=(actor_loss,value_loss,entropy_mean,anchor_loss,ratio.reshape(-1,self.num_agents),logratio.reshape(-1,self.num_agents),newlog.reshape(-1,self.num_agents),L.reshape(-1,self.num_agents),ah,ch,float(anchor_mean.detach()))
  ag,cg,arg,crg=self._opt(merged);return self._row(merged,mask.reshape(-1,self.num_agents),ag,cg,arg,crg)

 @torch.no_grad()
 def _policy_diagnostics(self,r,obs,actions,alive,ctx):
  logstd=[];attention=[]
  for t in range(obs.shape[0]):
   hidden=None if r.actor_hidden_before_step is None else torch.as_tensor(r.actor_hidden_before_step[t],dtype=torch.float32,device=self.device)
   ep=None if r.episode_masks is None else torch.as_tensor(r.episode_masks[t],dtype=torch.float32,device=self.device)
   if self.entity_attention_enabled:
    dist,_,diag=self.actor.distribution_step(obs[t],self._ctx(ctx[t],True),hidden,ep,alive[t],return_attention=True);attention.append(diag)
   else:dist,_=self.actor.distribution_step(obs[t],self._ctx(ctx[t],True),hidden,ep,alive[t])
   logstd.append(dist.scale.log())
  logs=torch.stack(logstd);live=alive>.5;live_actions=actions[live];live_logs=logs[live];result={}
  for index,name in enumerate(("psi","theta","v")):
   result[f"policy_log_std_mean_{name}"]=float(live_logs[:,index].mean())
   for threshold,label in ((.9,"0_9"),(.99,"0_99"),(.999,"0_999")):result[f"action_abs_gt_{label}_fraction_{name}"]=float((live_actions[:,index].abs()>threshold).float().mean())
  result["entity_attention_enabled"]=float(self.entity_attention_enabled)
  if attention:
   query_alive=alive>.5
   for group in ("ally","enemy"):
    weights=torch.stack([item[f"{group}_attention_weights"] for item in attention])
    entities=torch.stack([item[f"{group}_entity_alive"] for item in attention])
    entropy=-(weights*weights.clamp_min(1e-12).log()).sum(-1).mean(-1)
    top1=weights.max(-1).values.mean(-1)
    dead_mass=(weights*(entities<=.5).unsqueeze(-2)).sum(-1).mean(-1)
    result[f"{group}_attention_entropy"]=float(entropy[query_alive].mean())
    result[f"{group}_attention_top1_weight"]=float(top1[query_alive].mean())
    result[f"{group}_alive_entity_count"]=float(entities[query_alive].sum(-1).float().mean())
    result[f"{group}_attention_dead_mass"]=float(dead_mass[query_alive].mean())
   result["attention_dead_mass"]=max(result["ally_attention_dead_mass"],result["enemy_attention_dead_mass"])
  return result
 def module_protocol(self):
  raw=json.dumps(self.modules_config,sort_keys=True,separators=(",",":"));return {"enabled_modules":enabled_module_names(self.modules_config),"module_config":deepcopy(self.modules_config),"module_config_sha256":hashlib.sha256(raw.encode()).hexdigest()}
 def checkpoint_state(self,extra=None):
  return {"algorithm":"modular_mappo","modular_mappo_impl_version":MODULAR_MAPPO_IMPL_VERSION,"baseline_mappo_impl_version":MAPPO_IMPL_VERSION,"development_feature_versions":{"advantage_priority":ADVANTAGE_PRIORITY_VERSION,"ppo_stabilization":PPO_STABILIZATION_VERSION,"entity_attention":1},"actor":self.actor.state_dict(),"critic":self.critic.state_dict(),"actor_optimizer":self.actor_optimizer.state_dict(),"critic_optimizer":self.critic_optimizer.state_dict(),"popart":self.popart.state_dict(),"ppo_updates":self.ppo_update_count,"actor_updates":self.actor_update_count,"critic_updates":self.critic_update_count,"sampled_steps":self.sampled_steps,"vector_steps":self.vector_steps,"kl_hard_stop_count":self.kl_hard_stop_count,**self.module_protocol(),"warm_start_provenance":self.warm_start_provenance,"anchor_provenance":self.anchor_provenance,"anchor_reference_actor_state":None if self.anchor.reference_actor is None else self.anchor.reference_actor.state_dict(),"extra":extra or {}}
 def save(self,path,extra=None):Path(path).parent.mkdir(parents=True,exist_ok=True);torch.save(self.checkpoint_state(extra),path)
 def load(self,path,strict_protocol=True):
  state=torch.load(path,map_location=self.device,weights_only=False)
  if state.get("algorithm")!="modular_mappo":raise RuntimeError("not a modular_mappo checkpoint")
  checkpoint_version=state.get("modular_mappo_impl_version")
  if checkpoint_version!=MODULAR_MAPPO_IMPL_VERSION:raise RuntimeError(f"modular implementation version mismatch: checkpoint={checkpoint_version}, current={MODULAR_MAPPO_IMPL_VERSION}")
  if state.get("baseline_mappo_impl_version")!=MAPPO_IMPL_VERSION:raise RuntimeError("baseline MAPPO implementation version mismatch")
  if strict_protocol and state.get("module_config_sha256")!=self.module_protocol()["module_config_sha256"]:raise RuntimeError("checkpoint module protocol mismatch")
  if strict_protocol and (self.entity_attention_enabled or self.advantage_priority.enabled or self.ppo_stabilization.enabled):
   expected={"advantage_priority":ADVANTAGE_PRIORITY_VERSION,"ppo_stabilization":PPO_STABILIZATION_VERSION,"entity_attention":1}
   if state.get("development_feature_versions")!=expected:raise RuntimeError("checkpoint development feature version mismatch")
  self.actor.load_state_dict(state["actor"]);self.critic.load_state_dict(state["critic"]);self.popart.load_state_dict(state.get("popart",{}),strict=False)
  self.actor_optimizer.load_state_dict(state["actor_optimizer"]);self.critic_optimizer.load_state_dict(state["critic_optimizer"])
  self.warm_start_provenance=state.get("warm_start_provenance",{});self.anchor_provenance=state.get("anchor_provenance",{})
  reference_state=state.get("anchor_reference_actor_state")
  if self.anchor.enabled:
   if reference_state is None:raise RuntimeError("policy-anchor checkpoint is not self-contained")
   reference=deepcopy(self.actor).to(self.device);reference.load_state_dict(reference_state);self.anchor.attach(reference,self.anchor_provenance.get("reference_checkpoint"))
  for key,attr in (("ppo_updates","ppo_update_count"),("actor_updates","actor_update_count"),("critic_updates","critic_update_count"),("sampled_steps","sampled_steps"),("vector_steps","vector_steps")):setattr(self,attr,int(state.get(key,0)))
  self.kl_hard_stop_count=int(state.get("kl_hard_stop_count",0))
  if self.ppo_stabilization.enabled:
   lr=self.ppo_stabilization.actor_learning_rate(self.sampled_steps,self.total_sampled_steps)
   for group in self.actor_optimizer.param_groups:group["lr"]=lr
  return state.get("extra",{})

__all__=["MODULAR_MAPPO_IMPL_VERSION","ModularMAPPOTrainer"]
