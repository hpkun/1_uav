"""Short/formal modular MAPPO runner with raw/training reward separation."""
from __future__ import annotations
from copy import deepcopy
import csv,json
from pathlib import Path
import numpy as np
import yaml
from algorithm.common.vector_env import ParallelVectorEnv
from algorithm.common.protocol import config_sha256
from .factory import build_modular_mappo_trainer
from .buffer import ModularRolloutBatch
from .evaluation import evaluate_modular
from .protocol import checkpoint_architecture

class ModularMAPPOTrainingRunner:
 def __init__(self,env_config,algorithm_config,num_envs=None,total_sampled_steps=None,device=None,seed=None,output_dir=None,smoke=False,warm_start_checkpoint=None,reference_checkpoint=None):
  self.env_config=deepcopy(env_config);self.algorithm_config=deepcopy(algorithm_config);self.output_dir=Path(output_dir);self.output_dir.mkdir(parents=True,exist_ok=True);self.smoke=bool(smoke)
  t=algorithm_config["training"];self.num_envs=int(num_envs or t["num_train_envs"]);self.total_sampled_steps=int(total_sampled_steps or t["total_sampled_steps"]);self.seed=int(t["seed"] if seed is None else seed);self.device=str(device or t["device"])
  cfg=deepcopy(algorithm_config);cfg["training"]["seed"]=self.seed
  needs_compatible_width=bool(cfg.get("modules",{}).get("warm_start",{}).get("enabled",False))
  self.trainer=build_modular_mappo_trainer(cfg,self.device,None if needs_compatible_width else (64 if smoke else None))
  self.rollout_steps=4 if smoke else int(t["rollout_steps"]);self.eval_episodes=min(2,int(t["evaluation_episodes"])) if smoke else int(t["evaluation_episodes"]);self.eval_base=int(algorithm_config["implementation"]["evaluation_seed_base"]);self.evaluation_interval=int(t["evaluation_interval_sampled_steps"]);self.checkpoint_interval=int(algorithm_config["implementation"]["checkpoint_interval_sampled_steps"]);self.next_evaluation=self.evaluation_interval;self.next_checkpoint=self.checkpoint_interval;self.best_key=None
  self.current_stage,self.current_waves=self.trainer.curriculum.stage(0);self.runtime_env_config=self.trainer.curriculum.runtime_config(self.env_config,0);self.vector=None;self._make_vector();self.completed_records=[];self.last_metrics={};self.optimization_history=[];self.module_metrics=[];self.wave_sample_counts=np.zeros(3,dtype=np.int64);self.curriculum_transitions=[{"sampled_steps":0,"stage":self.current_stage,"total_waves":self.current_waves}]
  if warm_start_checkpoint:self.trainer.warm_start_provenance=self.trainer.warm_start.initialize(self.trainer,warm_start_checkpoint)
  if self.trainer.anchor.enabled:
   if not reference_checkpoint:raise ValueError("policy_anchor requires --reference-checkpoint")
   import torch
   state=torch.load(reference_checkpoint,map_location="cpu",weights_only=False);from algorithm.mappo.networks import SharedMAPPOActor
   n=algorithm_config["network"];ref=SharedMAPPOActor(int(n["observation_dim"]),int(n["action_dim"]),int(n["actor_hidden_layers"][0])).to(self.trainer.device);ref.load_state_dict(state["actor"]);self.trainer.anchor.attach(ref,str(reference_checkpoint));self.trainer.anchor_provenance={"reference_checkpoint":str(reference_checkpoint),"source_algorithm":state.get("algorithm")}
 def _make_vector(self):
  if self.vector is not None:self.vector.close()
  self.vector=ParallelVectorEnv(self.num_envs,self.runtime_env_config,self.seed+self.trainer.sampled_steps,range(self.eval_base,self.eval_base+self.eval_episodes));self.observations=self.vector.reset();self.alive=self.vector.current_alive_masks.copy();self.wave=np.ones(self.num_envs,np.int64);self.total=np.full(self.num_envs,self.current_waves,np.int64);self.episode_mask=np.zeros(self.num_envs,np.float32);self.actor_hidden,self.critic_hidden=self.trainer.initial_hidden(self.num_envs);self.episode_returns=np.zeros((self.num_envs,4))
  (self.output_dir/"runtime_env_config.yaml").write_text(yaml.safe_dump(self.runtime_env_config,sort_keys=False),encoding="utf-8")
 def _maybe_curriculum(self):
  stage,waves=self.trainer.curriculum.stage(self.trainer.sampled_steps)
  if (stage,waves)!=(self.current_stage,self.current_waves):self.current_stage,self.current_waves=stage,waves;self.curriculum_transitions.append({"sampled_steps":self.trainer.sampled_steps,"stage":stage,"total_waves":waves});self.runtime_env_config=self.trainer.curriculum.runtime_config(self.env_config,self.trainer.sampled_steps);self._make_vector()
 def collect_rollout(self,steps=None):
  keys=("observations","actions","raw_actions","old_log_probs","rewards","raw_environment_rewards","dones","alive_masks","next_observations","next_alive_masks","wave_indices","total_waves","contexts","next_contexts","actor_hidden_before_step","critic_hidden_before_step","episode_masks");s={k:[] for k in keys}
  for _ in range(int(steps or self.rollout_steps)):
   obs=self.observations.copy();alive=self.alive.copy();ctx=self.trainer.context_numpy(self.wave,self.total);ah=None if self.actor_hidden is None else self.actor_hidden.copy();ch=None if self.critic_hidden is None else self.critic_hidden.copy()
   actions,raw,log,newah=self.trainer.act(obs,alive,False,True,ctx,self.actor_hidden,self.episode_mask);_,newch=self.trainer.values_step(obs,alive,ctx,self.critic_hidden,self.episode_mask)
   result=self.vector.step_batch(actions);done=result.terminated|result.truncated;training,reward_metrics=self.trainer.reward_adapter.adapt(result.rewards,result.infos);nextwave=np.asarray([int(x.get("wave_index",1)) for x in result.infos]);nexttotal=np.asarray([int(x.get("total_waves",self.current_waves)) for x in result.infos]);nextctx=self.trainer.context_numpy(nextwave,nexttotal)
   for k in (1,2,3):self.wave_sample_counts[k-1]+=int((self.wave==k).sum())
   values=(obs,actions,raw,log,training,result.rewards.copy(),done.astype(np.float32),alive,result.transition_next_observations,result.next_alive_masks,self.wave.copy(),self.total.copy(),ctx,nextctx,ah,ch,self.episode_mask.copy())
   for k,v in zip(keys,values):s[k].append(v)
   self.episode_returns+=result.rewards
   for e,is_done in enumerate(done):
    if is_done:self.completed_records.append({"team_episode_return":float(self.episode_returns[e].sum()),**result.infos[e]});self.episode_returns[e].fill(0)
   self.observations=result.observations;self.alive=self.vector.current_alive_masks.copy();self.wave=np.where(done,1,nextwave);self.total=np.where(done,self.current_waves,nexttotal);self.episode_mask=(~done).astype(np.float32)
   self.actor_hidden=self.trainer.recurrent.apply_alive(newah,self.alive);self.critic_hidden=self.trainer.recurrent.apply_alive(newch,self.alive);self.trainer.recurrent.reset_for_episode(self.actor_hidden,done);self.trainer.recurrent.reset_for_episode(self.critic_hidden,done)
   self.module_metrics.append(reward_metrics)
  kwargs={k:(None if not v or v[0] is None else np.asarray(v)) for k,v in s.items()};return ModularRolloutBatch(**kwargs)
 def checkpoint_extra(self):return {"training_seed":self.seed,"training_gamma":self.trainer.gamma,"training_num_envs":self.num_envs,"training_total_sampled_steps":self.total_sampled_steps,"environment_variant":self.env_config.get("environment_variant","direct_v2_3"),"environment_config":self.env_config,"algorithm_config":self.algorithm_config,"environment_config_sha256":config_sha256(self.env_config),"algorithm_config_sha256":config_sha256(self.algorithm_config),"network_architecture":checkpoint_architecture(self.trainer),"curriculum_stage":self.current_stage,"current_total_waves":self.current_waves}
 def save(self,name="final.pt"):self.trainer.save(self.output_dir/name,self.checkpoint_extra())
 def resume(self,path):
  extra=self.trainer.load(path);self.current_stage=int(extra.get("curriculum_stage",self.current_stage));self.current_waves=int(extra.get("current_total_waves",self.current_waves));self.next_evaluation=((self.trainer.sampled_steps//self.evaluation_interval)+1)*self.evaluation_interval;self.next_checkpoint=((self.trainer.sampled_steps//self.checkpoint_interval)+1)*self.checkpoint_interval;self.runtime_env_config=self.trainer.curriculum.runtime_config(self.env_config,self.trainer.sampled_steps);self._make_vector()
 def evaluate(self):return evaluate_modular(self.trainer,self.env_config,range(self.eval_base,self.eval_base+self.eval_episodes))
 def _record_evaluation(self):
  result={"sampled_steps":self.trainer.sampled_steps,**self.evaluate()};path=self.output_dir/"evaluation_history.csv";exists=path.exists()
  with path.open("a",newline="",encoding="utf-8") as stream:
   writer=csv.DictWriter(stream,fieldnames=list(result));
   if not exists:writer.writeheader()
   writer.writerow(result)
  key=(result.get("clear_wave_3_probability",result.get("win_rate",0.)),result.get("average_waves_cleared",0.),result["average_return"])
  if self.best_key is None or key>self.best_key:self.best_key=key;self.trainer.save(self.output_dir/"best_eval.pt",{**self.checkpoint_extra(),"evaluation":result})
  return result
 def run(self):
  try:
   while self.trainer.sampled_steps<self.total_sampled_steps:
    self._maybe_curriculum();remaining=self.total_sampled_steps-self.trainer.sampled_steps;vsteps=min(self.rollout_steps,max(1,int(np.ceil(remaining/self.num_envs))));batch=self.collect_rollout(vsteps);self.last_metrics=self.trainer.update(batch);inc=vsteps*self.num_envs;self.trainer.sampled_steps=min(self.total_sampled_steps,self.trainer.sampled_steps+inc);self.trainer.vector_steps+=vsteps;row={"sampled_steps":self.trainer.sampled_steps,**self.last_metrics};self.optimization_history.append(row)
    with (self.output_dir/"optimization_metrics.jsonl").open("a",encoding="utf-8") as stream:stream.write(json.dumps(row)+"\n")
    if self.trainer.sampled_steps>=self.next_checkpoint:
     self.save(f"checkpoint_{self.trainer.sampled_steps}.pt");self.next_checkpoint+=self.checkpoint_interval
    if self.trainer.sampled_steps>=self.next_evaluation:
     self._record_evaluation();self.next_evaluation+=self.evaluation_interval
   evaluation=self._record_evaluation();self.save();total=max(int(self.wave_sample_counts.sum()),1);reward_total={k:float(sum(row.get(k,0.) for row in self.module_metrics)) for k in ("reward_bonus_total","reward_bonus_wave1","reward_bonus_wave2","reward_bonus_wave3")};summary={"sampled_steps":self.trainer.sampled_steps,"evaluation":evaluation,"optimization":self.last_metrics,"optimization_updates":len(self.optimization_history),"module_protocol":self.trainer.module_protocol(),"curriculum_stage":self.current_stage,"current_total_waves":self.current_waves,"curriculum_transitions":self.curriculum_transitions,"wave_samples":{f"wave_{k}":int(self.wave_sample_counts[k-1]) for k in (1,2,3)},"wave_fractions":{f"wave_{k}":float(self.wave_sample_counts[k-1]/total) for k in (1,2,3)},"reward_adapter_totals":reward_total,"warm_start_provenance":self.trainer.warm_start_provenance,"anchor_provenance":self.trainer.anchor_provenance};(self.output_dir/"run_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8");return summary
  finally:self.vector.close()

__all__=["ModularMAPPOTrainingRunner"]
