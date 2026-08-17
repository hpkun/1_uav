"""Formal 24-environment MADSAC runner with invariant update/data scheduling."""
from __future__ import annotations
import csv,hashlib,json
from pathlib import Path
from typing import Any
import numpy as np
from ..madsac.trainer import MADSACTrainer
from .evaluator import evaluate
from .vector_env import SyncVectorEnv


def config_signature(env_config:dict,algorithm_config:dict)->str:
    payload=json.dumps({"environment":env_config,"algorithm":algorithm_config},sort_keys=True,separators=(",",":"))
    return hashlib.sha256(payload.encode()).hexdigest()


class PaperTrainingRunner:
    def __init__(self,env_config:dict,algorithm_config:dict,num_envs:int|None=None,total_env_steps:int|None=None,device:str|None=None,seed:int|None=None,run_id:int=0,output_dir:str|Path|None=None,smoke:bool=False)->None:
        self.env_config,self.algorithm_config=env_config,algorithm_config; t=algorithm_config["training"]; a=algorithm_config["reproduction_assumptions"]
        self.num_envs=int(num_envs or t["num_train_envs"]); self.total_env_steps=int(total_env_steps or t["total_env_steps"]); self.device=str(device or t["device"]); self.seed=int(t["seed"] if seed is None else seed); self.run_id=int(run_id)
        self.batch_size=16 if smoke else int(t["batch_size"]); self.learning_starts=16 if smoke else int(a["learning_starts"]); self.updates_per_transition=.05 if smoke else float(a["updates_per_transition"])
        if self.learning_starts<self.batch_size: raise ValueError("learning_starts must be >= batch_size")
        replay_capacity=50_000 if smoke else int(t["replay_buffer_size"]); sig=config_signature(env_config,algorithm_config)
        self.trainer=MADSACTrainer(hidden_dim=int(algorithm_config["network"]["actor_hidden_layers"][0]),attention_heads=int(algorithm_config["network"]["attention_heads"]),learning_rate=float(t["learning_rate"]),gamma=float(t["gamma"]),tau=float(t["tau"]),alpha=float(t["alpha"]),policy_delay=int(a["policy_delay"]),replay_capacity=replay_capacity,batch_size=self.batch_size,device=self.device,seed=self.seed,actor_activation=a["actor_activation"],critic_activation=a["critic_activation"],log_std_min=float(a["log_std_min"]),log_std_max=float(a["log_std_max"]),config_signature=sig)
        base_seed=self.seed+self.run_id*int(a["seed_stride"]); self.vector=SyncVectorEnv(self.num_envs,env_config,base_seed,int(a["seed_stride"])); self.observations=self.vector.reset(); self.alive_masks=self.vector.current_alive_masks.copy()
        eval_base=int(a["evaluation_seed_base"]); self.evaluation_seeds=list(range(eval_base,eval_base+int(t["evaluation_episodes"])))
        if {self.vector.seed_for(i,0) for i in range(self.num_envs)}.intersection(self.evaluation_seeds): raise ValueError("training and evaluation seeds overlap")
        base=Path(output_dir or t["output_dir"]); self.output_dir=base/f"run_{self.run_id}_seed_{self.seed}"; self.output_dir.mkdir(parents=True,exist_ok=True)
        self.update_credit=0.0; self.next_evaluation=int(t["evaluation_interval_env_steps"]); self.episode_returns=np.zeros(self.num_envs); self.completed_records=[]; self.completed_episode_count=0; self.last_metrics={}; self.dead_slot_samples=0
        self.trainer.run_metadata={"seed":self.seed,"run_id":self.run_id,"num_envs":self.num_envs,"base_seed":base_seed,"evaluation_seeds":self.evaluation_seeds,"resume_vector_env_behavior":a["resume_vector_env_behavior"]}

    def startup_summary(self)->dict[str,Any]:
        return {"device":self.device,"num_envs":self.num_envs,"base_seed":self.vector.base_seed,"total_env_steps":self.total_env_steps,"batch":self.batch_size,"buffer":self.trainer.replay.capacity,"learning_starts":self.learning_starts,"policy_delay":self.trainer.policy_delay,"updates_per_transition":self.updates_per_transition,"run_id":self.run_id}

    def _scheduled_updates(self,new_transitions:int)->int:
        self.update_credit+=new_transitions*self.updates_per_transition; count=int(self.update_credit); self.update_credit-=count; return count

    def vector_step(self)->dict[str,Any]:
        actions=self.trainer.act(self.observations,self.alive_masks); result=self.vector.step_batch(actions)
        executed=np.stack([info["executed_red_actions"] for info in result.infos]); dones=result.terminated|result.truncated
        self.trainer.replay.push_batch(self.observations,executed,result.rewards,result.transition_next_observations,dones,result.alive_masks,result.next_alive_masks)
        self.episode_returns+=result.rewards[:,0]
        for i,done in enumerate(dones):
            if done: self.completed_records.append({"episode_return":float(self.episode_returns[i]),**result.infos[i]}); self.completed_episode_count+=1; self.episode_returns[i]=0
        self.observations=result.observations; self.alive_masks=self.vector.current_alive_masks.copy(); self.trainer.sampled_env_steps+=self.num_envs; self.trainer.vector_steps+=1
        self.dead_slot_samples+=int((result.alive_masks<.5).sum())
        updates=0
        if self.trainer.replay.size>=self.learning_starts:
            for _ in range(self._scheduled_updates(self.num_envs)): self.last_metrics=self.trainer.update(); updates+=1
        completed_now=[self.completed_records[-j] for j in range(1,int(dones.sum())+1)] if dones.any() else []
        mean_completed=lambda key:float(np.mean([r[key] for r in completed_now])) if completed_now else 0.0
        record={"sampled_env_steps":self.trainer.sampled_env_steps,"vector_steps":self.trainer.vector_steps,"gradient_updates":self.trainer.update_count,"actor_updates":self.trainer.actor_update_count,"replay_size":self.trainer.replay.size,"mean_reward":float(result.rewards[:,0].mean()),"completed_episodes":int(dones.sum()),"episode_return":mean_completed("episode_return"),"red_win_rate":mean_completed("red_win"),"blue_win_rate":mean_completed("blue_win"),"draw_timeout_rate":mean_completed("draw_or_timeout"),"red_attack_kills":mean_completed("red_attack_kills"),"blue_attack_kills":mean_completed("blue_attack_kills"),"red_boundary_losses":mean_completed("red_boundary_losses"),"blue_boundary_losses":mean_completed("blue_boundary_losses"),"red_survivors":mean_completed("red_survivors"),"blue_survivors":mean_completed("blue_survivors"),"mean_episode_length":mean_completed("episode_length"),**self.last_metrics}
        with (self.output_dir/"training_metrics.jsonl").open("a",encoding="utf-8") as stream: stream.write(json.dumps(record)+"\n")
        return {"new_transitions":self.num_envs,"gradient_updates":updates,"completed":int(dones.sum()),"dead_slots":int((result.alive_masks<.5).sum()),"metrics":self.last_metrics}

    def run(self)->dict[str,Any]:
        while self.trainer.sampled_env_steps<self.total_env_steps:
            self.vector_step()
            if self.trainer.sampled_env_steps>=self.next_evaluation:
                record={"sampled_env_steps":self.trainer.sampled_env_steps,**evaluate(self.trainer,self.env_config,self.evaluation_seeds)}; self.trainer.evaluation_history.append(record); self._write_evaluation(); self.next_evaluation+=int(self.algorithm_config["training"]["evaluation_interval_env_steps"])
        self.save_checkpoints(); return self.summary()

    def _write_evaluation(self)->None:
        (self.output_dir/"evaluation_history.json").write_text(json.dumps(self.trainer.evaluation_history,indent=2),encoding="utf-8")
        if self.trainer.evaluation_history:
            with (self.output_dir/"evaluation_history.csv").open("w",newline="",encoding="utf-8") as f:
                writer=csv.DictWriter(f,fieldnames=list(self.trainer.evaluation_history[0])); writer.writeheader(); writer.writerows(self.trainer.evaluation_history)

    def save_checkpoints(self)->None:
        self.trainer.episode_counters={"per_env":self.vector.episode_indices.tolist(),"completed":self.completed_episode_count}; self.trainer.run_metadata.update({"update_credit":self.update_credit,"next_evaluation":self.next_evaluation,"dead_slot_samples":self.dead_slot_samples}); self.trainer.save(self.output_dir/"latest_full_resume.pt",True); self.trainer.save(self.output_dir/"latest_evaluation.pt",False)

    def resume(self,path:str|Path)->None:
        self.trainer.load(path,require_replay=True)
        if int(self.trainer.run_metadata.get("num_envs",self.num_envs))!=self.num_envs: raise RuntimeError("resume num_envs mismatch")
        self.update_credit=float(self.trainer.run_metadata.get("update_credit",0)); self.next_evaluation=int(self.trainer.run_metadata.get("next_evaluation",self.next_evaluation)); self.dead_slot_samples=int(self.trainer.run_metadata.get("dead_slot_samples",0)); self.completed_episode_count=int(self.trainer.episode_counters.get("completed",0)); self.vector.episode_indices=np.asarray(self.trainer.episode_counters.get("per_env",[0]*self.num_envs),np.int64); self.observations=self.vector.reset(); self.alive_masks=self.vector.current_alive_masks.copy()

    def summary(self)->dict[str,Any]:
        records=self.completed_records; mean=lambda key:float(np.mean([r[key] for r in records])) if records else 0.0
        return {**self.startup_summary(),"sampled_env_steps":self.trainer.sampled_env_steps,"vector_steps":self.trainer.vector_steps,"gradient_updates":self.trainer.update_count,"actor_updates":self.trainer.actor_update_count,"replay_size":self.trainer.replay.size,"dead_slot_samples":self.dead_slot_samples,"completed_episodes":self.completed_episode_count,"mean_episode_return":mean("episode_return"),"red_win_rate":mean("red_win"),"blue_win_rate":mean("blue_win"),"draw_timeout_rate":mean("draw_or_timeout"),"red_attack_kills":mean("red_attack_kills"),"blue_attack_kills":mean("blue_attack_kills"),"red_boundary_losses":mean("red_boundary_losses"),"blue_boundary_losses":mean("blue_boundary_losses"),"red_survivors":mean("red_survivors"),"blue_survivors":mean("blue_survivors"),"mean_episode_length":mean("episode_length"),"last_update_metrics":self.last_metrics,"evaluation_history":self.trainer.evaluation_history}
