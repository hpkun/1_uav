"""End-to-end deterministic feed-forward MAPPO runner."""

from __future__ import annotations
import random,time
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from uav_env.algorithms.mappo.adapter import MAPPOEnvAdapter,SyncCombatVectorEnv
from uav_env.algorithms.mappo.checkpoint import load_checkpoint,save_checkpoint
from uav_env.algorithms.mappo.metrics import append_csv,evaluation_key
from uav_env.algorithms.mappo.networks import CentralizedCritic,SharedActor
from uav_env.algorithms.mappo.rollout_buffer import RolloutBuffer
from uav_env.algorithms.mappo.trainer import MAPPOTrainer
from uav_env.algorithms.mappo.value_normalizer import ValueNormalizer
from uav_env.envs import make_1v1_env,make_2v2_env


def resolve_device(name: str) -> torch.device:
    return torch.device("cuda" if name=="auto" and torch.cuda.is_available() else "cpu" if name=="auto" else name)

class MAPPORunner:
    def __init__(self, config: dict[str,Any], run_name: str, output_root: str|Path="outputs/mappo") -> None:
        self.config=config; self.seed=int(config["seed"]); random.seed(self.seed); np.random.seed(self.seed); torch.manual_seed(self.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(self.seed)
        torch.use_deterministic_algorithms(True,warn_only=True); self.device=resolve_device(str(config["device"])); print(f"MAPPO device: {self.device}")
        env_cfg=config["environment"]
        def factory():
            env=make_1v1_env(env_cfg["scenario"],env_cfg["opponent"]) if env_cfg["kind"]=="1v1" else make_2v2_env(env_cfg["scenario"],env_cfg["opponent"])
            return MAPPOEnvAdapter(env)
        probe=factory(); self.num_agents,self.obs_dim,self.state_dim=probe.num_agents,probe.obs_dim,probe.state_dim
        self.vector=SyncCombatVectorEnv([factory for _ in range(int(config["num_envs"]))],self.seed)
        self.actor=SharedActor(self.obs_dim,hidden_sizes=config["actor_hidden_sizes"],activation=config["activation"]).to(self.device)
        self.critic=CentralizedCritic(self.state_dim,self.num_agents,config["critic_hidden_sizes"],config["activation"]).to(self.device)
        self.normalizer=ValueNormalizer(); self.trainer=MAPPOTrainer(self.actor,self.critic,config,self.normalizer,self.device)
        run_id=str(config.get("run_id") or datetime.now().strftime("%Y%m%d_%H%M%S")); self.output_dir=Path(output_root)/run_name/run_id; self.output_dir.mkdir(parents=True,exist_ok=True)
        (self.output_dir/"config.yaml").write_text(yaml.safe_dump(config,sort_keys=False),encoding="utf-8"); self.writer=SummaryWriter(self.output_dir/"tensorboard")
        self.environment_steps=0; self.update_index=0; self.best_evaluation: dict[str,Any]|None=None; self.current=self.vector.reset(); self.episodes=0
        self.episode_return_accumulators=np.zeros(int(config["num_envs"]),dtype=np.float64)
        self.last_evaluation_step: int | None = None

    def resume(self,path: str,actor_only: bool=False) -> None:
        data=load_checkpoint(path,self.actor,self.critic,self.trainer.actor_optimizer,self.trainer.critic_optimizer,self.normalizer,actor_only,self.device)
        if not actor_only:
            self.environment_steps=int(data["environment_steps"]); self.update_index=int(data["update_index"]); self.best_evaluation=data["best_evaluation"]
            runner_state=data.get("runner_state")
            if runner_state is not None:
                self.vector.envs=runner_state["vector_envs"]
                self.current=runner_state["current"]
                self.episodes=int(runner_state["episodes"])
                self.episode_return_accumulators=np.asarray(runner_state.get("episode_return_accumulators",np.zeros(int(self.config["num_envs"]))),dtype=np.float64)
                self.last_evaluation_step=runner_state.get("last_evaluation_step")

    def _values(self,states: np.ndarray) -> np.ndarray:
        with torch.no_grad(): values=self.critic(torch.as_tensor(states,device=self.device))
        if self.config.get("use_value_normalization",True): values=self.normalizer.denormalize(values)
        return values.cpu().numpy().astype(np.float32)

    def collect(self) -> tuple[RolloutBuffer,dict[str,float]]:
        t=int(self.config["rollout_length"]); e=int(self.config["num_envs"]); buffer=RolloutBuffer(t,e,self.num_agents,self.obs_dim,self.state_dim)
        buffer.set_initial(self.current["local_obs"],self.current["global_state"],self.current["available_actions"]); saturation=[]; rollout_returns=[]
        for _ in range(t):
            obs=torch.as_tensor(self.current["local_obs"],device=self.device); states=torch.as_tensor(self.current["global_state"],device=self.device); available=torch.as_tensor(self.current["available_actions"],device=self.device)
            with torch.no_grad():
                dist=torch.distributions.Categorical(logits=self.actor(obs,available)); actions=dist.sample(); log_probs=dist.log_prob(actions); values=self.critic(states)
                if self.config.get("use_value_normalization",True): values=self.normalizer.denormalize(values)
            result=self.vector.step(actions.cpu().numpy()); terminal_values=np.zeros_like(values.cpu().numpy())
            self.episode_return_accumulators += np.asarray(result["rewards"],dtype=np.float64).sum(axis=1)
            for index,step in enumerate(result["terminal_steps"]):
                saturation.append(float(step.info.get("observation_saturation_ratio",np.mean(step.info.get("local_observation_saturation_ratio",[0.0])))))
                if step.truncated: terminal_values[index]=self._values(step.global_state[None,:])[0]
                if step.terminated or step.truncated:
                    self.episodes+=1; rollout_returns.append(float(self.episode_return_accumulators[index])); self.episode_return_accumulators[index]=0.0
            critic_masks=np.ones_like(self.current["alive_masks"],dtype=np.float32)
            buffer.insert(actions.cpu().numpy(),log_probs.cpu().numpy(),values.cpu().numpy(),result["rewards"],result["terminated"],result["truncated"],self.current["alive_masks"],critic_masks,result["next_local_obs"],result["next_global_state"],result["next_available_actions"],terminal_values)
            self.current={"local_obs":result["next_local_obs"],"global_state":result["next_global_state"],"alive_masks":result["next_alive_masks"],"available_actions":result["next_available_actions"]}
        buffer.finish(self._values(self.current["global_state"]),float(self.config["gamma"]),float(self.config["gae_lambda"]))
        return buffer,{"rollout_return_mean":float(np.mean(rollout_returns)) if rollout_returns else 0.0,"observation_saturation_mean":float(np.mean(saturation)) if saturation else 0.0,"observation_saturation_max":float(np.max(saturation)) if saturation else 0.0}

    def evaluate(self,episodes: int|None=None,seed_start: int=100000,deterministic: bool|None=None) -> dict[str,float]:
        count=int(episodes or self.config["evaluation_episodes"]); env_cfg=self.config["environment"]; outcomes=[]; returns=[]; steps=[]; red_crashes=[]; blue_crashes=[]; saturation=[]; frequencies=np.zeros(15); red_survivors=[]; blue_survivors=[]; damages=[]; hits=[]
        deterministic = bool(self.config.get("deterministic_evaluation", True)) if deterministic is None else deterministic
        for episode in range(count):
            env=MAPPOEnvAdapter(make_1v1_env(env_cfg["scenario"],env_cfg["opponent"]) if env_cfg["kind"]=="1v1" else make_2v2_env(env_cfg["scenario"],env_cfg["opponent"])); current=env.reset(seed_start+episode); total=0.; done=False
            while not done:
                with torch.no_grad():
                    logits=self.actor(torch.as_tensor(current.local_obs,device=self.device),torch.as_tensor(current.available_action_mask,device=self.device))
                    action=(torch.argmax(logits,-1) if deterministic else torch.distributions.Categorical(logits=logits).sample()).cpu().numpy()
                for value in action: frequencies[int(value)]+=1
                current=env.step(action); total+=float(current.agent_rewards.sum()); done=current.terminated or current.truncated
            outcome=current.info["outcome"]; outcomes.append(outcome); returns.append(total); steps.append(outcome.decision_steps)
            stats=current.info.get("statistics",{}); reason=str(outcome.termination_reason)
            if self.num_agents==1:
                red_crashes.append(float(reason=="red_ground_crash")); blue_crashes.append(float(reason=="blue_ground_crash")); red_survivors.append(float(outcome.red_alive)); blue_survivors.append(float(outcome.blue_alive)); damages.append(float(stats.get("red_effective_damage",0.0))); hits.append(float(stats.get("red_hits",0)))
            else:
                aircraft=stats["aircraft"]; red_crashes.append(float(sum(aircraft[f"red_{i}"]["ground_crashes"] for i in range(2))>0)); blue_crashes.append(float(sum(aircraft[f"blue_{i}"]["ground_crashes"] for i in range(2))>0)); red_survivors.append(float(outcome.red_survivors)); blue_survivors.append(float(outcome.blue_survivors)); damages.append(float(sum(aircraft[f"red_{i}"]["effective_damage"] for i in range(2)))); hits.append(float(sum(aircraft[f"red_{i}"]["hits"] for i in range(2))))
            saturation.append(float(current.info.get("observation_saturation_ratio",np.mean(current.info.get("local_observation_saturation_ratio",[0.])))))
        winners=[o.winner for o in outcomes]
        result={"red_win_rate":winners.count("red")/count,"blue_win_rate":winners.count("blue")/count,"draw_rate":winners.count("draw")/count,"timeout_rate":sum(o.termination_reason=="timeout" for o in outcomes)/count,"red_crash_rate":float(np.mean(red_crashes)),"blue_crash_rate":float(np.mean(blue_crashes)),"mean_episode_return":float(np.mean(returns)),"mean_agent_return":float(np.mean(returns))/self.num_agents,"mean_episode_steps":float(np.mean(steps)),"mean_red_survivors":float(np.mean(red_survivors)),"mean_blue_survivors":float(np.mean(blue_survivors)),"mean_effective_damage":float(np.mean(damages)),"mean_hits":float(np.mean(hits)),"mean_observation_saturation_ratio":float(np.mean(saturation))}
        result.update({f"action_{i}_frequency":float(frequencies[i]/max(frequencies.sum(),1)) for i in range(15)}); return result

    def run(self) -> Path:
        started=time.time(); start_steps=self.environment_steps; total=int(self.config["total_env_steps"])
        if self.environment_steps == 0:
            self._save("initial.pt")
        while self.environment_steps<total:
            if self.config.get("linear_lr_decay",False):
                fraction=max(0.0,1.0-self.environment_steps/max(total,1))
                for group in self.trainer.actor_optimizer.param_groups: group["lr"]=float(self.config["actor_lr"])*fraction
                for group in self.trainer.critic_optimizer.param_groups: group["lr"]=float(self.config["critic_lr"])*fraction
            buffer,rollout=self.collect(); metrics=self.trainer.update(buffer); self.environment_steps+=int(self.config["rollout_length"])*int(self.config["num_envs"]); self.update_index+=1
            elapsed=time.time()-started; row={"environment_steps":self.environment_steps,"decisions":self.environment_steps*self.num_agents,"episodes":self.episodes,"update_index":self.update_index,"wall_time":elapsed,"samples_per_second":(self.environment_steps-start_steps)/max(elapsed,1e-9),**metrics,**rollout}; append_csv(self.output_dir/"metrics.csv",row)
            for key,value in row.items():
                if isinstance(value,(int,float)): self.writer.add_scalar(key,value,self.environment_steps)
            if self.environment_steps%int(self.config["evaluation_interval"])<int(self.config["rollout_length"])*int(self.config["num_envs"]):
                evaluation={"environment_steps":self.environment_steps,**self.evaluate()}; append_csv(self.output_dir/"evaluations.csv",evaluation)
                self.last_evaluation_step=self.environment_steps
                if self.best_evaluation is None or evaluation_key(evaluation)>evaluation_key(self.best_evaluation): self.best_evaluation=evaluation; self._save("best.pt")
            if self.environment_steps%int(self.config["checkpoint_interval"])<int(self.config["rollout_length"])*int(self.config["num_envs"]): self._save(f"step_{self.environment_steps}.pt")
            self._save("last.pt")
        if self.last_evaluation_step != self.environment_steps:
            evaluation={"environment_steps":self.environment_steps,**self.evaluate()};append_csv(self.output_dir/"evaluations.csv",evaluation);self.last_evaluation_step=self.environment_steps
            if self.best_evaluation is None or evaluation_key(evaluation)>evaluation_key(self.best_evaluation):self.best_evaluation=evaluation;self._save("best.pt")
            self._save("last.pt")
        summary={"environment_steps":self.environment_steps,"updates":self.update_index,"episodes":self.episodes,"device":str(self.device),"best_evaluation":self.best_evaluation,"actor_parameters":sum(p.numel() for p in self.actor.parameters()),"critic_parameters":sum(p.numel() for p in self.critic.parameters())}; (self.output_dir/"final_summary.yaml").write_text(yaml.safe_dump(summary,sort_keys=False),encoding="utf-8"); self.writer.close(); return self.output_dir

    def _save(self,name: str) -> None:
        runner_state={"vector_envs":self.vector.envs,"current":self.current,"episodes":self.episodes,"episode_return_accumulators":self.episode_return_accumulators,"last_evaluation_step":self.last_evaluation_step}
        save_checkpoint(self.output_dir/"checkpoints"/name,self.actor,self.critic,self.trainer.actor_optimizer,self.trainer.critic_optimizer,self.normalizer,self.config,self.environment_steps,self.update_index,self.best_evaluation,runner_state)
