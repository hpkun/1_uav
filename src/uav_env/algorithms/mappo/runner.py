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

from uav_env.algorithms.mappo.adapter import CombatEnvDescription,MAPPOEnvAdapter,ParallelCombatVectorEnv,SyncCombatVectorEnv,make_adapter_from_description
from uav_env.algorithms.mappo.checkpoint import load_checkpoint,save_checkpoint
from uav_env.algorithms.mappo.metrics import append_csv,combat_outcome_rates,evaluation_key
from uav_env.algorithms.mappo.networks import CentralizedCritic,SharedActor
from uav_env.algorithms.mappo.rollout_buffer import RolloutBuffer
from uav_env.algorithms.mappo.trainer import MAPPOTrainer
from uav_env.algorithms.mappo.value_normalizer import ValueNormalizer


def resolve_device(name: str) -> torch.device:
    return torch.device("cuda" if name=="auto" and torch.cuda.is_available() else "cpu" if name=="auto" else name)

class MAPPORunner:
    def __init__(self, config: dict[str,Any], run_name: str, output_root: str|Path="outputs/mappo") -> None:
        self.config=config; self.seed=int(config["seed"]); random.seed(self.seed); np.random.seed(self.seed); torch.manual_seed(self.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(self.seed)
        torch.use_deterministic_algorithms(True,warn_only=True); self.device=resolve_device(str(config["device"])); print(f"MAPPO device: {self.device}")
        env_cfg=config["environment"]
        description=CombatEnvDescription(str(env_cfg["kind"]),str(env_cfg["scenario"]),str(env_cfg["opponent"]),env_cfg.get("multi_terminal_reward_profile"))
        probe=make_adapter_from_description(description); self.num_agents,self.obs_dim,self.state_dim=probe.num_agents,probe.obs_dim,probe.state_dim; probe.env.close()
        if config.get("vector_env","sync")=="parallel":
            self.vector=ParallelCombatVectorEnv(description,int(config["num_envs"]),self.seed)
        elif config.get("vector_env","sync")=="sync":
            self.vector=SyncCombatVectorEnv([lambda description=description: make_adapter_from_description(description) for _ in range(int(config["num_envs"]))],self.seed)
        else:
            raise ValueError("vector_env must be sync or parallel")
        self.actor=SharedActor(self.obs_dim,hidden_sizes=config["actor_hidden_sizes"],activation=config["activation"]).to(self.device)
        self.critic=CentralizedCritic(self.state_dim,self.num_agents,config["critic_hidden_sizes"],config["activation"]).to(self.device)
        self.normalizer=ValueNormalizer(); self.trainer=MAPPOTrainer(self.actor,self.critic,config,self.normalizer,self.device)
        run_id=str(config.get("run_id") or datetime.now().strftime("%Y%m%d_%H%M%S")); self.output_dir=Path(output_root)/run_name/run_id; self.output_dir.mkdir(parents=True,exist_ok=True)
        (self.output_dir/"config.yaml").write_text(yaml.safe_dump(config,sort_keys=False),encoding="utf-8"); self.writer=SummaryWriter(self.output_dir/"tensorboard")
        self.environment_steps=0; self.update_index=0; self.best_evaluation: dict[str,Any]|None=None; self.current=self.vector.reset(); self.episodes=0
        self.episode_return_accumulators=np.zeros(int(config["num_envs"]),dtype=np.float64)
        self.agent_sum_return_accumulators=np.zeros(int(config["num_envs"]),dtype=np.float64)
        self.last_evaluation_step: int | None = None

    def resume(self,path: str,actor_only: bool=False) -> None:
        data=load_checkpoint(path,self.actor,self.critic,self.trainer.actor_optimizer,self.trainer.critic_optimizer,self.normalizer,actor_only,self.device)
        if not actor_only:
            self.environment_steps=int(data["environment_steps"]); self.update_index=int(data["update_index"]); self.best_evaluation=data["best_evaluation"]
            runner_state=data.get("runner_state")
            if runner_state is not None:
                if "vector_env_state" in runner_state:
                    self.vector.set_state(runner_state["vector_env_state"])
                elif "vector_envs" in runner_state and isinstance(self.vector,SyncCombatVectorEnv):
                    self.vector.set_state(runner_state["vector_envs"])
                self.current=runner_state["current"]
                self.episodes=int(runner_state["episodes"])
                self.episode_return_accumulators=np.asarray(runner_state.get("episode_return_accumulators",np.zeros(int(self.config["num_envs"]))),dtype=np.float64)
                self.agent_sum_return_accumulators=np.asarray(runner_state.get("agent_sum_return_accumulators",self.episode_return_accumulators*self.num_agents),dtype=np.float64)
                self.last_evaluation_step=runner_state.get("last_evaluation_step")
                if "trainer_minibatch_rng_state" in runner_state:
                    self.trainer.minibatch_rng.bit_generator.state=runner_state["trainer_minibatch_rng_state"]

    def _values(self,states: np.ndarray) -> np.ndarray:
        with torch.no_grad(): values=self.critic(torch.as_tensor(states,device=self.device))
        return values.cpu().numpy().astype(np.float32)

    def collect(self) -> tuple[RolloutBuffer,dict[str,float]]:
        t=int(self.config["rollout_length"]); e=int(self.config["num_envs"]); buffer=RolloutBuffer(t,e,self.num_agents,self.obs_dim,self.state_dim)
        buffer.set_initial(self.current["local_obs"],self.current["global_state"],self.current["available_actions"]); saturation=[]; rollout_returns=[]; rollout_agent_sum_returns=[]; entropies=[]; action_counts=np.zeros(15); raw_dense=[]; assigned_dense=[]; event_rewards=[]; terminal_rewards=[]; absolute_rewards=[]; episode_timeouts=[]; episode_crashes=[]; episode_damages=[]; episode_hits=[]; attack_occupancies=[]
        for _ in range(t):
            obs=torch.as_tensor(self.current["local_obs"],device=self.device); states=torch.as_tensor(self.current["global_state"],device=self.device); available=torch.as_tensor(self.current["available_actions"],device=self.device)
            with torch.no_grad():
                dist=torch.distributions.Categorical(logits=self.actor(obs,available)); actions=dist.sample(); log_probs=dist.log_prob(actions); values=self.critic(states); active=np.asarray(self.current["alive_masks"],dtype=bool); entropies.extend(dist.entropy().cpu().numpy()[active].tolist())
            result=self.vector.step(actions.cpu().numpy()); terminal_values=np.zeros_like(values.cpu().numpy())
            for value in actions.cpu().numpy()[active]: action_counts[int(value)]+=1
            self.episode_return_accumulators += np.asarray(result["team_rewards"],dtype=np.float64)
            self.agent_sum_return_accumulators += np.asarray(result["agent_reward_sums"],dtype=np.float64)
            for index,step in enumerate(result["terminal_steps"]):
                saturation.append(float(step.info.get("observation_saturation_ratio",np.mean(step.info.get("local_observation_saturation_ratio",[0.0])))))
                if self.num_agents==1 and "reward_breakdown" in step.info:
                    breakdown=step.info["reward_breakdown"]; raw_dense.append(float(breakdown.dense)); assigned_dense.append(float(breakdown.dense)); event_rewards.append(float(breakdown.total-breakdown.dense-breakdown.terminal)); terminal_rewards.append(float(breakdown.terminal)); absolute_rewards.append(abs(float(breakdown.total)))
                elif "agent_reward_breakdowns" in step.info:
                    for breakdown in step.info["agent_reward_breakdowns"].values():
                        raw_dense.append(float(breakdown.raw_dense)); assigned_dense.append(float(breakdown.assigned_dense)); event_rewards.append(float(breakdown.event)); terminal_rewards.append(float(breakdown.terminal)); absolute_rewards.append(abs(float(breakdown.total)))
                if step.truncated: terminal_values[index]=self._values(step.global_state[None,:])[0]
                if step.terminated or step.truncated:
                    self.episodes+=1; rollout_returns.append(float(self.episode_return_accumulators[index])); rollout_agent_sum_returns.append(float(self.agent_sum_return_accumulators[index])); self.episode_return_accumulators[index]=0.0; self.agent_sum_return_accumulators[index]=0.0
                    outcome=step.info["outcome"]; statistics=step.info.get("statistics",{}); reason=str(outcome.termination_reason); episode_timeouts.append(float(reason=="timeout"))
                    if self.num_agents==1:
                        episode_crashes.append(float(reason in {"red_ground_crash","blue_ground_crash"})); episode_damages.append(float(statistics.get("red_effective_damage",0.0))); episode_hits.append(float(statistics.get("red_hits",0.0))); attack_occupancies.append(float(statistics.get("red_attack_area_steps",0))/max(float(outcome.decision_steps),1.0))
                    else:
                        aircraft=statistics.get("aircraft",{}); episode_crashes.append(float(any(float(values.get("ground_crashes",0))>0 for values in aircraft.values()))); episode_damages.append(float(sum(float(aircraft.get(f"red_{i}",{}).get("effective_damage",0.0)) for i in range(self.num_agents)))); episode_hits.append(float(sum(float(aircraft.get(f"red_{i}",{}).get("hits",0.0)) for i in range(self.num_agents)))); attack_occupancies.append(float(sum(float(aircraft.get(f"red_{i}",{}).get("attack_area_steps",0.0)) for i in range(self.num_agents)))/max(float(outcome.decision_steps*self.num_agents),1.0))
            critic_masks=np.ones_like(self.current["alive_masks"],dtype=np.float32)
            buffer.insert(actions.cpu().numpy(),log_probs.cpu().numpy(),values.cpu().numpy(),result["rewards"],result["terminated"],result["truncated"],self.current["alive_masks"],critic_masks,result["next_local_obs"],result["next_global_state"],result["next_available_actions"],terminal_values)
            self.current={"local_obs":result["next_local_obs"],"global_state":result["next_global_state"],"alive_masks":result["next_alive_masks"],"available_actions":result["next_available_actions"]}
        buffer.finish(self._values(self.current["global_state"]),float(self.config["gamma"]),float(self.config["gae_lambda"]))
        diagnostics={"rollout_return_mean":float(np.mean(rollout_returns)) if rollout_returns else 0.0,"rollout_team_episode_return_mean":float(np.mean(rollout_returns)) if rollout_returns else 0.0,"rollout_agent_sum_episode_return_mean":float(np.mean(rollout_agent_sum_returns)) if rollout_agent_sum_returns else 0.0,"rollout_mean_per_agent_episode_return":float(np.mean(rollout_agent_sum_returns))/self.num_agents if rollout_agent_sum_returns else 0.0,"rollout_episode_count":float(len(rollout_returns)),"observation_saturation_mean":float(np.mean(saturation)) if saturation else 0.0,"observation_saturation_max":float(np.max(saturation)) if saturation else 0.0,"rollout_action_entropy":float(np.mean(entropies)) if entropies else 0.0,"terminal_reward_absolute_proportion":float(np.sum(np.abs(terminal_rewards))/max(np.sum(absolute_rewards),1e-12)),"attack_area_occupancy":float(np.mean(attack_occupancies)) if attack_occupancies else 0.0,"attack_area_occupancy_available":float(bool(attack_occupancies)),"effective_damage":float(np.mean(episode_damages)) if episode_damages else 0.0,"hit_count":float(np.mean(episode_hits)) if episode_hits else 0.0,"timeout_rate":float(np.mean(episode_timeouts)) if episode_timeouts else 0.0,"ground_crash_rate":float(np.mean(episode_crashes)) if episode_crashes else 0.0}
        for name,values_ in (("raw_dense_reward",raw_dense),("assigned_dense_reward",assigned_dense),("event_reward",event_rewards),("terminal_reward",terminal_rewards)):
            diagnostics[f"{name}_mean"]=float(np.mean(values_)) if values_ else 0.0; diagnostics[f"{name}_std"]=float(np.std(values_)) if values_ else 0.0
        diagnostics.update({f"stochastic_action_{i}_frequency":float(action_counts[i]/max(action_counts.sum(),1.0)) for i in range(15)})
        return buffer,diagnostics

    def evaluate(self,episodes: int|None=None,seed_start: int=100000,deterministic: bool|None=None) -> dict[str,float]:
        count=int(episodes or self.config["validation_episodes"]); env_cfg=self.config["environment"]; outcomes=[]; returns=[]; agent_sum_returns=[]; steps=[]; red_crashes=[]; blue_crashes=[]; saturation=[]; frequencies=np.zeros(15); red_survivors=[]; blue_survivors=[]; damages=[]; hits=[]; attack_area_steps=[]; policy_entropies=[]; logit_margins=[]; terminal_proportions=[]
        deterministic = bool(self.config.get("deterministic_evaluation", True)) if deterministic is None else deterministic
        for episode in range(count):
            description=CombatEnvDescription(str(env_cfg["kind"]),str(env_cfg["scenario"]),str(env_cfg["opponent"]),env_cfg.get("multi_terminal_reward_profile")); env=make_adapter_from_description(description); current=env.reset(seed_start+episode); total=0.; agent_sum_total=0.; absolute_total=0.; terminal_absolute=0.; done=False
            while not done:
                with torch.no_grad():
                    logits=self.actor(torch.as_tensor(current.local_obs,device=self.device),torch.as_tensor(current.available_action_mask,device=self.device))
                    distribution=torch.distributions.Categorical(logits=logits); action=(torch.argmax(logits,-1) if deterministic else distribution.sample()).cpu().numpy(); active=current.agent_alive_mask.astype(bool); policy_entropies.extend(distribution.entropy().cpu().numpy()[active].tolist()); top2=torch.topk(logits,2,dim=-1).values.cpu().numpy(); logit_margins.extend((top2[...,0]-top2[...,1])[active].tolist())
                for value in action[active]: frequencies[int(value)]+=1
                current=env.step(action); total+=current.team_reward; agent_sum_total+=current.agent_reward_sum; absolute_total+=abs(current.team_reward)
                if self.num_agents==1: terminal_absolute+=abs(float(current.info["reward_breakdown"].terminal))
                else: terminal_absolute+=abs(float(np.mean([v.terminal for v in current.info["agent_reward_breakdowns"].values()])))
                done=current.terminated or current.truncated
            outcome=current.info["outcome"]; outcomes.append(outcome); returns.append(total); agent_sum_returns.append(agent_sum_total); terminal_proportions.append(terminal_absolute/max(absolute_total,1e-12)); steps.append(outcome.decision_steps)
            stats=current.info.get("statistics",{}); reason=str(outcome.termination_reason)
            if self.num_agents==1:
                red_crashes.append(float(reason=="red_ground_crash")); blue_crashes.append(float(reason=="blue_ground_crash")); red_survivors.append(float(outcome.red_alive)); blue_survivors.append(float(outcome.blue_alive)); damages.append(float(stats.get("red_effective_damage",0.0))); hits.append(float(stats.get("red_hits",0))); attack_area_steps.append(float(stats.get("red_attack_area_steps",0)))
            else:
                aircraft=stats["aircraft"]; red_crashes.append(float(sum(aircraft[f"red_{i}"]["ground_crashes"] for i in range(self.num_agents))>0)); blue_crashes.append(float(sum(aircraft[f"blue_{i}"]["ground_crashes"] for i in range(self.num_agents))>0)); red_survivors.append(float(outcome.red_survivors)); blue_survivors.append(float(outcome.blue_survivors)); damages.append(float(sum(aircraft[f"red_{i}"]["effective_damage"] for i in range(self.num_agents)))); hits.append(float(sum(aircraft[f"red_{i}"]["hits"] for i in range(self.num_agents)))); attack_area_steps.append(float(sum(aircraft[f"red_{i}"].get("attack_area_steps",0) for i in range(self.num_agents))))
            saturation.append(float(current.info.get("observation_saturation_ratio",np.mean(current.info.get("local_observation_saturation_ratio",[0.])))))
        winners=[o.winner for o in outcomes]
        combat_rates=combat_outcome_rates(outcomes); overall=combat_rates["overall_red_win_rate"]
        result={**combat_rates,"red_win_rate":overall,"blue_win_rate":winners.count("blue")/count,"red_crash_rate":float(np.mean(red_crashes)),"blue_crash_rate":float(np.mean(blue_crashes)),"mean_episode_return":float(np.mean(returns)),"mean_team_episode_return":float(np.mean(returns)),"mean_agent_sum_episode_return":float(np.mean(agent_sum_returns)),"mean_per_agent_episode_return":float(np.mean(agent_sum_returns))/self.num_agents,"mean_agent_return":float(np.mean(agent_sum_returns))/self.num_agents,"mean_episode_steps":float(np.mean(steps)),"mean_red_survivors":float(np.mean(red_survivors)),"mean_blue_survivors":float(np.mean(blue_survivors)),"mean_effective_damage":float(np.mean(damages)),"mean_hits":float(np.mean(hits)),"mean_attack_area_steps":float(np.mean(attack_area_steps)),"mean_observation_saturation_ratio":float(np.mean(saturation)),"policy_entropy_mean":float(np.mean(policy_entropies)),"logits_top1_top2_margin_mean":float(np.mean(logit_margins)),"terminal_reward_proportion":float(np.mean(terminal_proportions))}
        result.update({f"action_{i}_frequency":float(frequencies[i]/max(frequencies.sum(),1)) for i in range(15)}); return result

    def _run_impl(self) -> Path:
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
                evaluation={"environment_steps":self.environment_steps,"evaluation_split":"validation",**self.evaluate(int(self.config["validation_episodes"]),int(self.config["validation_seed_start"]))}; append_csv(self.output_dir/"evaluations.csv",evaluation)
                self.last_evaluation_step=self.environment_steps
                if self.best_evaluation is None or evaluation_key(evaluation,str(self.config["checkpoint_selection"]))>evaluation_key(self.best_evaluation,str(self.config["checkpoint_selection"])): self.best_evaluation=evaluation; self._save("best.pt")
            if self.environment_steps < total and self.environment_steps%int(self.config["checkpoint_interval"])<int(self.config["rollout_length"])*int(self.config["num_envs"]): self._save(f"step_{self.environment_steps}.pt")
        if self.last_evaluation_step != self.environment_steps:
            evaluation={"environment_steps":self.environment_steps,"evaluation_split":"validation",**self.evaluate(int(self.config["validation_episodes"]),int(self.config["validation_seed_start"]))};append_csv(self.output_dir/"evaluations.csv",evaluation);self.last_evaluation_step=self.environment_steps
            if self.best_evaluation is None or evaluation_key(evaluation,str(self.config["checkpoint_selection"]))>evaluation_key(self.best_evaluation,str(self.config["checkpoint_selection"])):self.best_evaluation=evaluation;self._save("best.pt")
        self._save("last.pt")
        test_evaluations={}
        for label in ("initial","last","best"):
            self.resume(str(self.output_dir/"checkpoints"/f"{label}.pt"),actor_only=True)
            test_evaluations[label]=self.evaluate(int(self.config["test_episodes"]),int(self.config["test_seed_start"]),deterministic=True)
        summary={"environment_steps":self.environment_steps,"updates":self.update_index,"episodes":self.episodes,"device":str(self.device),"checkpoint_selection":self.config["checkpoint_selection"],"validation_best_evaluation":self.best_evaluation,"test_seed_start":self.config["test_seed_start"],"test_episodes":self.config["test_episodes"],"test_evaluations":test_evaluations,"actor_parameters":sum(p.numel() for p in self.actor.parameters()),"critic_parameters":sum(p.numel() for p in self.critic.parameters())}; (self.output_dir/"final_summary.yaml").write_text(yaml.safe_dump(summary,sort_keys=False),encoding="utf-8"); self.writer.close(); return self.output_dir

    def run(self) -> Path:
        """Run training and always close resident vector workers."""

        try:
            return self._run_impl()
        except KeyboardInterrupt:
            latest = self._latest_step_checkpoint()
            if latest is None:
                print("Training interrupted; no step checkpoint is available. Workers are closing.")
            else:
                print(f"Training interrupted; resume from the latest step checkpoint: {latest.resolve()}")
            raise
        finally:
            self.close()

    def _latest_step_checkpoint(self) -> Path | None:
        """Return the highest existing periodic step checkpoint without saving."""

        checkpoints = self.output_dir / "checkpoints"
        candidates: list[tuple[int, Path]] = []
        for path in checkpoints.glob("step_*.pt"):
            try:
                candidates.append((int(path.stem.removeprefix("step_")), path))
            except ValueError:
                continue
        return max(candidates, default=(0, None), key=lambda item: item[0])[1]

    def close(self) -> None:
        """Close vector workers and the TensorBoard writer idempotently."""

        self.vector.close()
        self.writer.close()

    def _save(self,name: str) -> None:
        runner_state={"vector_env_state":self.vector.get_state(),"current":self.current,"episodes":self.episodes,"episode_return_accumulators":self.episode_return_accumulators,"agent_sum_return_accumulators":self.agent_sum_return_accumulators,"last_evaluation_step":self.last_evaluation_step,"trainer_minibatch_rng_state":self.trainer.minibatch_rng.bit_generator.state}
        save_checkpoint(self.output_dir/"checkpoints"/name,self.actor,self.critic,self.trainer.actor_optimizer,self.trainer.critic_optimizer,self.normalizer,self.config,self.environment_steps,self.update_index,self.best_evaluation,runner_state)
