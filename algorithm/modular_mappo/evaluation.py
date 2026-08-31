"""Raw-environment evaluation with one canonical recurrent-state lifecycle."""
import numpy as np
from env.factory import make_combat_environment
from algorithm.common.evaluator import episode_return_metrics,persistent_mission_metrics


def evaluate_modular_episode(trainer, env_config, seed, include_trace=False):
 env=make_combat_environment(env_config);obs,_=env.reset(int(seed));alive=env.red_alive_mask.copy()
 ah,ch=trainer.initial_hidden(1);wave=1;total=int(env_config.get("persistent_waves",{}).get("total_waves",1));ret=np.zeros(4);ep=np.zeros(1,np.float32)
 actions_trace=[];wave_trace=[]
 while True:
  ctx=trainer.context_numpy(np.asarray([wave]),np.asarray([total]))
  actions,ah=trainer.act(obs[None],alive[None],True,False,ctx,ah,ep)
  _,ch=trainer.values_step(obs[None],alive[None],ctx,ch,ep)
  if include_trace:actions_trace.append(actions[0].copy());wave_trace.append(wave)
  obs,reward,terminated,truncated,info=env.step(actions[0]);ret+=reward
  alive=np.asarray(info["red_alive_mask"],np.float32)
  ah=trainer.recurrent.apply_alive(ah,alive[None]);ch=trainer.recurrent.apply_alive(ch,alive[None])
  ep[:]=1;wave=int(info.get("wave_index",1));total=int(info.get("total_waves",total))
  if terminated or truncated:
   team,agent=episode_return_metrics(ret);record={"episode_return":team,"mean_agent_episode_return":agent,**info}
   if include_trace:record.update({"action_trace":np.asarray(actions_trace),"wave_trace":np.asarray(wave_trace)})
   return record

def evaluate_modular(trainer,env_config,seeds):
 records=[evaluate_modular_episode(trainer,env_config,seed) for seed in seeds]
 mean=lambda k:float(np.mean([r[k] for r in records]))
 result={"average_return":mean("episode_return"),"average_agent_return":mean("mean_agent_episode_return"),"win_rate":mean("red_success"),"loss_rate":mean("blue_win"),"draw_rate":mean("draw"),"timeout_rate":float(np.mean([r["termination_reason"]=="red_failure_timeout" for r in records])),"average_red_loss":mean("red_losses"),"average_blue_loss":mean("blue_losses"),"average_red_attack_kills":mean("red_attack_kills"),"average_blue_attack_kills":mean("blue_attack_kills"),"average_red_boundary_exits":mean("red_boundary_exits"),"average_blue_boundary_exits":mean("blue_boundary_exits"),"evaluation_boundary_exit_rate":float(np.mean([r["red_boundary_exits"]>0 for r in records])),"average_red_ground_losses":mean("red_ground_losses"),"average_blue_ground_losses":mean("blue_ground_losses"),"average_episode_length":mean("episode_length"),"evaluation_episodes":len(records),**{f"{side}_{event}_episode_rate":float(np.mean([r[f"{side}_first_{event}_step"] is not None for r in records])) for side in ("red","blue") for event in ("fire_window","attempt","hit","kill")},**{f"average_episode_{name}_total":mean(f"episode_{name}_total") for name in ("r1","r2","r3","r4")},**persistent_mission_metrics(records)}
 for key in list(result):
  if key.startswith("average_red_survivors_after_wave_"):
   result[key+"_conditional_on_clear"]=result[key]
 return result
__all__=["evaluate_modular","evaluate_modular_episode"]
