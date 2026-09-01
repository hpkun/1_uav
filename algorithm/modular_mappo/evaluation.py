"""Raw-environment evaluation with one canonical recurrent-state lifecycle."""
import numpy as np
from env.factory import make_combat_environment
from algorithm.common.evaluator import episode_return_metrics,persistent_mission_metrics

def per_wave_episode_diagnostics(info,total_waves=3):
 """Flatten only terminal fields recorded by the environment (no inference)."""
 records={int(row["wave_index"]):row for row in info.get("per_wave_metrics",[])};out={}
 mappings={"red_survivors_after_wave":"red_survivors_end","blue_survivors_after_wave":"blue_survivors_end","red_attack_kills_after_wave":"red_attack_kills","red_boundary_losses_after_wave":"red_boundary_exits","red_ground_losses_after_wave":"red_ground_losses"}
 for wave in range(1,total_waves+1):
  row=records.get(wave)
  for name,source in mappings.items():out[f"{name}_{wave}"]=None if row is None else row.get(source)
  out[f"wave_{wave}_entry_step"]=None if row is None else row.get("start_step")
  out[f"wave_{wave}_duration"]=None if row is None else row.get("duration_steps")
 return out

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
   team,agent=episode_return_metrics(ret);record={"episode_return":team,"mean_agent_episode_return":agent,**info,**per_wave_episode_diagnostics(info,total)}
   if include_trace:record.update({"action_trace":np.asarray(actions_trace),"wave_trace":np.asarray(wave_trace)})
   return record

def evaluate_modular(trainer,env_config,seeds):
 records=[evaluate_modular_episode(trainer,env_config,seed) for seed in seeds]
 mean=lambda k:float(np.mean([r[k] for r in records]))
 result={"average_return":mean("episode_return"),"average_agent_return":mean("mean_agent_episode_return"),"win_rate":mean("red_success"),"loss_rate":mean("blue_win"),"draw_rate":mean("draw"),"timeout_rate":float(np.mean([r["termination_reason"]=="red_failure_timeout" for r in records])),"average_red_loss":mean("red_losses"),"average_blue_loss":mean("blue_losses"),"average_red_attack_kills":mean("red_attack_kills"),"average_blue_attack_kills":mean("blue_attack_kills"),"average_red_boundary_exits":mean("red_boundary_exits"),"average_blue_boundary_exits":mean("blue_boundary_exits"),"evaluation_boundary_exit_rate":float(np.mean([r["red_boundary_exits"]>0 for r in records])),"average_red_ground_losses":mean("red_ground_losses"),"average_blue_ground_losses":mean("blue_ground_losses"),"average_episode_length":mean("episode_length"),"evaluation_episodes":len(records),**{f"{side}_{event}_episode_rate":float(np.mean([r[f"{side}_first_{event}_step"] is not None for r in records])) for side in ("red","blue") for event in ("fire_window","attempt","hit","kill")},**{f"average_episode_{name}_total":mean(f"episode_{name}_total") for name in ("r1","r2","r3","r4")},**persistent_mission_metrics(records)}
 for wave in range(1,int(env_config.get("persistent_waves",{}).get("total_waves",1))+1):
  names=("red_survivors_after_wave","blue_survivors_after_wave","red_attack_kills_after_wave","red_boundary_losses_after_wave","red_ground_losses_after_wave")
  for name in names:
   values=[r[f"{name}_{wave}"] for r in records if r[f"{name}_{wave}"] is not None];value=float(np.mean(values)) if values else None
   result[f"average_{name}_{wave}_conditional_on_record"]=value;result[f"{name}_{wave}"]=value
  for name in (f"wave_{wave}_entry_step",f"wave_{wave}_duration"):
   values=[r[name] for r in records if r[name] is not None];result[name]=float(np.mean(values)) if values else None
 for wave in range(1,int(env_config.get("persistent_waves",{}).get("total_waves",1))+1):
  key=f"average_red_survivors_after_wave_{wave}";result[key+"_conditional_on_clear"]=result[key]
 return result
__all__=["evaluate_modular","evaluate_modular_episode","per_wave_episode_diagnostics"]
