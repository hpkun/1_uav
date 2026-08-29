"""Raw-environment evaluation with context and recurrent-state lifecycle."""
import numpy as np
from env.factory import make_combat_environment
from algorithm.common.evaluator import episode_return_metrics,persistent_mission_metrics

def evaluate_modular(trainer,env_config,seeds):
 records=[]
 for seed in seeds:
  env=make_combat_environment(env_config);obs,info=env.reset(int(seed));alive=env.red_alive_mask.copy();ah,ch=trainer.initial_hidden(1);wave=1;total=int(env_config.get("persistent_waves",{}).get("total_waves",1));ret=np.zeros(4);ep=np.zeros(1,np.float32)
  while True:
   ctx=trainer.context_numpy(np.asarray([wave]),np.asarray([total]));actions,ah=trainer.act(obs[None],alive[None],True,False,ctx,ah,ep);_,ch=trainer.values_step(obs[None],alive[None],ctx,ch,ep)
   obs,reward,terminated,truncated,info=env.step(actions[0]);ret+=reward;alive=np.asarray(info["red_alive_mask"],np.float32);ep[:]=1
   wave=int(info.get("wave_index",1));total=int(info.get("total_waves",total))
   if terminated or truncated:
    team,agent=episode_return_metrics(ret);records.append({"episode_return":team,"mean_agent_episode_return":agent,**info});break
 mean=lambda k:float(np.mean([r[k] for r in records]))
 result={"average_return":mean("episode_return"),"average_agent_return":mean("mean_agent_episode_return"),"win_rate":mean("red_success"),"loss_rate":mean("blue_win"),"draw_rate":mean("draw"),"timeout_rate":float(np.mean([r["termination_reason"]=="red_failure_timeout" for r in records])),"average_red_loss":mean("red_losses"),"average_blue_loss":mean("blue_losses"),"average_red_boundary_exits":mean("red_boundary_exits"),"evaluation_boundary_exit_rate":float(np.mean([r["red_boundary_exits"]>0 for r in records])),"average_red_ground_losses":mean("red_ground_losses"),"average_blue_ground_losses":mean("blue_ground_losses"),"average_episode_length":mean("episode_length"),"evaluation_episodes":len(records),**persistent_mission_metrics(records)}
 return result
__all__=["evaluate_modular"]
