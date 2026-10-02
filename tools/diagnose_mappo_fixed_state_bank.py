#!/usr/bin/env python3
"""Optional read-only 88M fixed-bank and deployment diagnostic for MAPPO."""
from __future__ import annotations
import argparse,csv,json,math,sys
from pathlib import Path
import numpy as np,torch,yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.mappo.factory import build_mappo_trainer
from env.factory import make_combat_environment

DEFAULTS={
 "AttentionFreshBest":ROOT/"outputs/mappo_attention_seed5303_1p5m/best_eval.pt",
 "AttentionFreshFinal":ROOT/"outputs/mappo_attention_seed5303_1p5m/checkpoint_1500000.pt",
 "ControlBest":ROOT/"outputs/mappo_attn_lr3e4_cont_seed5303_1p5m/best_eval.pt",
 "ControlFinal":ROOT/"outputs/mappo_attn_lr3e4_cont_seed5303_1p5m/final.pt",
 "TreatmentBest":ROOT/"outputs/mappo_attn_lr1e4_cont_seed5303_1p5m/best_eval.pt",
 "TreatmentFinal":ROOT/"outputs/mappo_attn_lr1e4_cont_seed5303_1p5m/final.pt",
}

def config_for(checkpoint:Path):return yaml.safe_load((checkpoint.parent/"algorithm_config.yaml").read_text(encoding="utf-8"))
def load_policy(checkpoint:Path,device:str):
 trainer=build_mappo_trainer(config_for(checkpoint),device);trainer.load(checkpoint);return trainer
def stats(values):
 values=np.asarray(values,float);return {name:float(value) for name,value in zip(("p1","p10","median","p90","p99","min","max","mean"),(*np.quantile(values,[.01,.1,.5,.9,.99]),values.min(),values.max(),values.mean()))}
def entry_geometry(env):
 red=[state for state in env.red if state.alive];blue=[state for state in env.blue if state.alive]
 positions=np.asarray([[state.x,state.y,state.altitude] for state in red],float);blue_positions=np.asarray([[state.x,state.y,state.altitude] for state in blue],float)
 dispersion=0.0 if len(red)<2 else float(np.mean([np.linalg.norm(positions[i]-positions[j]) for i in range(len(red)) for j in range(i+1,len(red))]))
 nearest=float(min(np.linalg.norm(a-b) for a in positions for b in blue_positions))
 radial_velocity=[]
 for state in red:
  radius=max(math.hypot(state.x,state.y),1e-9);velocity=state.velocity_vector();radial_velocity.append((state.x*velocity[0]+state.y*velocity[1])/radius)
 return {"wave":env.wave_index,"step":env.steps,"red_survivors":len(red),"min_boundary_margin":float(min(env.arena_radius-math.hypot(s.x,s.y) for s in red)),"altitude_mean":float(np.mean([s.altitude for s in red])),"altitude_min":float(min(s.altitude for s in red)),"speed_mean":float(np.mean([s.v for s in red])),"heading_circular_concentration":float(abs(np.mean(np.exp(1j*np.asarray([s.psi for s in red]))))),"pitch_std":float(np.std([s.theta for s in red])),"formation_dispersion":dispersion,"nearest_blue_distance":nearest,"radial_velocity_mean":float(np.mean(radial_velocity)),"red_fire_ready_fraction":float(np.mean([state.armed for state in env.red_fire_states])),"remaining_horizon":env.max_steps-env.steps}
def run_episode(trainer,env_config,seed,deterministic,collect=False,stride=10):
 env=make_combat_environment(env_config);obs,_=env.reset(seed);records=[];entries=[];returns=np.zeros(4,float)
 policy_seed=int(seed+(0 if deterministic else 777));torch.manual_seed(policy_seed);torch.cuda.manual_seed_all(policy_seed)
 while True:
  if collect and (env.steps%stride==0):records.append({"observation":obs.copy(),"alive":env.red_alive_mask.copy(),"wave":env.wave_index,"remaining_horizon":env.max_steps-env.steps})
  action=trainer.act(obs,env.red_alive_mask,deterministic=deterministic);obs,reward,terminated,truncated,info=env.step(action);returns+=reward
  if info.get("spawned_next_wave"):
   entries.append(entry_geometry(env));records.append({"observation":obs.copy(),"alive":env.red_alive_mask.copy(),"wave":env.wave_index,"remaining_horizon":env.max_steps-env.steps})
  if terminated or truncated:return {"seed":seed,"deterministic":deterministic,"return":float(returns.sum()),"W":int(info["waves_cleared"]),"red_loss":int(info["red_losses"]),"boundary":int(info["red_boundary_exits"]),"ground":int(info["red_ground_losses"]),"episode_length":int(info["episode_length"])},records,entries
def write_csv(path,rows):
 fields=[]
 for row in rows:
  for key in row:
   if key not in fields:fields.append(key)
 with path.open("w",newline="",encoding="utf-8") as stream:writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
def main():
 parser=argparse.ArgumentParser();parser.add_argument("--device",default="cuda",choices=("cuda",));parser.add_argument("--episodes",type=int,default=12);parser.add_argument("--bank-episodes",type=int,default=12);parser.add_argument("--stride",type=int,default=10);parser.add_argument("--seed-base",type=int,default=88_700_000);parser.add_argument("--output-dir",type=Path,default=ROOT/"outputs/mappo_fixed_state_bank_diagnostic");args=parser.parse_args()
 if not torch.cuda.is_available():raise RuntimeError("CUDA mandatory")
 if args.seed_base<88_000_000 or args.seed_base+max(args.episodes,args.bank_episodes)>=89_000_000:raise RuntimeError("diagnostic seeds must remain inside independent 88M range")
 output=args.output_dir if args.output_dir.is_absolute() else ROOT/args.output_dir
 if output.exists():raise FileExistsError(output)
 for path in DEFAULTS.values():
  if not path.is_file():raise FileNotFoundError(path)
 env_config=yaml.safe_load((ROOT/"configs/persistent_wave_v2_environment.yaml").read_text(encoding="utf-8"));policies={name:load_policy(path,args.device) for name,path in DEFAULTS.items()}
 collector=policies["ControlBest"];bank=[];entries=[]
 for seed in range(args.seed_base,args.seed_base+args.bank_episodes):
  _,states,new_entries=run_episode(collector,env_config,seed,True,True,args.stride);bank.extend(states);entries.extend({"bank_seed":seed,**row} for row in new_entries)
 if not bank:raise RuntimeError("state bank empty")
 std_rows=[];state_outputs={}
 for name,trainer in policies.items():
  state_outputs[name]=[]
  for wave in (1,2,3):
   raw=[]
   for item in bank:
    if item["wave"]!=wave:continue
    tensor=torch.as_tensor(item["observation"],dtype=torch.float32,device=trainer.device);alive=torch.as_tensor(item["alive"],dtype=torch.bool,device=trainer.device)
    with torch.no_grad():distribution=trainer.actor.distribution(tensor);action=torch.tanh(distribution.mean);value=trainer.critic(tensor.unsqueeze(0),alive.float().unsqueeze(0)).squeeze(0)
    for agent in torch.flatnonzero(alive).tolist():
     raw.append(distribution.scale[agent].log().cpu().numpy());state_outputs[name].append({"wave":wave,"remaining":item["remaining_horizon"],"observation":item["observation"][agent],"action":action[agent].cpu().numpy(),"value":float(value[agent])})
   if raw:
    array=np.asarray(raw)
    for index,axis in enumerate(("psi","theta","v")):std_rows.append({"method":name,"wave":wave,"axis":axis,"states":len(array),**{f"log_std_{key}":value for key,value in stats(array[:,index]).items()}})
 deployment=[]
 for name,trainer in policies.items():
  for seed in range(args.seed_base+1000,args.seed_base+1000+args.episodes):
   for deterministic in (True,False):deployment.append({"method":name,**run_episode(trainer,env_config,seed,deterministic)[0]})
 alias=[];reference=state_outputs["ControlBest"]
 for wave_a,wave_b in ((1,2),(1,3),(2,3)):
  a=[row for row in reference if row["wave"]==wave_a];b=[row for row in reference if row["wave"]==wave_b]
  if not a or not b:continue
  b_obs=np.asarray([row["observation"] for row in b]);
  for row in a:
   distance=np.linalg.norm(b_obs-row["observation"],axis=1);index=int(np.argmin(distance));other=b[index]
   alias.append({"wave_a":wave_a,"wave_b":wave_b,"observation_l2":float(distance[index]),"remaining_horizon_delta":other["remaining"]-row["remaining"],"control_action_l2":float(np.linalg.norm(other["action"]-row["action"])),"control_value_delta":other["value"]-row["value"]})
 output.mkdir(parents=True);write_csv(output/"policy_log_std_by_wave.csv",std_rows);write_csv(output/"entry_state_geometry.csv",entries);write_csv(output/"deployment_modes.csv",deployment);write_csv(output/"cross_wave_alias_neighbours.csv",alias)
 report={"status":"DIAGNOSTIC_COMPLETE","diagnostic_seed_range":[args.seed_base,args.seed_base+max(args.episodes,args.bank_episodes)+1000-1],"bank_policy":"ControlBest","bank_states":len(bank),"entry_states":len(entries),"checkpoints":{name:str(path.relative_to(ROOT)) for name,path in DEFAULTS.items()},"note":"Legal states only; training and formal evaluation ranges untouched."};(output/"diagnostic_summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8");print(json.dumps(report,indent=2))
if __name__=="__main__":main()
