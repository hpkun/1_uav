#!/usr/bin/env python3
"""Read-only deterministic/stochastic deployment evaluation for MARC checkpoints."""
from __future__ import annotations
import argparse,json,hashlib,sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.common.evaluator import episode_return_metrics
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modules.wave_survival_pbrs import mission_context_numpy
from algorithm.train_modular_mappo import load_config
from env.factory import make_combat_environment

def episode_policy_seed(base,environment_seed,repeat):
    payload=f"{int(base)}:{int(environment_seed)}:{int(repeat)}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8],"little")%(2**31-1)

def one_episode(trainer,env_config,seed,deterministic,policy_seed):
    env=make_combat_environment(env_config);obs,_=env.reset(seed);alive=env.red_alive_mask.copy();ah,ch=trainer.initial_hidden(1);ep=np.zeros(1,np.float32);ret=np.zeros(4);wave=1
    devices=[trainer.device.index or 0] if trainer.device.type=="cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(policy_seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(policy_seed)
        while True:
            ctx=mission_context_numpy(trainer,np.asarray([wave]),np.asarray([3]),env.blue_alive_mask[None],np.asarray([env.steps]),env.max_steps)
            actions,ah=trainer.act(obs[None],alive[None],deterministic,False,ctx,ah,ep,wave_indices=np.asarray([wave]));_,ch=trainer.values_step(obs[None],alive[None],ctx,ch,ep)
            obs,reward,terminated,truncated,info=env.step(actions[0]);ret+=reward;alive=np.asarray(info["red_alive_mask"],np.float32)
            ah=trainer.recurrent.apply_alive(ah,alive[None]);ch=trainer.recurrent.apply_alive(ch,alive[None]);ep[:]=1;wave=int(info.get("wave_index",1))
            if terminated or truncated:
                team,_=episode_return_metrics(ret)
                return {"seed":seed,"W1":float(info["waves_cleared"]>=1),"W2":float(info["waves_cleared"]>=2),"W3":float(info["waves_cleared"]>=3),"AW":float(info["waves_cleared"]),"Return":team,
                    "Boundary":float(info["red_boundary_exits"]),"Ground":float(info["red_ground_losses"]),"EpisodeLength":float(info["episode_length"])}

def aggregate(rows):
    result={key:float(np.mean([r[key] for r in rows])) for key in ("W1","W2","W3","AW","Return","Boundary","Ground","EpisodeLength")}
    result["Q2"]=None if result["W1"]==0 else result["W2"]/result["W1"];result["Q3"]=None if result["W2"]==0 else result["W3"]/result["W2"]
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--checkpoint",type=Path,required=True);parser.add_argument("--env-config",type=Path,required=True);parser.add_argument("--episodes",type=int,required=True);parser.add_argument("--seed-base",type=int,required=True)
    parser.add_argument("--policy-mode",choices=("deterministic","stochastic"),required=True);parser.add_argument("--stochastic-repeats",type=int,default=3);parser.add_argument("--policy-seed-base",type=int,default=91_000_000);parser.add_argument("--output",type=Path)
    args=parser.parse_args();checkpoint=args.checkpoint if args.checkpoint.is_absolute() else ROOT/args.checkpoint;env_path=args.env_config if args.env_config.is_absolute() else ROOT/args.env_config
    if not torch.cuda.is_available():raise RuntimeError("CUDA is mandatory; no CPU fallback")
    if args.episodes<1 or args.stochastic_repeats<1:raise ValueError("episodes/repeats must be positive")
    state=torch.load(checkpoint,map_location="cuda",weights_only=False);extra=state.get("extra",{});run=checkpoint.parent
    config=load_config(run/"algorithm_config.yaml");env_config=load_config(env_path)
    if state.get("algorithm")!="modular_mappo" or extra.get("development_method")!="marc_mappo_v1" or state.get("development_feature_versions",{}).get("milestone_aware_retention_credit")!=1:raise RuntimeError("checkpoint is not MARC-MAPPO V1")
    trainer=build_modular_mappo_trainer(config,"cuda",total_sampled_steps=config["training"]["total_sampled_steps"]);trainer.load(checkpoint,strict_protocol=True,restore_rng=False)
    repeats=1 if args.policy_mode=="deterministic" else args.stochastic_repeats;results=[]
    for repeat in range(repeats):
        rows=[]
        for seed in range(args.seed_base,args.seed_base+args.episodes):
            policy_seed=episode_policy_seed(args.policy_seed_base,seed,repeat)
            rows.append(one_episode(trainer,env_config,seed,args.policy_mode=="deterministic",policy_seed))
        results.append({"repeat":repeat,"metrics":aggregate(rows)})
    metrics={key:{"mean":float(np.mean([r["metrics"][key] for r in results])),"std":float(np.std([r["metrics"][key] for r in results]))}
             for key in results[0]["metrics"] if all(r["metrics"][key] is not None for r in results)}
    report={"checkpoint":str(checkpoint),"checkpoint_sampled_steps":int(state["sampled_steps"]),"training_seed":int(extra["training_seed"]),"environment_config":str(env_path),"policy_mode":args.policy_mode,
      "environment_seed_range":[args.seed_base,args.seed_base+args.episodes-1],"episodes_per_repeat":args.episodes,"stochastic_repeats":repeats,"per_repeat":results,"aggregate":metrics,"training_performed":False}
    if args.output:
        output=args.output if args.output.is_absolute() else ROOT/args.output
        if output.exists():raise FileExistsError(output)
        output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
