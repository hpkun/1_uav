"""Focused preflight for the Actor-only GRU History MAPPO development protocol."""
from __future__ import annotations

import argparse,inspect,json,sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from algorithm.modular_mappo.buffer import ModularRolloutBatch,recurrent_batch_plan
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.protocol import checkpoint_architecture
from algorithm.modular_mappo.runner import ModularMAPPOTrainingRunner
from algorithm.train_modular_mappo import load_config

ENV=ROOT/"configs/persistent_wave_v2_environment.yaml"
CONFIG=ROOT/"configs/dev_actor_only_gru_history_3m.yaml"
PLAIN=ROOT/"configs/diag_mappo_learnability_common_3m.yaml"
MANIFEST=ROOT/"experiments/actor_only_gru_history_manifest.json"
REGISTRY=ROOT/"experiments/current_seed_provenance.json"
FORMAL_ROOT=ROOT/"outputs/dev_actor_only_gru_history_3m"
AUDIT=ROOT/"outputs/actor_only_gru_history_preflight"
SMOKE=ROOT/"outputs/dev_actor_only_gru_history_cuda_smoke/seed88500001"
SEEDS=(5301,5302,5303)

def _yaml(path:Path)->dict[str,Any]:return yaml.safe_load(path.read_text(encoding="utf-8"))
def _json(path:Path)->dict[str,Any]:return json.loads(path.read_text(encoding="utf-8"))

def synthetic_rollout(trainer,time_steps:int=8,num_envs:int=2)->ModularRolloutBatch:
    rng=np.random.default_rng(885);n=4
    obs=rng.normal(size=(time_steps,num_envs,n,52)).astype(np.float32)
    alive=np.ones((time_steps,num_envs,n),np.float32);alive[time_steps//2:,0,3]=0
    episode=np.ones((time_steps,num_envs),np.float32);episode[0]=0
    if time_steps>4:episode[4,1]=0
    contexts=np.zeros((time_steps,num_envs,n,0),np.float32)
    raw=np.zeros((time_steps,num_envs,n,3),np.float32);actions=np.zeros_like(raw);old=np.zeros((time_steps,num_envs,n),np.float32)
    actor_hidden,_=trainer.initial_hidden(num_envs);saved=[]
    for t in range(time_steps):
        saved.append(actor_hidden.copy())
        with torch.no_grad():
            o=torch.as_tensor(obs[t],device=trainer.device);a=torch.as_tensor(alive[t],device=trainer.device);h=torch.as_tensor(actor_hidden,device=trainer.device);ep=torch.as_tensor(episode[t],device=trainer.device)
            dist,new_hidden=trainer.actor.distribution_step(o,None,h,ep,a);sample=dist.mean;action=torch.tanh(sample)
            raw[t]=sample.cpu().numpy();actions[t]=action.cpu().numpy();old[t]=trainer.actor._squashed_log_prob(dist,sample,action).cpu().numpy();actor_hidden=new_hidden.cpu().numpy()
    rewards=rng.normal(size=(time_steps,num_envs,n)).astype(np.float32);dones=np.zeros((time_steps,num_envs),np.float32)
    return ModularRolloutBatch(obs,actions,raw,old,rewards,rewards.copy(),dones,alive,obs.copy(),alive.copy(),np.ones((time_steps,num_envs),int),np.full((time_steps,num_envs),3,int),contexts,contexts.copy(),np.asarray(saved),None,episode)

def source_lifecycle_audit()->dict[str,bool]:
    import algorithm.modular_mappo.runner as runner_module
    import algorithm.modular_mappo.trainer as trainer_module
    import algorithm.modular_mappo.evaluation as evaluation_module
    collect=inspect.getsource(runner_module.ModularMAPPOTrainingRunner.collect_rollout)
    minibatch=inspect.getsource(trainer_module.ModularMAPPOTrainer._recurrent_minibatch)
    evaluation=inspect.getsource(evaluation_module.evaluate_modular_episode)
    checks={
      "pre_action_hidden_saved":"actor_before = None if self.actor_hidden is None else self.actor_hidden.copy()" in collect and "actor_before" in collect,
      "chunk_restores_stored_hidden_detached":"saved[s,e]" in minibatch and ".detach()" in minibatch,
      "episode_mask_used_in_recompute":"EP[t]" in minibatch and "distribution_step" in minibatch,
      "wave_transition_does_not_reset_hidden":"wave_cleared_this_step" not in collect[collect.index("self.actor_hidden ="):],
      "alive_mask_blocks_dead_hidden":"apply_alive(new_actor, self.alive)" in collect,
      "new_log_prob_uses_recurrent_history":"newlog=self.actor._squashed_log_prob(dist,R[t],A[t])" in minibatch,
      "evaluation_carries_and_alive_masks_hidden":"actions,ah=trainer.act" in evaluation and "apply_alive(ah" in evaluation,
      "deterministic_evaluation":"trainer.act(obs[None],alive[None],True" in evaluation,
    }
    if not all(checks.values()):raise RuntimeError(f"recurrent lifecycle source audit failed: {checks}")
    return checks

def validate(check_outputs:bool=True)->dict[str,Any]:
    env=_yaml(ENV);cfg=load_config(CONFIG);plain=load_config(PLAIN);manifest=_json(MANIFEST);registry=_json(REGISTRY)
    if cfg.get("development_method")!="actor_only_gru_history":raise RuntimeError("development_method mismatch")
    if (env["environment_variant"],env["persistent_waves"]["total_waves"],env["simulation"]["max_steps"],env["scenario"]["team_size"])!=("persistent_wave_v2",3,3000,4):raise RuntimeError("environment contract mismatch")
    enabled=sorted(k for k,v in cfg["modules"].items() if isinstance(v,dict) and v.get("enabled",False))
    if enabled!=["actor_lr_decay","recurrent_memory"]:raise RuntimeError(f"enabled modules mismatch: {enabled}")
    recurrent=cfg["modules"]["recurrent_memory"]
    if recurrent!={"enabled":True,"mode":"actor_gru","hidden_dim":128,"sequence_length":32}:raise RuntimeError("recurrent protocol mismatch")
    training_keys=("actor_learning_rate","critic_learning_rate","gamma","gae_lambda","clip_ratio","value_loss_coefficient","entropy_coefficient","max_grad_norm","rollout_steps","ppo_epochs","minibatch_size","num_train_envs","total_sampled_steps","evaluation_episodes","evaluation_interval_sampled_steps")
    if any(cfg["training"][k]!=plain["training"][k] for k in training_keys):raise RuntimeError("training protocol differs from Plain MAPPO")
    trainer=build_modular_mappo_trainer(cfg,"cpu",256,3_000_000);arch=checkpoint_architecture(trainer)
    expected={"actor_input_dim":52,"actor_context_dim":0,"actor_gru_hidden_dim":128,"critic_context_dim":0,"critic_gru_hidden_dim":0,"critic_input_dim":52}
    if any(arch.get(k)!=v for k,v in expected.items()):raise RuntimeError(f"architecture mismatch: {arch}")
    if list(cfg["network"]["actor_hidden_layers"])!=[256,256] or not isinstance(trainer.actor.gru,torch.nn.GRUCell) or hasattr(trainer.critic,"gru"):raise RuntimeError("GRU placement or critic topology mismatch")
    actor_hidden,critic_hidden=trainer.initial_hidden(3)
    if actor_hidden.shape!=(3,4,128) or np.any(actor_hidden) or critic_hidden is not None:raise RuntimeError("initial hidden state mismatch")
    probe=np.ones((2,4,128),np.float32);alive=np.asarray([[1,0,1,0],[1,1,1,1]],np.float32)
    across_wave=trainer.recurrent.apply_alive(probe.copy(),alive)
    if not np.all(across_wave[0,0]==1) or np.any(across_wave[0,1]) or not np.all(across_wave[1]==1):raise RuntimeError("alive hidden lifecycle mismatch")
    trainer.recurrent.reset_for_episode(across_wave,np.asarray([False,True]))
    if not np.any(across_wave[0]) or np.any(across_wave[1]):raise RuntimeError("episode reset lifecycle mismatch")
    plan=recurrent_batch_plan(256,24,32,512,10)
    expected_plan={"sequence_chunks":192,"sequences_per_minibatch":16,"recurrent_minibatches_per_epoch":12,"optimizer_steps":120}
    if plan!=expected_plan:raise RuntimeError(f"sequence arithmetic mismatch: {plan}")
    lr={str(s):trainer.actor_lr_decay.learning_rate(s,.0003) for s in (0,600000,750000,900000,3000000)}
    expected_lr={"0":.0003,"600000":.0003,"750000":.0002,"900000":.0001,"3000000":.0001}
    if any(abs(lr[k]-v)>1e-15 for k,v in expected_lr.items()):raise RuntimeError(f"LR mismatch: {lr}")
    small=deepcopy(cfg);small["training"]["ppo_epochs"]=1;small["training"]["minibatch_size"]=64
    update_trainer=build_modular_mappo_trainer(small,"cpu",32,1000);before={k:v.detach().clone() for k,v in update_trainer.actor.gru.state_dict().items()}
    metrics=update_trainer.update(synthetic_rollout(update_trainer))
    changed=any(not torch.equal(v,update_trainer.actor.gru.state_dict()[k]) for k,v in before.items())
    if not all(np.isfinite(float(v)) for v in metrics.values()) or metrics["actor_gru_grad_norm"]<=0 or metrics["critic_gru_grad_norm"]!=0 or not changed:raise RuntimeError("synthetic recurrent update failed")
    checkpoint=AUDIT/"synthetic_recurrent_roundtrip.pt";AUDIT.mkdir(parents=True,exist_ok=True);update_trainer.save(checkpoint,{"development_method":"actor_only_gru_history"})
    restored=build_modular_mappo_trainer(small,"cpu",32,1000);restored.load(checkpoint);state=torch.load(checkpoint,map_location="cpu",weights_only=False)
    if not all(torch.equal(v,restored.actor.gru.state_dict()[k]) for k,v in update_trainer.actor.gru.state_dict().items()) or state["module_config"]["recurrent_memory"]!=small["modules"]["recurrent_memory"]:raise RuntimeError("checkpoint recurrent provenance mismatch")
    future=registry["evaluation_ranges"]["45000000..45000199"]
    if future.get("executed") is not False or manifest["future_final"].get("executed") is not False:raise RuntimeError("45M future-final block is not untouched")
    existing=[str(FORMAL_ROOT/f"seed{s}") for s in SEEDS if (FORMAL_ROOT/f"seed{s}").exists()]
    if check_outputs and existing:raise RuntimeError(f"formal output directories are not fresh: {existing}")
    return {"status":"READY_FOR_ACTOR_ONLY_GRU_HISTORY_3M","enabled_modules":enabled,"architecture":arch,"sequence_plan":plan,"actor_lr_by_step":lr,
      "hidden_lifecycle":{"initial_actor_zero":True,"initial_critic_none":True,"survivor_across_wave_retained":True,"dead_agent_zeroed":True,"episode_done_zeroed":True},
      "source_lifecycle_audit":source_lifecycle_audit(),"synthetic_update":{"finite":True,"actor_gru_grad_norm":metrics["actor_gru_grad_norm"],"critic_gru_grad_norm":metrics["critic_gru_grad_norm"],"actor_gru_parameters_updated":True},
      "checkpoint_roundtrip":True,"training_seeds":list(SEEDS),"formal_outputs_fresh":not existing,"future_final":{"range":[45000000,45000199],"executed":False}}

def cuda_smoke()->dict[str,Any]:
    if not torch.cuda.is_available():raise RuntimeError("CUDA required for smoke; CPU fallback forbidden")
    if SMOKE.exists():raise RuntimeError(f"CUDA smoke output already exists: {SMOKE}")
    cfg=load_config(CONFIG);cfg["training"]["seed"]=88_500_001
    runner=ModularMAPPOTrainingRunner(_yaml(ENV),cfg,1,32,"cuda",88_500_001,SMOKE,False)
    try:
        rollout=runner.collect_rollout(4);before={k:v.detach().clone() for k,v in runner.trainer.actor.gru.state_dict().items()};metrics=runner.trainer.update(rollout)
        carried=bool(rollout.actor_hidden_before_step is not None and np.linalg.norm(rollout.actor_hidden_before_step[1:])>0)
        changed=any(not torch.equal(v,runner.trainer.actor.gru.state_dict()[k]) for k,v in before.items());runner.save_checkpoint(SMOKE/"checkpoint_smoke.pt")
        restored=build_modular_mappo_trainer(cfg,"cuda",256,32);restored.load(SMOKE/"checkpoint_smoke.pt")
        finite=all(torch.isfinite(v).all().item() for group in (restored.actor.state_dict(),restored.critic.state_dict()) for v in group.values())
    finally:runner.vector.close()
    if not carried or not changed or metrics["actor_gru_grad_norm"]<=0 or not finite:raise RuntimeError("CUDA recurrent smoke failed")
    result={"seed":88_500_001,"device":"cuda","environment_started":True,"rollout_steps":4,"actor_hidden_carried":True,"recurrent_update_finite":True,"actor_gru_grad_norm":metrics["actor_gru_grad_norm"],"checkpoint_roundtrip":True,"evaluation_episodes":0,"future_final_used":False,"checkpoint":str(SMOKE/"checkpoint_smoke.pt")}
    (SMOKE/"smoke.json").write_text(json.dumps(result,indent=2),encoding="utf-8");return result

def write_result(result:dict[str,Any])->None:
    AUDIT.mkdir(parents=True,exist_ok=True);(AUDIT/"preflight.json").write_text(json.dumps(result,indent=2),encoding="utf-8");(AUDIT/"preflight.md").write_text("# Actor-only GRU History preflight\n\n```json\n"+json.dumps(result,indent=2)+"\n```\n",encoding="utf-8")

def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--cuda-smoke",action="store_true");parser.add_argument("--summary",action="store_true");args=parser.parse_args()
    result=validate();result["cuda_smoke"]=cuda_smoke() if args.cuda_smoke else "NOT_REQUESTED";write_result(result)
    print("[PREFLIGHT] READY | method=actor_only_gru_history | actor_gru=128 | critic_gru=0 | seq=32 | chunks=192 | minibatches=12 | seeds=5301,5302,5303 | future_final=UNUSED" if args.summary else json.dumps(result,indent=2))

if __name__=="__main__":main()
