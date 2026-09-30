#!/usr/bin/env python3
"""Strict preflight for the matched Fixed10 Control versus DAWE V1 screen."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.common.protocol import runtime_source_manifest
from algorithm.train_modular_mappo import load_config
from algorithm.modular_mappo.protocol import validate_dawe_branch,validate_dawe_config_pair

SEEDS=(5301,5302,5303);SOURCE_STEP=1_505_280;TARGET=1_805_280
SOURCE=lambda seed:ROOT/f"outputs/diag_mappo_learnability/l3_seed{seed}/checkpoint_1505280.pt"
OUTPUTS=lambda seed:(ROOT/f"outputs/dev_dawe_control_seed{seed}_300k",ROOT/f"outputs/dev_dawe_v1_seed{seed}_300k")
CONTROL=ROOT/"configs/dev_dawe_fixed10_control_300k.yaml";TREATMENT=ROOT/"configs/dev_dawe_fixed10_v1_300k.yaml"
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as stream:
  for block in iter(lambda:stream.read(1024*1024),b""):h.update(block)
 return h.hexdigest()
def main():
 if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
 env=load_config(ROOT/"configs/persistent_wave_v2_environment.yaml")
 control=load_config(CONTROL);treatment=load_config(TREATMENT);validate_dawe_config_pair(control,treatment)
 expected={"total_sampled_steps":TARGET,"num_train_envs":24,"rollout_steps":256,"gamma":.999,
  "gae_lambda":.95,"clip_ratio":.2,"entropy_coefficient":.01,"value_loss_coefficient":.5,
  "max_grad_norm":.5,"ppo_epochs":10,"minibatch_size":512}
 for name,config,enabled in (("Control",control,False),("DAWE",treatment,True)):
  if any(config["training"].get(key)!=value for key,value in expected.items()):raise RuntimeError(f"{name} training protocol mismatch")
  modules=sorted(key for key,value in config["modules"].items() if isinstance(value,dict) and value.get("enabled",False))
  wanted=["actor_gradient_clipping","actor_lr_decay"]+(["deployment_aligned_wave_exploration"] if enabled else [])
  if modules!=wanted:raise RuntimeError(f"{name} enabled modules mismatch: {modules}")
  dawe=config["modules"]["deployment_aligned_wave_exploration"]
  if dawe!={"enabled":enabled,"mode":"fixed_wave_std_multiplier","wave1_multiplier":.25,"wave2_multiplier":.25,"wave3_multiplier":1.0}:raise RuntimeError(f"{name} DAWE config mismatch")
  clip=config["modules"]["actor_gradient_clipping"]
  if clip!={"enabled":True,"mode":"actor_only_fixed_norm","actor_max_grad_norm":.5,"critic_max_grad_norm_unchanged":True,"critic_max_grad_norm":.5}:raise RuntimeError(f"{name} actor clip mismatch")
 if (env.get("environment_variant"),env["persistent_waves"]["total_waves"],env["simulation"]["max_steps"],env["scenario"]["team_size"])!=("persistent_wave_v2",3,3000,4):raise RuntimeError("environment mismatch")
 if (control["network"]["observation_dim"],control["network"]["action_dim"],control["network"]["num_agents"])!=(52,3,4):raise RuntimeError("network mismatch")
 sources=[]
 for seed in SEEDS:
  path=SOURCE(seed)
  if not path.is_file():raise FileNotFoundError(path)
  state=torch.load(path,map_location="cpu",weights_only=False);extra=state.get("extra",{});rng=state.get("rng_state",{})
  if int(state.get("sampled_steps",-1))!=SOURCE_STEP or int(extra.get("training_seed",-1))!=seed:raise RuntimeError(f"seed{seed} source identity mismatch")
  if state.get("enabled_modules")!=["actor_lr_decay"]:raise RuntimeError(f"seed{seed} source modules mismatch")
  if not state.get("actor_optimizer",{}).get("state") or not state.get("critic_optimizer",{}).get("state"):raise RuntimeError(f"seed{seed} optimizer state incomplete")
  required=("python_random_state","numpy_random_state","torch_cpu_rng_state","torch_cuda_rng_state_all","trainer_permutation_rng_state")
  if any(key not in rng for key in required):raise RuntimeError(f"seed{seed} RNG state incomplete")
  if float(state["actor_optimizer"]["param_groups"][0]["lr"])!=1e-4:raise RuntimeError(f"seed{seed} source actor LR mismatch")
  runtime={"training_seed":seed,"training_num_envs":24,"training_smoke":False}
  validate_dawe_branch(state,env,control,runtime);validate_dawe_branch(state,env,treatment,runtime)
  sources.append({"seed":seed,"path":str(path.relative_to(ROOT)),"sha256":sha(path),"status":"PASS"})
 existing=[str(path.relative_to(ROOT)) for seed in SEEDS for path in OUTPUTS(seed) if path.exists()]
 if existing:raise RuntimeError(f"formal output directories already exist: {existing}")
 manifest=runtime_source_manifest(ROOT)
 report={"status":"READY_FOR_DAWE_FIXED10_300K_SCREEN","cuda":torch.cuda.get_device_name(0),
  "sources":sources,"config_pair_only_three_registered_differences":"PASS","expected_ppo_updates":49,
  "expected_actor_optimizer_steps":5860,"expected_critic_optimizer_steps":5860,
  "formal_outputs_absent":"6/6","evaluation_seed_range":[44_000_000,44_000_049],
  "uses_45m":False,"runtime_source_manifest_sha256":manifest["runtime_source_manifest_sha256"],
  "runtime_source_manifest_files":manifest["runtime_source_manifest_files"]}
 print(json.dumps(report,indent=2));print(report["status"])
if __name__=="__main__":main()
